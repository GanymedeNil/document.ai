"""异步查询：会话检查点与查询任务（开发文档 8.4、8.5、AC-8-6）。

1. LangGraph 的 PostgreSQL 检查点表（langgraph-checkpoint-postgres 3.2.0）。库自带的 setup() 用
   CREATE INDEX CONCURRENTLY，不能在迁移事务里执行，运行时建表又违反“表结构只经迁移”的约定，所以这里按该版本的
   10 个迁移步骤原样建表（索引不加 CONCURRENTLY：表是新建的），并在 checkpoint_migrations 记下 0–9 已执行，
   库的 setup() 因此不会再做任何事。升级该库时若它增加了迁移步骤，写新的迁移文件补上；启动时
   eaer.orchestrator.tasks.checkpointer 核对版本，不一致即报错。
2. query_task：一次提问的执行任务，与 query_log 一一对应（同一 id）。保存图的输入（prepare 的结果）、
   提问人身份（恢复时重新解析授权范围）、进度、取消请求与租约。
3. 不变量：已结束（completed、cancelled、failed）的提问记录与任务不可再改；运行中的任务必须有租约，
   排队与已结束的任务不能有租约；提问记录的身份字段（提问人、会话、问题、时间）不可修改。

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-13
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

CHECKPOINT_MIGRATIONS = 10       # langgraph-checkpoint-postgres 3.2.0 的 MIGRATIONS 条数


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE checkpoint_migrations (v INTEGER PRIMARY KEY);
        CREATE TABLE checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT,
            checkpoint JSONB NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}',
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        );
        CREATE TABLE checkpoint_blobs (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            channel TEXT NOT NULL,
            version TEXT NOT NULL,
            type TEXT NOT NULL,
            blob BYTEA,
            PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
        );
        CREATE TABLE checkpoint_writes (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            channel TEXT NOT NULL,
            type TEXT,
            blob BYTEA NOT NULL,
            task_path TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
        );
        CREATE INDEX checkpoints_thread_id_idx ON checkpoints(thread_id);
        CREATE INDEX checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);
        CREATE INDEX checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);
        INSERT INTO checkpoint_migrations (v) SELECT generate_series(0, 9);

        CREATE TABLE query_task (
            id                  uuid PRIMARY KEY REFERENCES query_log(id),
            tenant_id           text NOT NULL,
            session_id          uuid NOT NULL,
            user_id             text NOT NULL,
            identity            jsonb NOT NULL,      -- user、tenant、roles：恢复时重新解析授权范围
            state               jsonb NOT NULL,      -- 图的输入（时间点、Router 决定等），尚无检查点时从这里开始
            status              text NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'cancelled', 'failed')),
            background          boolean NOT NULL DEFAULT false,   -- 超过同步时限、已转为后台任务
            progress            jsonb NOT NULL DEFAULT '{}',
            cancel_requested_at timestamptz,
            locked_by           text,
            locked_until        timestamptz,
            attempts            int NOT NULL DEFAULT 0,
            last_error          text,
            created_at          timestamptz NOT NULL,
            updated_at          timestamptz NOT NULL,
            CONSTRAINT query_task_lease CHECK ((status = 'running') = (locked_until IS NOT NULL AND locked_by IS NOT NULL))
        );
        CREATE INDEX query_task_open ON query_task (created_at) WHERE status IN ('queued', 'running');
        CREATE INDEX query_task_session ON query_task (session_id);

        CREATE FUNCTION eaer_guard_query_log() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF OLD.status <> 'running' THEN
            RAISE EXCEPTION 'invariant: 已结束的提问记录不可修改（%，状态 %）', OLD.id, OLD.status;
          END IF;
          IF NEW.id <> OLD.id OR NEW.tenant_id <> OLD.tenant_id OR NEW.session_id <> OLD.session_id
             OR NEW.user_id <> OLD.user_id OR NEW.asked_at <> OLD.asked_at OR NEW.question <> OLD.question THEN
            RAISE EXCEPTION 'invariant: 提问记录的提问人、会话、问题与时间不可修改（%）', OLD.id;
          END IF;
          RETURN NEW;
        END $$;
        CREATE TRIGGER query_log_guard BEFORE UPDATE ON query_log
          FOR EACH ROW EXECUTE FUNCTION eaer_guard_query_log();

        CREATE FUNCTION eaer_guard_query_task() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF OLD.status IN ('completed', 'cancelled', 'failed') THEN
            RAISE EXCEPTION 'invariant: 已结束的查询任务不可修改（%，状态 %）', OLD.id, OLD.status;
          END IF;
          IF NEW.id <> OLD.id OR NEW.tenant_id <> OLD.tenant_id OR NEW.session_id <> OLD.session_id
             OR NEW.user_id <> OLD.user_id OR NEW.identity <> OLD.identity THEN
            RAISE EXCEPTION 'invariant: 查询任务的提问人与会话不可修改（%）', OLD.id;
          END IF;
          RETURN NEW;
        END $$;
        CREATE TRIGGER query_task_guard BEFORE UPDATE ON query_task
          FOR EACH ROW EXECUTE FUNCTION eaer_guard_query_task();
    """)


def downgrade() -> None:
    op.execute(r"""
        DROP TRIGGER IF EXISTS query_task_guard ON query_task;
        DROP FUNCTION IF EXISTS eaer_guard_query_task();
        DROP TRIGGER IF EXISTS query_log_guard ON query_log;
        DROP FUNCTION IF EXISTS eaer_guard_query_log();
        DROP TABLE IF EXISTS query_task;
        DROP TABLE IF EXISTS checkpoint_writes, checkpoint_blobs, checkpoints, checkpoint_migrations;
    """)
