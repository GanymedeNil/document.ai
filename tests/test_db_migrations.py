"""AC-4-1：迁移脚本在空库执行成功，回滚脚本可恢复上一版本。
另核对 0005 建的检查点表与 langgraph-checkpoint-postgres 自己的 setup() 建出的完全一致（列、类型、默认值、主键、索引），
且库认为全部迁移步骤已执行（8.4）。"""
from __future__ import annotations

import psycopg

from eaer.db import downgrade, migrate

TABLES = {"tenant_config", "acl_policy", "document", "document_version", "version_validity",
          "version_relation", "evidence", "lifecycle_event", "review_task", "ingest_job", "query_log", "evidence_embedding"}
CHECKPOINT_TABLES = {"checkpoint_migrations", "checkpoints", "checkpoint_blobs", "checkpoint_writes"}
TABLES_0005 = {"query_task"} | CHECKPOINT_TABLES
TRIGGERS = {"version_delete_guard", "version_update_guard", "validity_guard", "relation_guard",
            "lifecycle_event_append_only", "forced_verify_append_only"}
TRIGGERS_0005 = {"query_log_guard", "query_task_guard"}


def _state(url):
    with psycopg.connect(url) as c:
        tables = {r[0] for r in c.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
        triggers = {r[0] for r in c.execute("SELECT tgname FROM pg_trigger WHERE NOT tgisinternal")}
    return tables - {"alembic_version"}, triggers


def test_upgrade_downgrade_upgrade(empty_db_url):
    full = (TABLES | TABLES_0005, TRIGGERS | TRIGGERS_0005)
    migrate(empty_db_url)
    assert _state(empty_db_url) == full
    downgrade(empty_db_url)
    assert _state(empty_db_url) == (set(), set())
    migrate(empty_db_url)
    assert _state(empty_db_url) == full


def test_each_revision_downgrades_one_step(empty_db_url):
    """逐个版本回滚：0005 回滚去掉查询任务与检查点表；0004 回滚只去掉强制验证清单的触发器。"""
    migrate(empty_db_url)
    downgrade(empty_db_url, "0004")
    assert _state(empty_db_url) == (TABLES, TRIGGERS)
    downgrade(empty_db_url, "0003")
    assert _state(empty_db_url) == (TABLES, TRIGGERS - {"forced_verify_append_only"})
    migrate(empty_db_url)
    assert _state(empty_db_url) == (TABLES | TABLES_0005, TRIGGERS | TRIGGERS_0005)


def _checkpoint_schema(url):
    q = ("SELECT c.table_name, c.column_name, c.data_type, c.is_nullable, c.column_default "
         "FROM information_schema.columns c WHERE c.table_schema = 'public' AND c.table_name = ANY(%s) "
         "ORDER BY 1, 2")
    with psycopg.connect(url) as c:
        cols = c.execute(q, (sorted(CHECKPOINT_TABLES),)).fetchall()
        idx = c.execute("SELECT tablename, indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = ANY(%s) "
                        "ORDER BY 1, 2", (sorted(CHECKPOINT_TABLES),)).fetchall()
        versions = [r[0] for r in c.execute("SELECT v FROM checkpoint_migrations ORDER BY v")]
    return cols, idx, versions


def test_checkpoint_tables_match_the_library(empty_db_url, make_empty_db):
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    migrate(empty_db_url)
    ours = _checkpoint_schema(empty_db_url)
    lib_url = make_empty_db()
    with psycopg.connect(lib_url, autocommit=True, row_factory=dict_row) as c:
        PostgresSaver(c).setup()
    assert ours == _checkpoint_schema(lib_url)
    assert ours[2] == list(range(len(PostgresSaver.MIGRATIONS)))      # setup() 不会再执行任何步骤
