"""回答缓存的两处失效漏洞（开发文档 5.3、10.1 第 7 步、AC-10-4）。冒烟测试（eaer/smoke.py：审核通过、取消归档后
反复提问，等待索引跟上）发现。

1. 索引任务完成时没有清缓存。0006 的触发器在版本状态、权限、有效期、关系变化时清缓存，但这些变化发生在数据库里，
   索引是随后由后台任务异步跟上的（5.3）。窗口内的提问检索的还是旧索引：刚审核通过、刚取消归档的文档查不到；
   这样的回答进了缓存，索引跟上之后仍原样返回，直到过期（cache_ttl_seconds，默认 600 秒）。
   → 写入索引、索引更新（publish_index、index_update）的任务完成时，清除该版本所属租户的缓存。

2. 清除与写入的先后。一次提问先检索、后写缓存；如果在这之间发生了失效（清除时缓存里还没有这一条），旧状态算出的
   回答会在清除之后写进去，同样一直留到过期。命中时对所引证据的权限与状态核对挡得住“不该给的引用”，挡不住
   “该有的没有”（拒答、少了刚可见的文档）。
   → 每个租户一个失效序号 query_cache_epoch：所有清缓存的路径（0006 的两个函数、上面的索引任务）都把它加一。
     提问开始时读序号，写缓存时锁住序号行（FOR SHARE，正在提交的失效会先完成）再比较，变了就不写
     （eaer/orchestrator/cache.py）。

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-14
"""
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

_CLEAR_FOR_VERSION_0006 = r"""
        CREATE OR REPLACE FUNCTION eaer_cache_clear_for_version(vid uuid) RETURNS void LANGUAGE sql AS $$
          DELETE FROM query_cache WHERE tenant_id = (
            SELECT d.tenant_id FROM document_version v JOIN document d ON d.id = v.document_id WHERE v.id = vid)
        $$;
        CREATE OR REPLACE FUNCTION eaer_cache_on_tenant_row() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          DELETE FROM query_cache WHERE tenant_id = CASE WHEN TG_OP = 'INSERT' THEN NEW.tenant_id ELSE OLD.tenant_id END;
          RETURN NULL;
        END $$;
"""


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE query_cache_epoch (
            tenant_id text PRIMARY KEY,
            epoch     bigint NOT NULL DEFAULT 0 CHECK (epoch >= 0)
        );

        CREATE FUNCTION eaer_cache_invalidate(t text) RETURNS void LANGUAGE plpgsql AS $$
        BEGIN
          IF t IS NULL THEN
            RETURN;
          END IF;
          INSERT INTO query_cache_epoch (tenant_id, epoch) VALUES (t, 1)
            ON CONFLICT (tenant_id) DO UPDATE SET epoch = query_cache_epoch.epoch + 1;
          DELETE FROM query_cache WHERE tenant_id = t;
        END $$;

        CREATE OR REPLACE FUNCTION eaer_cache_clear_for_version(vid uuid) RETURNS void LANGUAGE sql AS $$
          SELECT eaer_cache_invalidate(
            (SELECT d.tenant_id FROM document_version v JOIN document d ON d.id = v.document_id WHERE v.id = vid))
        $$;
        CREATE OR REPLACE FUNCTION eaer_cache_on_tenant_row() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          PERFORM eaer_cache_invalidate(CASE WHEN TG_OP = 'INSERT' THEN NEW.tenant_id ELSE OLD.tenant_id END);
          RETURN NULL;
        END $$;

        CREATE FUNCTION eaer_cache_on_index_job() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          PERFORM eaer_cache_clear_for_version(NEW.version_id);
          RETURN NULL;
        END $$;
        CREATE TRIGGER cache_on_index_job AFTER UPDATE OF status ON ingest_job
          FOR EACH ROW WHEN (NEW.status = 'done' AND OLD.status IS DISTINCT FROM 'done'
                             AND NEW.stage IN ('publish_index', 'index_update') AND NEW.version_id IS NOT NULL)
          EXECUTE FUNCTION eaer_cache_on_index_job();
    """)


def downgrade() -> None:
    op.execute(r"""
        DROP TRIGGER IF EXISTS cache_on_index_job ON ingest_job;
        DROP FUNCTION IF EXISTS eaer_cache_on_index_job();
    """ + _CLEAR_FOR_VERSION_0006 + r"""
        DROP FUNCTION IF EXISTS eaer_cache_invalidate(text);
        DROP TABLE IF EXISTS query_cache_epoch;
    """)
