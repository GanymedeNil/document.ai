"""回答缓存与失效（开发文档 10.1 第 7 步、10.2、AC-10-4）。

query_cache：同步完成的提问结果。键是租户、授权范围哈希、索引构建编号、问题、提问日、时间参数、模式提示、模型与
Deep 开关的哈希；tenant_id、scope_hash 另存一列，读取时一并作为条件。

失效放在数据库里：下列变化都会清除该租户的全部缓存（覆盖“相关缓存”，宁可多清）——
- 版本的状态、权限策略、归档时间变化（发布、归档、取消归档、权限变更）；
- 已发布的有效期与关系的写入或更正（待审核阶段的待确认值不算）；
- 权限策略行本身被修改；租户配置（阈值、预算、强制验证清单）变化。
应用代码不必记得清缓存，任何写入路径都绕不过去。

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-13
"""
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE query_cache (
            key            text PRIMARY KEY,
            tenant_id      text NOT NULL,
            scope_hash     text NOT NULL,
            index_build_id text NOT NULL,
            result         jsonb NOT NULL,
            source_query   uuid,
            created_at     timestamptz NOT NULL,
            expires_at     timestamptz NOT NULL,
            CHECK (expires_at > created_at)
        );
        CREATE INDEX query_cache_tenant ON query_cache (tenant_id);
        CREATE INDEX query_cache_expires ON query_cache (expires_at);

        CREATE FUNCTION eaer_cache_clear_for_version(vid uuid) RETURNS void LANGUAGE sql AS $$
          DELETE FROM query_cache WHERE tenant_id = (
            SELECT d.tenant_id FROM document_version v JOIN document d ON d.id = v.document_id WHERE v.id = vid)
        $$;

        CREATE FUNCTION eaer_cache_on_version() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          PERFORM eaer_cache_clear_for_version(NEW.id);
          RETURN NULL;
        END $$;
        CREATE TRIGGER cache_on_version AFTER UPDATE OF status, acl_policy_id, archived_at ON document_version
          FOR EACH ROW WHEN (OLD.status IS DISTINCT FROM NEW.status OR OLD.acl_policy_id IS DISTINCT FROM NEW.acl_policy_id
                             OR OLD.archived_at IS DISTINCT FROM NEW.archived_at)
          EXECUTE FUNCTION eaer_cache_on_version();

        CREATE FUNCTION eaer_cache_on_version_fact() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          PERFORM eaer_cache_clear_for_version(NEW.version_id);
          RETURN NULL;
        END $$;
        CREATE TRIGGER cache_on_validity AFTER INSERT OR UPDATE ON version_validity
          FOR EACH ROW WHEN (NEW.recorded_at IS NOT NULL) EXECUTE FUNCTION eaer_cache_on_version_fact();

        CREATE FUNCTION eaer_cache_on_relation() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          PERFORM eaer_cache_clear_for_version(NEW.from_version_id);
          PERFORM eaer_cache_clear_for_version(NEW.to_version_id);
          RETURN NULL;
        END $$;
        CREATE TRIGGER cache_on_relation AFTER INSERT OR UPDATE ON version_relation
          FOR EACH ROW WHEN (NEW.recorded_at IS NOT NULL) EXECUTE FUNCTION eaer_cache_on_relation();

        CREATE FUNCTION eaer_cache_on_tenant_row() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          DELETE FROM query_cache WHERE tenant_id = CASE WHEN TG_OP = 'INSERT' THEN NEW.tenant_id ELSE OLD.tenant_id END;
          RETURN NULL;
        END $$;
        CREATE TRIGGER cache_on_acl_policy AFTER UPDATE ON acl_policy
          FOR EACH ROW EXECUTE FUNCTION eaer_cache_on_tenant_row();
        CREATE TRIGGER cache_on_tenant_config AFTER INSERT OR UPDATE OR DELETE ON tenant_config
          FOR EACH ROW EXECUTE FUNCTION eaer_cache_on_tenant_row();
    """)


def downgrade() -> None:
    op.execute(r"""
        DROP TRIGGER IF EXISTS cache_on_tenant_config ON tenant_config;
        DROP TRIGGER IF EXISTS cache_on_acl_policy ON acl_policy;
        DROP TRIGGER IF EXISTS cache_on_relation ON version_relation;
        DROP TRIGGER IF EXISTS cache_on_validity ON version_validity;
        DROP TRIGGER IF EXISTS cache_on_version ON document_version;
        DROP FUNCTION IF EXISTS eaer_cache_on_tenant_row();
        DROP FUNCTION IF EXISTS eaer_cache_on_relation();
        DROP FUNCTION IF EXISTS eaer_cache_on_version_fact();
        DROP FUNCTION IF EXISTS eaer_cache_on_version();
        DROP FUNCTION IF EXISTS eaer_cache_clear_for_version(uuid);
        DROP TABLE IF EXISTS query_cache;
    """)
