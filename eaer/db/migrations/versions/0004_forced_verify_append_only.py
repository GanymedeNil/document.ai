"""强制验证清单只增不减（开发文档 8.1、11.2、AC-11-5）。

服务层只提供追加；这里是第二道防线：已有规则不能删除、修改或调换顺序，带规则的租户配置不能删除。

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-12
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        ALTER TABLE tenant_config ADD CONSTRAINT forced_verify_rules_is_array
          CHECK (jsonb_typeof(forced_verify_rules) = 'array');

        CREATE FUNCTION eaer_guard_forced_verify() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP = 'DELETE' THEN
            IF jsonb_array_length(OLD.forced_verify_rules) > 0 THEN
              RAISE EXCEPTION 'invariant: 强制验证清单只增不减，不能删除带规则的租户配置（租户 %）', OLD.tenant_id;
            END IF;
            RETURN OLD;
          END IF;
          IF jsonb_typeof(NEW.forced_verify_rules) IS DISTINCT FROM 'array' THEN
            RAISE EXCEPTION 'invariant: 强制验证清单必须是数组（租户 %）', OLD.tenant_id;
          END IF;
          IF jsonb_array_length(NEW.forced_verify_rules) < jsonb_array_length(OLD.forced_verify_rules)
             OR EXISTS (SELECT 1 FROM generate_series(0, jsonb_array_length(OLD.forced_verify_rules) - 1) AS i
                        WHERE NEW.forced_verify_rules -> i IS DISTINCT FROM OLD.forced_verify_rules -> i) THEN
            RAISE EXCEPTION 'invariant: 强制验证清单只能追加，已有规则不可删除或修改（租户 %）', OLD.tenant_id;
          END IF;
          RETURN NEW;
        END $$;

        CREATE TRIGGER forced_verify_append_only BEFORE UPDATE OR DELETE ON tenant_config
          FOR EACH ROW EXECUTE FUNCTION eaer_guard_forced_verify();
    """)


def downgrade() -> None:
    op.execute("""
        DROP TRIGGER forced_verify_append_only ON tenant_config;
        DROP FUNCTION eaer_guard_forced_verify();
        ALTER TABLE tenant_config DROP CONSTRAINT forced_verify_rules_is_array;
    """)
