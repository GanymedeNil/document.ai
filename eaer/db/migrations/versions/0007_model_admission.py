"""模型准入记录（开发文档 12.3、AC-12-3）。

model_admission：每次准入检查的结果，按租户与生成模型记录 Verified、Deep 是否准入，以及能力矩阵（各项指标、所用门槛、
标定出的阈值、评测集规模与标定集和评测集的划分）。只追加：准入结论是审计依据，不能事后改写或删除。
写入新的准入结论会清除该租户的回答缓存（开关变了，缓存里的回答可能是按旧开关算的）。

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-13
"""
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE model_admission (
            id          uuid PRIMARY KEY,
            tenant_id   text NOT NULL,
            model_id    text NOT NULL,
            verified_ok boolean NOT NULL,
            deep_ok     boolean NOT NULL,
            report      jsonb NOT NULL,
            created_at  timestamptz NOT NULL
        );
        CREATE INDEX model_admission_latest ON model_admission (tenant_id, model_id, created_at DESC);

        CREATE FUNCTION eaer_guard_model_admission() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          RAISE EXCEPTION 'invariant: 准入记录只能追加，不能修改或删除（%）', OLD.id;
        END $$;
        CREATE TRIGGER model_admission_append_only BEFORE UPDATE OR DELETE ON model_admission
          FOR EACH ROW EXECUTE FUNCTION eaer_guard_model_admission();

        CREATE TRIGGER cache_on_model_admission AFTER INSERT ON model_admission
          FOR EACH ROW EXECUTE FUNCTION eaer_cache_on_tenant_row();
    """)


def downgrade() -> None:
    op.execute(r"""
        DROP TRIGGER IF EXISTS cache_on_model_admission ON model_admission;
        DROP TRIGGER IF EXISTS model_admission_append_only ON model_admission;
        DROP FUNCTION IF EXISTS eaer_guard_model_admission();
        DROP TABLE IF EXISTS model_admission;
    """)
