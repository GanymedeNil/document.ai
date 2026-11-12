"""审核任务增加 suggestions：入库第 3 阶段抽取出的建议值，与审核人的更正（corrections）分开存放。

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-11
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE review_task ADD COLUMN suggestions jsonb;
        ALTER TABLE review_task DROP CONSTRAINT deleted_has_no_content;
        ALTER TABLE review_task ADD CONSTRAINT deleted_has_no_content
          CHECK (status <> 'deleted' OR (version_id IS NULL AND corrections IS NULL AND suggestions IS NULL));
        CREATE INDEX ingest_job_version_stage ON ingest_job (version_id, stage);
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS ingest_job_version_stage;
        ALTER TABLE review_task DROP CONSTRAINT deleted_has_no_content;
        ALTER TABLE review_task DROP COLUMN suggestions;
        ALTER TABLE review_task ADD CONSTRAINT deleted_has_no_content
          CHECK (status <> 'deleted' OR (version_id IS NULL AND corrections IS NULL));
    """)
