"""证据向量缓存：按“证据 + 模型”保存向量。

取消归档、权限变更、换分词器重建索引时直接复用，只有换 Embedding 模型才需要重新计算，
避免按量计费的线上模型被重复收费。证据删除时一并删除。

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-11
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE evidence_embedding (
          evidence_id text NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
          model       text NOT NULL,
          vector      real[] NOT NULL CHECK (cardinality(vector) > 0),
          created_at  timestamptz NOT NULL DEFAULT now(),
          PRIMARY KEY (evidence_id, model)
        );
    """)


def downgrade() -> None:
    op.execute("DROP TABLE evidence_embedding;")
