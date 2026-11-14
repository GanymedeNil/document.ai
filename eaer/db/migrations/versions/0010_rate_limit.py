"""频率限制的计数（开发文档 11.1 的 429 rate_limited）。

按“租户:用户”与接口类别（query、upload）每分钟一个计数；接口进程用 INSERT … ON CONFLICT 原子加一，多个接口实例
共用同一份计数（集群版横向扩展时限制照样准确）。每个用户进入新的一分钟时顺手删掉自己早于一小时的计数，表不会长大。

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-14
"""
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE rate_limit_counter (
            subject      text NOT NULL,
            kind         text NOT NULL CHECK (kind IN ('query', 'upload')),
            window_start timestamptz NOT NULL,
            count        integer NOT NULL CHECK (count > 0),
            PRIMARY KEY (subject, kind, window_start)
        );
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS rate_limit_counter;")
