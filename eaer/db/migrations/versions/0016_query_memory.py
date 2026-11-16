"""多轮记忆：提问记录保存结合上文改写后的问题（2026-11-16，设计变更记录；eaer/orchestrator/memory.py）。

追问（“那二类呢？”）结合同一会话里之前的提问改写成完整的问题，之后的检索、生成与核对都用改写后的问题；
query_log.question 仍是用户的原话，改写后的问题记在 standalone_question，会话接口与提问接口原样返回。
不是追问、或问题本身完整时为空。已有的不可修改规则（query_log_guard：已结束的提问记录不可修改）同样覆盖这一列。

Revision ID: 0016
Revises: 0015
Create Date: 2026-11-16
"""
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        ALTER TABLE query_log ADD COLUMN standalone_question text
          CONSTRAINT query_log_standalone_not_blank CHECK (standalone_question IS NULL OR btrim(standalone_question) <> '');
    """)


def downgrade() -> None:
    op.execute(r"""
        ALTER TABLE query_log DROP COLUMN IF EXISTS standalone_question;
    """)
