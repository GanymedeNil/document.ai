"""提问记录保存模型的思考过程（2026-11-16，设计变更记录）。

生成服务开启思考（EAER_GEN_THINKING=enabled）时，OpenAI 兼容接口在 message.reasoning_content 里给出思考文字。
编排器按步骤（prepare、first、escalate、verify、assemble）收集，随回答写进 query_log.reasoning，会话接口与提问接口
原样返回，界面上放在回答前的“思考过程”里。没有开启思考、或模型不给思考文字时为空数组。

- reasoning：[{"step": 节点名, "text": 思考文字}]，按发生顺序。只追加到 query_log，不进 trace（13.3：原文不进 trace）。
  已有的不可修改规则（query_log_guard：已结束的提问记录不可修改）同样覆盖这一列。

Revision ID: 0014
Revises: 0013
Create Date: 2026-11-16
"""
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        ALTER TABLE query_log ADD COLUMN reasoning jsonb NOT NULL DEFAULT '[]'::jsonb
          CONSTRAINT query_log_reasoning_is_array CHECK (jsonb_typeof(reasoning) = 'array');
    """)


def downgrade() -> None:
    op.execute(r"""
        ALTER TABLE query_log DROP COLUMN IF EXISTS reasoning;
    """)
