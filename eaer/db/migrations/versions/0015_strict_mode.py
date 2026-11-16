"""严格模式与宽松模式（2026-11-16，设计变更记录）。

- tenant_config.strict_mode：租户的默认回答方式。false（默认）为宽松模式：可以按证据里的单价与问题给的数量写出算式
  算合计（代码核对算术，结果标“未验证”）、可以按常识判断问题中的对象属于证据里的哪一类（单列一条，标“常识推断”）、
  证据只覆盖一部分时先答能答的部分；制度的规定与取值仍只能出自证据。true 为严格模式：只认证据原文，
  证据没写明对象属于哪一类时逐类给出规定。提问接口可按次覆盖。
- query_log.strict：这次提问实际用的模式，便于复现与审计；本迁移之前的记录为空（当时只有现在的严格写法，不回填推断）。
  已有的不可修改规则（query_log_guard：已结束的提问记录不可修改）同样覆盖这一列。

Revision ID: 0015
Revises: 0014
Create Date: 2026-11-16
"""
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        ALTER TABLE tenant_config ADD COLUMN strict_mode boolean NOT NULL DEFAULT false;
        ALTER TABLE query_log ADD COLUMN strict boolean;
    """)


def downgrade() -> None:
    op.execute(r"""
        ALTER TABLE query_log DROP COLUMN IF EXISTS strict;
        ALTER TABLE tenant_config DROP COLUMN IF EXISTS strict_mode;
    """)
