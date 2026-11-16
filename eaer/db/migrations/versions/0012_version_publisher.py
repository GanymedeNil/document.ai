"""版本的出处（发布方、来源名称）：数据源随文档给出，写在证据标注里，与文档日期一起让模型分清“某来源某日的那篇”。

记录类文档（报告、新闻、纪要）的问题常按出处与日期指认文档（“TechCrunch 10 月 7 日那篇报道”），而证据标注原先只有标题、
版本与有效期；正文里即使写了出处，也只在开头一段，检索到的其余段落看不出属于哪一篇。文档日期已有 doc_date 列，这里只加出处。

- 可为空：数据源给不出时不推断；
- 有值时不能是空白、最长 200 字（服务层先去掉首尾空白、截断，数据库是第二道防线）。

Revision ID: 0012
Revises: 0011
Create Date: 2026-11-15
"""
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        ALTER TABLE document_version ADD COLUMN publisher text
          CONSTRAINT version_publisher_not_blank
          CHECK (publisher IS NULL OR char_length(btrim(publisher)) BETWEEN 1 AND 200);
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE document_version DROP COLUMN IF EXISTS publisher;")
