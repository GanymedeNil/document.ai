"""解析结果的暂存（开发文档 6.2 第 2、6 阶段）：文本解析的结果留给深度解析复用，不再对原件解析两遍。

扫描件与图片的 OCR 很慢，深度解析（表格、图表）原先要把原件重新解析一遍，OCR 做两次（入库吞吐测量：100 页 PNG
文本解析 245 秒、深度解析 246 秒）。这里把文本解析得到的块（标题、段落、表格、图片位置与章节）存下来，图片本身另存一表；
深度解析时解析器版本相同就直接用，完成后删除。

- 与版本同生同灭：外键 ON DELETE CASCADE。删除待审核版本时整行版本被删，暂存随之删除，不留内容（4.4、6.5）；
- 只是加速，不是事实来源：没有暂存（旧数据、解析器升级、深度解析已做完后再排队）就照原样重新解析。

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-14
"""
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE parse_cache (
            version_id uuid PRIMARY KEY REFERENCES document_version(id) ON DELETE CASCADE,
            parser     text NOT NULL,
            title      text,
            blocks     jsonb NOT NULL,
            created_at timestamptz NOT NULL
        );
        CREATE TABLE parse_cache_image (
            version_id  uuid NOT NULL REFERENCES parse_cache(version_id) ON DELETE CASCADE,
            block_order integer NOT NULL CHECK (block_order >= 0),
            mime        text NOT NULL CHECK (mime IN ('image/png', 'image/jpeg', 'image/gif', 'image/webp')),
            data        bytea NOT NULL,
            PRIMARY KEY (version_id, block_order)
        );
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS parse_cache_image; DROP TABLE IF EXISTS parse_cache;")
