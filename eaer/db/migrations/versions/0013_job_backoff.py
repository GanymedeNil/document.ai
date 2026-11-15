"""后台任务失败后的退避与对账补写的来源（开发文档 5.4、6.2）。

2026-11-15 MultiHop-RAG 重跑时，本地 Embedding 服务在内存预算下短时间连续返回 503：任务出错后立即重新排队，
几秒内用完全部尝试次数，publish_index 与 index_update 都标为失败，版本可搜索、索引里却没有它，也没有什么会补写。

- not_before：失败后重新排队的任务在这个时刻之前不领取（Worker 按 30 秒、2 分钟、10 分钟退避）；为空表示随时可领取。
  领取时只跳过还没到时间的任务，其他任务照常按阶段的优先级领取（eaer/domain/jobs.py）。
- origin：任务由谁补排。目前只有 'reconcile'（对账发现索引与数据库不一致后补的 index_update，eaer/ingestion/repair.py），
  监控指标据此统计补写了多少、成没成；流水线与生命周期排的任务为空。

Revision ID: 0013
Revises: 0012
Create Date: 2026-11-15
"""
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        ALTER TABLE ingest_job ADD COLUMN not_before timestamptz;
        ALTER TABLE ingest_job ADD COLUMN origin text
          CONSTRAINT ingest_job_origin_known CHECK (origin IN ('reconcile'));
    """)


def downgrade() -> None:
    op.execute(r"""
        ALTER TABLE ingest_job DROP COLUMN IF EXISTS origin;
        ALTER TABLE ingest_job DROP COLUMN IF EXISTS not_before;
    """)
