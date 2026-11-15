"""入库队列与索引更新延迟的统计（开发文档 5.3、13.3 的监控指标、AC-5-6）。

索引更新延迟 = 索引任务完成时间 − 任务排队时间（ingest_job.updated_at − created_at，状态为 done）。
索引写入时同步刷新（refresh），任务完成即可检索，所以这就是“事件发生到检索可见”的时间。
生命周期事件每次都单独排队一个任务；深度解析完成后的索引更新在同一版本已有排队中的任务时不再排队
（pipeline._enqueue），它的延迟由更早排队的那个任务计入，统计只会偏大、不会偏小。
"""
from __future__ import annotations

from datetime import datetime

import psycopg

INDEX_STAGES = ("publish_index", "index_update")
TARGET_P95_SECONDS = 60.0          # 5.3 的初值


def index_latency(conn: psycopg.Connection, since: datetime | None = None, *,
                  stages: tuple[str, ...] = INDEX_STAGES) -> dict:
    """已完成的索引任务的延迟（秒）：总体与按阶段的 n、P50、P95、最大值。since 按排队时间过滤。"""
    rows = conn.execute(
        "SELECT stage, count(*) AS n, "
        "  percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch FROM updated_at - created_at)) AS p50, "
        "  percentile_cont(0.95) WITHIN GROUP (ORDER BY extract(epoch FROM updated_at - created_at)) AS p95, "
        "  max(extract(epoch FROM updated_at - created_at)) AS max "
        "FROM ingest_job WHERE status = 'done' AND stage = ANY(%s) AND (%s::timestamptz IS NULL OR created_at >= %s) "
        "GROUP BY GROUPING SETS ((stage), ()) ORDER BY stage NULLS FIRST",
        (list(stages), since, since)).fetchall()
    out: dict = {"n": 0, "p50": None, "p95": None, "max": None, "by_stage": {}}
    for r in rows:
        vals = {"n": r["n"], **{k: None if r[k] is None else round(float(r[k]), 3) for k in ("p50", "p95", "max")}}
        if r["stage"] is None:
            out.update(vals)
        else:
            out["by_stage"][r["stage"]] = vals
    return out


def queue_stats(conn: psycopg.Connection, now: datetime) -> dict:
    """入库队列：各阶段排队中、执行中的数量，最早一个排队任务已等待的秒数，失败数（13.3）；
    waiting_retry 是排队中的任务里失败后正在退避、还没到再次领取时间的（迁移 0013），已计入 queued。"""
    rows = conn.execute(
        "SELECT stage, status, count(*) AS n, min(created_at) AS oldest, "
        "  count(*) FILTER (WHERE not_before > %s) AS waiting FROM ingest_job "
        "WHERE status <> 'done' GROUP BY stage, status ORDER BY stage, status", (now,)).fetchall()
    out: dict = {"queued": {}, "running": {}, "failed": {}, "waiting_retry": {}, "oldest_queued_seconds": {}}
    for r in rows:
        out[r["status"]][r["stage"]] = r["n"]
        if r["status"] == "queued":
            out["oldest_queued_seconds"][r["stage"]] = round((now - r["oldest"]).total_seconds(), 3)
            if r["waiting"]:
                out["waiting_retry"][r["stage"]] = r["waiting"]
    return out


def repair_stats(conn: psycopg.Connection, since: datetime | None = None) -> dict:
    """对账补写的索引任务（origin = 'reconcile'，eaer/ingestion/repair.py）：since 之后补了多少、各状态多少。
    failed 不为 0 说明有版本补写后仍写不进索引，需要人工查看任务的 last_error。"""
    rows = conn.execute(
        "SELECT status, count(*) AS n FROM ingest_job WHERE origin = 'reconcile' "
        "AND (%s::timestamptz IS NULL OR created_at >= %s) GROUP BY status", (since, since)).fetchall()
    by = {r["status"]: r["n"] for r in rows}
    return {"enqueued": sum(by.values()), **{s: by.get(s, 0) for s in ("queued", "running", "done", "failed")}}
