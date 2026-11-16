"""监控指标（开发文档 13.3）：一次输出一份 JSON，供监控系统定时采集。

    python -m eaer.metrics [--since-minutes 60] [--opensearch http://… --alias eaer]

- 提问：各模式的请求量与 P95 耗时、模式升级比例（mode_path 多于一段）、拒答比例、失败数、验证结论分布
  （回答里各结论的 supported / unverified / unsupported 数）——取自 query_log，按提问时间统计最近一段；
- 入库：各阶段排队中、执行中、失败、失败后退避中的任务数，最早排队任务的等待时间，索引更新延迟（eaer/ingestion/stats.py）；
- 对账补写（index_repair）：后台任务进程对账后补排的索引任务，最近一段补了多少、完成与失败各多少（eaer/ingestion/repair.py）；
- 索引对账差异数（AC-5-2）：给出 --opensearch 时按租户对账，数据库中可搜索的版本与索引中的版本之差。
只输出数量与比例，不含问题、回答等原文（10.2）。环境变量 EAER_DATABASE_URL。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import psycopg
from psycopg.rows import dict_row


def _round(v, digits: int) -> float | None:
    return None if v is None else round(float(v), digits)


def query_metrics(conn: psycopg.Connection, since: datetime) -> dict:
    rows = conn.execute(
        "SELECT mode, count(*) AS n, "
        "  percentile_cont(0.95) WITHIN GROUP (ORDER BY (cost->>'seconds')::float) AS p95, "
        "  avg((cardinality(mode_path) > 1)::int) AS escalated, avg(abstained::int) AS abstained "
        "FROM query_log WHERE asked_at >= %s AND status = 'completed' GROUP BY ROLLUP (mode) ORDER BY mode NULLS FIRST",
        (since,)).fetchall()
    out: dict = {"completed": 0, "p95_seconds": None, "escalation_rate": None, "abstain_rate": None, "by_mode": {}}
    for r in rows:
        vals = {"n": r["n"], "p95_seconds": _round(r["p95"], 3), "escalation_rate": _round(r["escalated"], 4),
                "abstain_rate": _round(r["abstained"], 4)}
        if r["mode"] is None:
            out.update({"completed": vals["n"], **{k: v for k, v in vals.items() if k != "n"}})
        else:
            out["by_mode"][r["mode"]] = vals
    other = conn.execute("SELECT status, count(*) AS n FROM query_log WHERE asked_at >= %s AND status <> 'completed' "
                         "GROUP BY status", (since,)).fetchall()
    out.update({r["status"]: r["n"] for r in other})
    verdicts = conn.execute(
        "SELECT c->>'status' AS status, count(*) AS n FROM query_log q, jsonb_array_elements(q.claims) c "
        "WHERE q.asked_at >= %s AND q.status = 'completed' GROUP BY 1", (since,)).fetchall()
    out["claims"] = {r["status"]: r["n"] for r in verdicts}
    return out


def collect(conn: psycopg.Connection, now: datetime, since: datetime, *, index=None) -> dict:
    from eaer.ingestion.stats import TARGET_P95_SECONDS, index_latency, queue_stats, repair_stats

    out = {"window": {"since": since.isoformat(), "until": now.isoformat()},
           "queries": query_metrics(conn, since), "ingest_queue": queue_stats(conn, now),
           "index_latency": {**index_latency(conn, since), "target_p95": TARGET_P95_SECONDS},
           "index_repair": repair_stats(conn, since)}
    if index is not None:
        from eaer.retrieval.reconcile import reconcile
        tenants = [r["tenant_id"] for r in conn.execute("SELECT tenant_id FROM tenant_config ORDER BY tenant_id")]
        out["reconcile"] = {t: {k: len(v) for k, v in reconcile(conn, index, t).items()} for t in tenants}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eaer.metrics", description="监控指标（开发文档 13.3）")
    p.add_argument("--since-minutes", type=float, default=60.0)
    p.add_argument("--opensearch", help="给出时一并做索引对账")
    p.add_argument("--alias", default=os.environ.get("EAER_INDEX_ALIAS", "eaer"))
    args = p.parse_args(argv)
    index = None
    if args.opensearch:
        from eaer.retrieval.opensearch import OpenSearchIndex, connect
        index = OpenSearchIndex(connect(args.opensearch), alias=args.alias)
    now = datetime.now(timezone.utc)
    with psycopg.connect(os.environ["EAER_DATABASE_URL"], autocommit=True, row_factory=dict_row) as conn:
        print(json.dumps(collect(conn, now, now - timedelta(minutes=args.since_minutes), index=index),
                         ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
