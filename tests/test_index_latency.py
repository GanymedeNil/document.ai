"""AC-5-6 索引更新延迟（开发文档 5.3）：压测下统计，P95 不超过 60 秒（初值）。

- 统计口径（eaer/ingestion/stats.py）：已完成的索引任务的“完成时间 − 排队时间”，总体与按阶段的 P50、P95、最大值；
  队列长度、最早排队任务的等待时间、失败数（13.3 的监控指标）；`python -m eaer.ingestion stats` 输出 JSON；
- 压测（eaer/evalrun/indexlag.py）：突发批量审核、持续的归档 / 取消归档 / 权限变更 / 有效期更正、陆续上传的新文档、
  并发查询，后台任务线程执行与生产相同的 Worker 循环；结束后任务全部完成、没有失败、对账差异为 0、查询没有错误，
  P95 在目标之内。这里用真实的 OpenSearch 与测试替身向量化、规模较小，只验证压测与统计本身；
  真实规模与真实 Embedding 服务的数字由 `python -m eaer.evalrun.indexlag` 产生，记入 CLAUDE.md。
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from eaer.ingestion.stats import index_latency, queue_stats

OS_URL = os.environ.get("EAER_TEST_OPENSEARCH_URL")
T = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)


def job(conn, stage, status, waited: float, *, created=T):
    conn.execute("INSERT INTO ingest_job (id, version_id, stage, status, created_at, updated_at) "
                 "VALUES (%s, NULL, %s, %s, %s, %s)",
                 (uuid.uuid4(), stage, status, created, created + timedelta(seconds=waited)))


def test_latency_statistics(conn):
    for s in range(1, 21):                       # publish_index：1..20 秒
        job(conn, "publish_index", "done", s)
    for s in (2, 4):
        job(conn, "index_update", "done", s)
    job(conn, "parse_text", "done", 500)         # 不是索引任务，不计入
    job(conn, "publish_index", "done", 900, created=T - timedelta(hours=2))   # 早于统计起点
    job(conn, "index_update", "queued", 0, created=T - timedelta(seconds=30))
    job(conn, "publish_index", "failed", 3)
    lat = index_latency(conn, T - timedelta(minutes=1))
    assert lat["n"] == 22 and lat["max"] == 20.0
    assert lat["by_stage"]["publish_index"] == {"n": 20, "p50": 10.5, "p95": 19.05, "max": 20.0}
    assert lat["by_stage"]["index_update"] == {"n": 2, "p50": 3.0, "p95": 3.9, "max": 4.0}
    assert index_latency(conn)["max"] == 900.0
    q = queue_stats(conn, T + timedelta(seconds=10))
    assert q["queued"] == {"index_update": 1} and q["failed"] == {"publish_index": 1}
    assert q["oldest_queued_seconds"] == {"index_update": 40.0}


def test_empty_queue(conn):
    assert index_latency(conn) == {"n": 0, "p50": None, "p95": None, "max": None, "by_stage": {}}
    assert queue_stats(conn, T) == {"queued": {}, "running": {}, "failed": {}, "waiting_retry": {},
                                    "oldest_queued_seconds": {}}


def test_index_jobs_are_claimed_first(conn, tmp_path):
    """优先级（eaer/domain/jobs.py）：先排队的深度解析、解析不会挡住后排队的索引任务；同一优先级按排队先后。"""
    from eaer.domain.jobs import PRIORITY
    from eaer.ingestion.pipeline import Ingestion, Worker
    from eaer.storage import LocalFileStore

    order = ["parse_deep", "parse_text", "publish_index", "index_update", "publish_index"]
    for k, stage in enumerate(order):
        conn.execute("INSERT INTO ingest_job (id, version_id, stage, status, priority, created_at, updated_at) "
                     "VALUES (%s, NULL, %s, 'queued', %s, %s, %s)",
                     (uuid.uuid4(), stage, PRIORITY[stage], T + timedelta(seconds=k), T + timedelta(seconds=k)))
    w = Worker(Ingestion(conn, LocalFileStore(tmp_path), lambda: T + timedelta(minutes=1)))
    w.handlers |= {"publish_index": lambda v: None, "index_update": lambda v: None}
    claimed = []
    while (j := w.claim()) is not None:
        claimed.append((j["stage"], j["created_at"]))
    assert [s for s, _ in claimed] == ["index_update", "publish_index", "publish_index", "parse_text", "parse_deep"]
    assert claimed[1][1] < claimed[2][1]


def test_enqueued_jobs_carry_their_priority(conn, tmp_path):
    from eaer.domain.jobs import PRIORITY
    from eaer.ingestion.pipeline import Ingestion, Worker
    from eaer.lifecycle.service import Lifecycle
    from eaer.review.service import Review
    from eaer.storage import LocalFileStore

    store = LocalFileStore(tmp_path)
    ing, lc = Ingestion(conn, store), Lifecycle(conn)
    lc.ensure_tenant("t1")
    r = ing.receive("t1", "<h1>办法</h1><p>第一条　住宿上限为每晚650元。</p>".encode(), "a.html",
                    acl_policy_id=lc.create_acl_policy("t1", ["R_STAFF"]))
    Worker(ing).run_until_idle()
    task = conn.execute("SELECT id FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()["id"]
    assert Review(conn, lc, store).decide("t1", [{"task_id": str(task), "decision": "approve",
                                                 "corrections": {"doc_type": "normative"}}], "rv")[0]["ok"]
    lc.archive_document(r.document_id, "rv")
    rows = conn.execute("SELECT stage, priority FROM ingest_job").fetchall()
    assert rows and all(x["priority"] == PRIORITY[x["stage"]] for x in rows), rows
    assert {x["stage"] for x in rows} >= {"parse_text", "classify_extract", "publish_index", "parse_deep", "index_update"}


def test_stats_command(conn, db_url, monkeypatch, capsys):
    from eaer.ingestion.__main__ import main

    job(conn, "publish_index", "done", 3, created=datetime.now(timezone.utc) - timedelta(minutes=5))
    monkeypatch.setenv("EAER_DATABASE_URL", db_url)
    assert main(["stats", "--since-minutes", "60"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["index_latency"]["n"] == 1 and out["index_latency_target_p95"] == 60.0


@pytest.mark.skipif(not OS_URL, reason="未设置 EAER_TEST_OPENSEARCH_URL")
def test_load_run_meets_the_target(db_url, tmp_path):
    from eaer.evalrun.indexlag import report, run_load, verdict
    from eaer.ingestion.indexer import fresh_index
    from eaer.retrieval.embed import HashingEmbedder
    from eaer.retrieval.opensearch import OpenSearchIndex, connect
    from eaer.storage import LocalFileStore

    alias = f"eaer-lag-{uuid.uuid4().hex[:8]}"
    with fresh_index(connect(OS_URL), alias, build_id="lag", dims=64):
        r = run_load(db_url, LocalFileStore(tmp_path), lambda: OpenSearchIndex(connect(OS_URL), alias=alias),
                     HashingEmbedder(64), docs=40, workers=2, query_threads=2, event_rate=5, uploads=5, paras=6,
                     min_seconds=5, timeout=300)
    assert not verdict(r, 60.0), report(r, 60.0)
    lat = r["index_latency"]
    assert lat["by_stage"]["publish_index"]["n"] == 45                  # 40 份突发审核 + 5 份新上传
    assert lat["by_stage"]["index_update"]["n"] >= 1 and sum(r["events"].values()) >= 1
    assert r["queries"] > 0 and r["uploads"] == 5
