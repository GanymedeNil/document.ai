"""索引更新延迟压测（开发文档 5.3、AC-5-6）：在压力下统计索引任务从排队到完成（检索可见）的 P95，目标 ≤ 60 秒（初值）。

    EAER_ENV=eval python -m eaer.evalrun.indexlag --admin-url postgresql://… --opensearch http://localhost:9200 \\
        --docs 500 --workers 2 --query-threads 4 --event-rate 2 --uploads 50

压力同时来自四处，与生产中索引任务会遇到的竞争相同：
- 审核：默认一次批量审核通过 --docs 份（突发，AC-6-5 的规模）；--approve-rate 改为每秒审核通过这么多份（持续负载）。
  每份产生一个写入索引任务和一个深度解析任务；
- 持续变更：审核开始后按 --event-rate 每秒发起归档、取消归档、权限变更、有效期更正，各产生索引更新任务，
  持续到批量审核的索引任务全部完成为止（至少 --min-seconds 秒）；
- 新文档：同时陆续上传 --uploads 份，解析完成即审核通过，它们的解析任务与索引任务抢同一批后台进程；
- 查询：--query-threads 个线程持续检索（Embedding 加 OpenSearch），与索引写入抢资源。
后台进程是 --workers 个线程，每个线程与生产的后台任务进程执行同样的 Worker 循环（领取、执行、写回）。

延迟按数据库中的任务记录统计（eaer/ingestion/stats.py）。同时报告：任务失败数、结束后的对账差异（AC-5-2）、
查询次数与 P95、错误数。Embedding 用 EAER_EMBED_* 配置的服务（与生产相同）；--hashing 改用测试替身，
这时的数字只说明流程，不是产品结论。文档是按模板生成的制度文本（每份 --paras 条），不涉及任何评测数据集。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

TENANT = "t1"
ACTOR = "indexlag"
TOPICS = ["差旅住宿", "会议组织", "办公用品采购", "车辆使用", "加班审批", "培训报销", "印章管理", "档案借阅",
          "资产盘点", "合同审批", "来访接待", "信息安全"]


def document(i: int, paras: int) -> tuple[str, bytes]:
    topic = TOPICS[i % len(TOPICS)]
    title = f"{topic}管理办法（第{i}号）"
    body = [f"第一条　为规范{topic}工作，制定本办法。"]
    for k in range(2, paras):
        body.append(f"第{_cn(k)}条　{topic}事项第{k}项的标准为每次{(i * 37 + k * 11) % 900 + 100}元，"
                    f"由第{k % 5 + 1}级负责人审批，审批期限为{k % 7 + 1}个工作日。")
    body.append(f"第{_cn(paras)}条　本办法自2026年{i % 12 + 1}月1日起施行。")
    return f"doc-{i:05d}.html", (f"<h1>{title}</h1>" + "".join(f"<p>{p}</p>" for p in body)).encode("utf-8")


def _cn(n: int) -> str:
    digits = "零一二三四五六七八九"
    if n < 10:
        return digits[n]
    tens, ones = divmod(n, 10)
    return ("" if tens == 1 else digits[tens]) + "十" + (digits[ones] if ones else "")


@dataclass
class Load:
    queries: list[float] = field(default_factory=list)
    query_errors: list[str] = field(default_factory=list)
    events: dict[str, int] = field(default_factory=dict)
    event_errors: list[str] = field(default_factory=list)
    max_queue: int = 0
    uploads: int = 0


def run_load(db_url: str, store, make_index, embedder, *, docs: int, workers: int, query_threads: int,
             event_rate: float, uploads: int, paras: int = 8, min_seconds: float = 30.0, timeout: float = 1800.0,
             approve_rate: float | None = None, seed: int = 7, log=lambda s: None) -> dict:
    """在 db_url 指向的空库上压测；make_index() 每次返回一个新的 SearchIndex 句柄（各线程各用一个）。"""
    from eaer.domain.identity import Scope
    from eaer.ingestion.indexer import Indexer
    from eaer.ingestion.pipeline import Ingestion, Worker
    from eaer.ingestion.stats import index_latency, queue_stats
    from eaer.lifecycle.service import Lifecycle
    from eaer.retrieval.reconcile import reconcile
    from eaer.retrieval.service import Retriever
    from eaer.review.service import Review

    rnd = random.Random(seed)

    def connect():
        return psycopg.connect(db_url, autocommit=True, row_factory=dict_row)

    main = connect()
    lc = Lifecycle(main)
    lc.ensure_tenant(TENANT)
    acl = lc.create_acl_policy(TENANT, ["R_STAFF"])
    ing = Ingestion(main, store)

    # 准备：上传并解析（不计入延迟）
    t_prep = time.monotonic()
    received = [ing.receive(TENANT, data, name, acl_policy_id=acl) for name, data in
                (document(i, paras) for i in range(docs))]
    Worker(ing).run_until_idle()
    log(f"准备完成：{docs} 份文档上传并解析，{time.monotonic() - t_prep:.1f} 秒")
    tasks = {r["version_id"]: r["id"] for r in main.execute("SELECT id, version_id FROM review_task WHERE status = 'open'")}

    stop_workers, stop_load = threading.Event(), threading.Event()
    load = Load()
    lock = threading.Lock()

    def worker_loop():
        c = connect()
        w = Worker(Ingestion(c, store), Indexer(c, make_index(), embedder, "lag"))
        while not stop_workers.is_set():
            try:
                if not w.run_once():
                    time.sleep(0.2)
            except psycopg.Error as e:          # 连接级错误：记下并继续（任务本身的错误由 Worker 记入任务）
                log(f"后台线程数据库错误：{e}")
                time.sleep(0.5)
        c.close()

    questions = [f"{t}事项第{k}项的标准是多少" for t in TOPICS for k in range(2, paras)]

    def query_loop():
        c = connect()
        r = Retriever(c, make_index(), embedder)
        scope = Scope(TENANT, frozenset({"R_STAFF"}))
        mine = random.Random(rnd.random())
        while not stop_load.is_set():
            q = questions[mine.randrange(len(questions))]
            t = time.monotonic()
            try:
                r.search(scope, q, today=date.today())
                with lock:
                    load.queries.append(time.monotonic() - t)
            except Exception as e:  # noqa: BLE001 — 统计错误数
                with lock:
                    load.query_errors.append(f"{type(e).__name__}: {e}")
        c.close()

    published: list[uuid.UUID] = []          # 已审核通过的文档（事件只作用于它们）
    archived: set[uuid.UUID] = set()

    def event_loop():
        c = connect()
        lce = Lifecycle(c)
        interval = 1.0 / event_rate if event_rate > 0 else None
        while interval and not stop_load.is_set():
            time.sleep(interval)
            with lock:
                pool = [d for d in published if d not in archived]
                back = list(archived)
            kind = rnd.choice(["archive", "unarchive", "acl", "validity"] if back else ["archive", "acl", "validity"])
            try:
                if kind == "unarchive":
                    d = rnd.choice(back)
                    lce.unarchive_document(d, ACTOR)
                    with lock:
                        archived.discard(d)
                elif not pool:
                    continue
                elif kind == "archive":
                    d = rnd.choice(pool)
                    lce.archive_document(d, ACTOR)
                    with lock:
                        archived.add(d)
                elif kind == "acl":
                    lce.change_acl(rnd.choice(pool), rnd.choice([["R_STAFF"], ["R_STAFF", "R_FIN"]]), ACTOR)
                else:
                    v = c.execute("SELECT id FROM document_version WHERE document_id = %s AND status = 'searchable' "
                                  "LIMIT 1", (rnd.choice(pool),)).fetchone()
                    if v is None:
                        continue
                    lce.correct_validity(v["id"], ACTOR, valid_from=date(2026, rnd.randint(1, 12), 1))
                with lock:
                    load.events[kind] = load.events.get(kind, 0) + 1
            except Exception as e:  # noqa: BLE001 — 例如并发下文档状态已变：记下，不中断压测
                with lock:
                    load.event_errors.append(f"{kind}: {type(e).__name__}: {e}")
        c.close()

    def upload_loop():
        c = connect()
        ing_u = Ingestion(c, store)
        rev = Review(c, Lifecycle(c), store)
        pending: list[uuid.UUID] = []
        for i in range(uploads):
            if stop_load.is_set():
                break
            name, data = document(docs + i, paras)
            pending.append(ing_u.receive(TENANT, data, name, acl_policy_id=acl).version_id)
            with lock:
                load.uploads += 1
            time.sleep(0.5)
            pending = _approve_parsed(c, rev, pending, published, lock)
        deadline = time.monotonic() + timeout          # 收尾阶段也要把已上传的审核通过（解析排在索引任务之后）
        while pending and time.monotonic() < deadline:
            time.sleep(0.5)
            pending = _approve_parsed(c, rev, pending, published, lock)
        c.close()

    threads = [threading.Thread(target=worker_loop, daemon=True) for _ in range(workers)]
    loaders = ([threading.Thread(target=query_loop, daemon=True) for _ in range(query_threads)]
               + [threading.Thread(target=event_loop, daemon=True), threading.Thread(target=upload_loop, daemon=True)])
    t0 = datetime.now(timezone.utc)
    started = time.monotonic()
    for t in threads + loaders:
        t.start()

    # 突发：一次提交全部（每份单独成事务，AC-6-5）；持续：按 approve_rate 每秒审核通过
    rev = Review(main, Lifecycle(main), store)
    ok = 0
    step = 1 if approve_rate else 100
    for chunk in range(0, docs, step):
        if approve_rate:
            time.sleep(max(0.0, started + chunk / approve_rate - time.monotonic()))
        part = received[chunk:chunk + step]
        res = rev.decide(TENANT, [{"task_id": str(tasks[r.version_id]), "decision": "approve",
                                   "corrections": {"doc_type": "normative"}} for r in part], ACTOR)
        ok += sum(x["ok"] for x in res)
        with lock:
            published.extend(r.document_id for r, x in zip(part, res) if x["ok"])
    log(f"批量审核通过 {ok}/{docs}，用时 {time.monotonic() - started:.1f} 秒")

    def burst_done() -> bool:
        return main.execute("SELECT count(*) AS n FROM ingest_job WHERE stage = 'publish_index' AND status <> 'done'"
                            ).fetchone()["n"] == 0

    while time.monotonic() - started < timeout:
        q = queue_stats(main, datetime.now(timezone.utc))
        load.max_queue = max(load.max_queue, sum(q["queued"].values()) + sum(q["running"].values()))
        if burst_done() and time.monotonic() - started >= min_seconds:
            break
        time.sleep(1)
    stop_load.set()
    for t in loaders:
        t.join(timeout)
    # 收尾：剩下的任务做完（不再产生新事件）
    while time.monotonic() - started < timeout:
        unfinished = main.execute("SELECT count(*) AS n FROM ingest_job WHERE status IN ('queued', 'running')"
                                  ).fetchone()["n"]
        if unfinished == 0:
            break
        time.sleep(0.5)
    elapsed = time.monotonic() - started
    stop_workers.set()
    for t in threads:
        t.join(60)

    lat = index_latency(main, t0)
    failed = main.execute("SELECT stage, count(*) AS n FROM ingest_job WHERE status = 'failed' GROUP BY stage").fetchall()
    unfinished = main.execute("SELECT count(*) AS n FROM ingest_job WHERE status IN ('queued', 'running')").fetchone()["n"]
    diff = reconcile(main, make_index(), TENANT)
    main.close()
    qs = sorted(load.queries)
    return {
        "docs": docs, "approve_rate": approve_rate, "paras": paras, "workers": workers, "query_threads": query_threads, "event_rate": event_rate,
        "uploads": load.uploads, "embedder": embedder.name, "seconds": round(elapsed, 1),
        "index_latency": lat, "max_queue": load.max_queue,
        "failed_jobs": {r["stage"]: r["n"] for r in failed}, "unfinished_jobs": unfinished,
        "reconcile": {k: len(v) for k, v in diff.items()},
        "events": load.events, "event_errors": len(load.event_errors), "event_error_samples": load.event_errors[:3],
        "queries": len(qs), "query_errors": len(load.query_errors), "query_error_samples": load.query_errors[:3],
        "query_p50": round(qs[len(qs) // 2], 3) if qs else None,
        "query_p95": round(qs[min(len(qs) - 1, int(len(qs) * 0.95))], 3) if qs else None,
    }


def _approve_parsed(conn, rev, pending: list, published: list, lock) -> list:
    """解析完成（审核任务可以通过）的就审核通过，返回还没解析完的。"""
    if not pending:
        return pending
    rows = conn.execute(
        "SELECT t.id, t.version_id, v.document_id FROM review_task t JOIN document_version v ON v.id = t.version_id "
        "WHERE t.status = 'open' AND t.version_id = ANY(%s) AND NOT EXISTS (SELECT 1 FROM ingest_job j "
        "  WHERE j.version_id = t.version_id AND j.stage IN ('parse_text', 'classify_extract') AND j.status <> 'done')",
        (pending,)).fetchall()
    if rows:
        res = rev.decide(TENANT, [{"task_id": str(r["id"]), "decision": "approve",
                                   "corrections": {"doc_type": "normative"}} for r in rows], ACTOR)
        with lock:
            published.extend(r["document_id"] for r, x in zip(rows, res) if x["ok"])
    done = {r["version_id"] for r in rows}
    return [v for v in pending if v not in done]


def verdict(r: dict, target: float) -> list[str]:
    """不通过的原因；空表示通过。"""
    out = []
    p95 = r["index_latency"]["p95"]
    if p95 is None:
        out.append("没有完成的索引任务")
    elif p95 > target:
        out.append(f"索引更新延迟 P95 {p95} 秒，超过 {target} 秒")
    if r["failed_jobs"]:
        out.append(f"失败任务 {r['failed_jobs']}")
    if r["unfinished_jobs"]:
        out.append(f"压测结束时仍有 {r['unfinished_jobs']} 个任务未完成")
    if any(r["reconcile"].values()):
        out.append(f"对账差异 {r['reconcile']}")
    if r["query_errors"]:
        out.append(f"查询错误 {r['query_errors']} 次")
    return out


def report(r: dict, target: float) -> str:
    lat = r["index_latency"]
    how = f"每秒审核通过 {r['approve_rate']} 份，共 {r['docs']} 份" if r.get("approve_rate") else f"{r['docs']} 份文档突发审核"
    lines = [f"索引更新延迟压测（AC-5-6）：{how} + 上传 {r['uploads']} 份 + 每秒 {r['event_rate']} 个变更事件 + "
             f"{r['query_threads']} 个查询线程；后台 {r['workers']} 个；Embedding {r['embedder']}；用时 {r['seconds']} 秒",
             f"索引任务 {lat['n']} 个：P50 {lat['p50']} 秒，P95 {lat['p95']} 秒，最大 {lat['max']} 秒（目标 P95 ≤ {target} 秒）"]
    for stage, s in lat["by_stage"].items():
        lines.append(f"  {stage}：{s['n']} 个，P50 {s['p50']}，P95 {s['p95']}，最大 {s['max']}")
    lines += [f"队列最长 {r['max_queue']}；失败任务 {r['failed_jobs'] or 0}；结束时未完成 {r['unfinished_jobs']}；对账差异 {r['reconcile']}",
              f"变更事件 {r['events']}（并发冲突 {r['event_errors']} 次，事件本身被拒绝，不影响统计）",
              f"查询 {r['queries']} 次，P50 {r['query_p50']} 秒，P95 {r['query_p95']} 秒，错误 {r['query_errors']} 次"]
    problems = verdict(r, target)
    lines.append("结论：通过" if not problems else "结论：不通过——" + "；".join(problems))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from eaer.ingestion.indexer import fresh_index
    from eaer.ingestion.stats import TARGET_P95_SECONDS
    from eaer.retrieval.opensearch import DEFAULT_ANALYZER, OpenSearchIndex, connect
    from eaer.storage import LocalFileStore

    p = argparse.ArgumentParser(prog="eaer.evalrun.indexlag", description="索引更新延迟压测（AC-5-6）")
    p.add_argument("--admin-url", required=True)
    p.add_argument("--opensearch", required=True)
    p.add_argument("--docs", type=int, default=500)
    p.add_argument("--paras", type=int, default=8, help="每份文档的条数")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--query-threads", type=int, default=4)
    p.add_argument("--event-rate", type=float, default=2.0, help="每秒变更事件数")
    p.add_argument("--uploads", type=int, default=50)
    p.add_argument("--approve-rate", type=float, help="每秒审核通过的份数（持续负载）；不给则一次全部审核通过（突发）")
    p.add_argument("--min-seconds", type=float, default=30.0)
    p.add_argument("--target", type=float, default=TARGET_P95_SECONDS)
    p.add_argument("--hashing", action="store_true", help="用测试替身向量化（数字只说明流程）")
    p.add_argument("--out", help="把结果（JSON）写入文件")
    args = p.parse_args(argv)
    if os.environ.get("EAER_ENV") != "eval":
        print("只在评测环境运行（EAER_ENV=eval）", file=sys.stderr)
        return 2
    if args.hashing:
        from eaer.retrieval.embed import HashingEmbedder
        embedder = HashingEmbedder(64)
        dims = 64
    else:
        from eaer.adapters.online import OnlineModelClient
        model = OnlineModelClient.from_env()
        if model.emb is None:
            print("未配置 Embedding 服务（EAER_EMBED_*）；只验证流程时加 --hashing", file=sys.stderr)
            return 2
        embedder, dims = model.embedder(), model.embedding_dims
    from eaer.evalrun.analyzers import scratch_databases

    client = connect(args.opensearch)
    alias = f"eaer-lag-{uuid.uuid4().hex[:8]}"
    import tempfile
    with tempfile.TemporaryDirectory() as tmp, scratch_databases(args.admin_url) as new_db, \
            fresh_index(client, alias, build_id="lag", dims=dims, analyzer=DEFAULT_ANALYZER):
        r = run_load(new_db(), LocalFileStore(Path(tmp)), lambda: OpenSearchIndex(connect(args.opensearch), alias=alias),
                     embedder, docs=args.docs, workers=args.workers, query_threads=args.query_threads,
                     event_rate=args.event_rate, uploads=args.uploads, paras=args.paras, min_seconds=args.min_seconds,
                     approve_rate=args.approve_rate,
                     log=lambda s: print(s, file=sys.stderr))
    print(report(r, args.target))
    if args.out:
        Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if not verdict(r, args.target) else 1


if __name__ == "__main__":
    sys.exit(main())
