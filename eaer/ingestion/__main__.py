"""后台任务进程与索引运维入口。

    python -m eaer.ingestion                 后台任务进程：解析、分类与抽取、深度解析、索引任务，以及转入后台的提问
    python -m eaer.ingestion --no-index      只做解析与抽取，不处理索引任务与提问（留在队列里）
    python -m eaer.ingestion rebuild --build-id b2   全量重建索引并切换别名（开发文档 5.3）
    python -m eaer.ingestion ensure-index    别名不存在时建一个（首次部署、或索引丢失后从数据库重建）；已存在则什么都不做
    python -m eaer.ingestion requeue-deep    对全部已发布版本重新排队深度解析（例如刚开启看图能力后补读已有文档的图片）
    python -m eaer.ingestion drop-retired    删除别名不再指向、且超过 24 小时的旧索引（应定时运行）
    python -m eaer.ingestion purge           立即做一次 Trace 清理（后台任务进程每 EAER_PURGE_SECONDS 秒自动做一次，默认 3600）
    python -m eaer.ingestion stats           入库队列与索引更新延迟（JSON，供监控采集；--since-minutes 默认 60）
    python -m eaer.ingestion reconcile       按租户对账：数据库中可搜索的版本与索引中的版本之差（JSON，有差异时退出码 1；
                                             --tenant 只查给定的租户）
    python -m eaer.ingestion reconcile --repair   对账并补任务：缺的、多的版本补排 index_update，由后台任务进程执行
                                             （后台任务进程每 EAER_RECONCILE_SECONDS 秒自动做一次，默认 3600；见 repair.py）

环境变量：EAER_DATABASE_URL、EAER_STORAGE_DIR（同接口进程）；EAER_POLL_SECONDS（空闲轮询间隔，默认 2）；
EAER_OPENSEARCH_URL、EAER_INDEX_ALIAS（默认 eaer）、EAER_TEXT_ANALYZER（standard / cjk / smartcn / icu_analyzer /
jieba，默认 standard）；Embedding 服务 EAER_EMBED_*（见 eaer/adapters/online.py）。
处理索引任务时这几项缺一不可：缺少即启动报错，不会让索引任务默默排队（开发文档 12.1“部署时报错”）；
也不能用测试替身向量化。
转入后台、或执行者中断的提问（开发文档 8.5）也由本进程接着执行，需要与接口进程相同的查询配置与 EAER_TOKENS_FILE
（见 eaer/runtime.py）；不开放提问时设置 EAER_DISABLE_QUERY=1。
任务失败后按 30 秒、2 分钟、10 分钟退避再试，第 4 次仍失败标为 failed（pipeline.Worker）；索引任务永久失败留下的差异由对账补写（repair.py）。
EAER_TRACE_EXPORTER none（默认）/ console / otlp：后台任务与接着执行的提问记 trace（13.3，见 eaer/observability.py）。
图片读取（6.2 第 6 阶段）：生成服务声明了看图能力（EAER_GEN_VISION=1）时，深度解析把文档里的图片交给它转写成
证据（eaer/ingestion/figures.py）；没有声明时跳过图片。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import psycopg
from psycopg.rows import dict_row

from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.runtime import ConfigError, query_runner, segmenter_for
from eaer.storage import LocalFileStore

__all__ = ["ConfigError", "build_worker", "index_parts", "main", "segmenter_for"]

log = logging.getLogger("eaer.ingestion")


def index_parts(env: dict[str, str]):
    """返回 (OpenSearch 客户端, 别名, 分词器名, 应用侧分词器, Embedder)；缺配置时抛 ConfigError。"""
    from eaer.adapters.online import OnlineModelClient
    from eaer.retrieval.opensearch import connect

    if not env.get("EAER_OPENSEARCH_URL"):
        raise ConfigError("未设置 EAER_OPENSEARCH_URL，无法处理索引任务")
    model = OnlineModelClient.from_env(env)
    if model.emb is None:
        raise ConfigError("未配置 Embedding 服务（EAER_EMBED_*），无法处理索引任务")
    from eaer.retrieval.opensearch import DEFAULT_ANALYZER
    analyzer = env.get("EAER_TEXT_ANALYZER", DEFAULT_ANALYZER)
    return (connect(env["EAER_OPENSEARCH_URL"]), env.get("EAER_INDEX_ALIAS", "eaer"), analyzer,
            segmenter_for(analyzer), model.embedder())


def figure_reader_for(env: dict[str, str]):
    """生成服务声明了看图能力时的图片读取器，否则 None。"""
    from eaer.adapters.online import OnlineModelClient
    from eaer.ingestion.figures import FigureReader

    model = OnlineModelClient.from_env(env)
    return FigureReader(model) if model.gen is not None and model.vision else None


def build_worker(conn: psycopg.Connection, store, env: dict[str, str], *, with_index: bool = True) -> Worker:
    ing = Ingestion(conn, store, figure_reader=figure_reader_for(env))
    if not with_index:
        return Worker(ing)
    from eaer.ingestion.indexer import Indexer
    from eaer.retrieval.opensearch import OpenSearchIndex

    client, alias, analyzer, seg, embedder = index_parts(env)
    ix = OpenSearchIndex(client, alias=alias, segmenter=seg)
    current = sorted(ix.current())
    if len(current) != 1:
        raise ConfigError(f"别名 {alias} 没有指向唯一的索引（{current}）：先运行 python -m eaer.ingestion rebuild")
    recorded, expected = ix.text_analysis(), (seg.name if seg else analyzer)
    if recorded != expected:
        raise ConfigError(f"索引的分词方式是 {recorded}，本进程配置的是 {expected}：两者必须一致，换分词器要重建索引")
    return Worker(ing, Indexer(conn, ix, embedder, current[0].removeprefix(f"{alias}-")))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eaer.ingestion")
    p.add_argument("command", nargs="?", default="worker", choices=["worker", "rebuild", "ensure-index", "drop-retired",
                                                                   "purge", "stats", "requeue-deep", "reconcile"])
    p.add_argument("--no-index", action="store_true", help="只做解析与抽取，不处理索引任务")
    p.add_argument("--build-id", help="rebuild：新索引的构建编号")
    p.add_argument("--retain-hours", type=float, default=24.0, help="drop-retired：旧索引保留时长")
    p.add_argument("--since-minutes", type=float, default=60.0, help="stats：统计最近多少分钟内排队的索引任务")
    p.add_argument("--repair", action="store_true", help="reconcile：为有差异的版本补排索引任务")
    p.add_argument("--tenant", action="append", help="reconcile：只对这个租户（可重复）；默认全部租户")
    args = p.parse_args(argv)
    env = dict(os.environ)
    try:
        if args.command == "drop-retired":
            from datetime import timedelta

            from eaer.ingestion.indexer import drop_retired
            from eaer.retrieval.opensearch import connect
            if not env.get("EAER_OPENSEARCH_URL"):
                raise ConfigError("未设置 EAER_OPENSEARCH_URL")
            print("已删除：", drop_retired(connect(env["EAER_OPENSEARCH_URL"]), env.get("EAER_INDEX_ALIAS", "eaer"),
                                         older_than=timedelta(hours=args.retain_hours)))
            return 0
        conn = psycopg.connect(env["EAER_DATABASE_URL"], autocommit=True, row_factory=dict_row)
        if args.command == "stats":
            from datetime import datetime, timedelta, timezone

            from eaer.ingestion.stats import TARGET_P95_SECONDS, index_latency, queue_stats, repair_stats
            now = datetime.now(timezone.utc)
            since = now - timedelta(minutes=args.since_minutes)
            print(json.dumps({"queue": queue_stats(conn, now), "index_latency": index_latency(conn, since),
                              "index_latency_target_p95": TARGET_P95_SECONDS,
                              "index_repair": repair_stats(conn, since)}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "reconcile":
            return reconcile_command(conn, env, repair=args.repair, tenants=args.tenant)
        if args.command == "purge":
            from datetime import datetime, timezone

            from eaer.orchestrator.tasks import purge
            print("已删除：", purge(conn, datetime.now(timezone.utc)))
            return 0
        if args.command == "requeue-deep":
            import uuid as _uuid
            from datetime import datetime, timezone

            from eaer.domain.jobs import PRIORITY
            now = datetime.now(timezone.utc)
            rows = conn.execute(
                "SELECT v.id FROM document_version v WHERE v.status IN ('searchable', 'archived') AND NOT EXISTS ("
                "  SELECT 1 FROM ingest_job j WHERE j.version_id = v.id AND j.stage = 'parse_deep' AND j.status = 'queued')"
            ).fetchall()
            for r in rows:
                conn.execute("INSERT INTO ingest_job (id, version_id, stage, status, priority, created_at, updated_at) "
                             "VALUES (%s, %s, 'parse_deep', 'queued', %s, %s, %s)",
                             (_uuid.uuid4(), r["id"], PRIORITY["parse_deep"], now, now))
            print(f"已为 {len(rows)} 个版本重新排队深度解析")
            return 0
        if args.command == "ensure-index":
            from datetime import datetime, timezone

            from eaer.ingestion.indexer import rebuild
            from eaer.retrieval.opensearch import OpenSearchIndex
            client, alias, analyzer, seg, embedder = index_parts(env)
            current = sorted(OpenSearchIndex(client, alias=alias).current())
            if current:
                print(f"别名 {alias} 已指向 {current}，无需新建")
                return 0
            build_id = args.build_id or datetime.now(timezone.utc).strftime("b%Y%m%d%H%M%S")
            print(rebuild(conn, client, alias, build_id=build_id, embedder=embedder, analyzer=analyzer, segmenter=seg))
            return 0
        if args.command == "rebuild":
            from eaer.ingestion.indexer import rebuild
            if not args.build_id:
                raise ConfigError("rebuild 需要 --build-id")
            client, alias, analyzer, seg, embedder = index_parts(env)
            print(rebuild(conn, client, alias, build_id=args.build_id, embedder=embedder, analyzer=analyzer,
                          segmenter=seg))
            return 0
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s：%(message)s")
        from eaer.observability import TraceConfigError, configure
        try:
            configure(env, "eaer-worker")
        except TraceConfigError as e:
            raise ConfigError(str(e))
        store = LocalFileStore(env["EAER_STORAGE_DIR"])
        worker = build_worker(conn, store, env, with_index=not args.no_index)
        runner = None if args.no_index else query_runner(env, store)
    except ConfigError as e:
        print(f"配置错误：{e}", file=sys.stderr)
        return 2
    poll = float(env.get("EAER_POLL_SECONDS", "2"))
    purge_every, last_purge = float(env.get("EAER_PURGE_SECONDS", "3600")), 0.0
    reconcile_every, last_reconcile = float(env.get("EAER_RECONCILE_SECONDS", "3600")), 0.0
    while True:
        done = worker.run_until_idle(limit=1000) + (runner.run_pending(limit=10) if runner else 0)
        if time.monotonic() - last_purge >= purge_every:          # Trace 清理（10.2、AC-10-10）
            from datetime import datetime, timezone

            from eaer.orchestrator.tasks import purge
            purge(conn, datetime.now(timezone.utc))
            last_purge = time.monotonic()
        if worker.indexer is not None and time.monotonic() - last_reconcile >= reconcile_every:   # 对账补写（5.4）
            periodic_repair(conn, worker.indexer.index)
            last_reconcile = time.monotonic()
        if done == 0:
            time.sleep(poll)


def periodic_repair(conn: psycopg.Connection, index) -> list | None:
    """后台任务进程定期的对账补写。对账本身出错（如 OpenSearch 暂时不可用）只记日志，不影响任务处理，下一轮再做。"""
    from eaer.ingestion.repair import repair_all
    try:
        return repair_all(conn, index)
    except Exception as e:  # noqa: BLE001
        log.error("索引对账失败，下一轮再做：%s: %s", type(e).__name__, e)
        return None


def reconcile_command(conn: psycopg.Connection, env: dict[str, str], *, repair: bool, tenants: list[str] | None) -> int:
    """reconcile 命令：只读对账，或 --repair 时补任务（补的任务由后台任务进程执行）。只需要 OpenSearch，不需要模型服务。"""
    from eaer.retrieval.opensearch import OpenSearchIndex, connect

    if not env.get("EAER_OPENSEARCH_URL"):
        raise ConfigError("未设置 EAER_OPENSEARCH_URL")
    index = OpenSearchIndex(connect(env["EAER_OPENSEARCH_URL"]), alias=env.get("EAER_INDEX_ALIAS", "eaer"))
    if not index.current():
        raise ConfigError(f"别名 {index.alias} 不存在：先运行 python -m eaer.ingestion ensure-index")
    if tenants is None:
        tenants = [r["tenant_id"] for r in conn.execute("SELECT tenant_id FROM tenant_config ORDER BY tenant_id")]
    if not repair:
        from eaer.retrieval.reconcile import reconcile
        out = {t: {k: sorted(v) for k, v in reconcile(conn, index, t).items()} for t in tenants}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1 if any(v for d in out.values() for v in d.values()) else 0
    from eaer.ingestion.repair import repair_all
    results = repair_all(conn, index, tenants=tenants)
    if results is None:
        print("另一个进程正在做索引对账，稍后再试", file=sys.stderr)
        return 1
    print(json.dumps({r.tenant_id: r.to_json() for r in results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
