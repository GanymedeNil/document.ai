"""后台任务进程与索引运维入口。

    python -m eaer.ingestion                 后台任务进程：解析、分类与抽取、深度解析、索引任务，以及转入后台的提问
    python -m eaer.ingestion --no-index      只做解析与抽取，不处理索引任务与提问（留在队列里）
    python -m eaer.ingestion rebuild --build-id b2   全量重建索引并切换别名（开发文档 5.3）
    python -m eaer.ingestion drop-retired    删除别名不再指向、且超过 24 小时的旧索引（应定时运行）
    python -m eaer.ingestion purge           立即做一次 Trace 清理（后台任务进程每 EAER_PURGE_SECONDS 秒自动做一次，默认 3600）

环境变量：EAER_DATABASE_URL、EAER_STORAGE_DIR（同接口进程）；EAER_POLL_SECONDS（空闲轮询间隔，默认 2）；
EAER_OPENSEARCH_URL、EAER_INDEX_ALIAS（默认 eaer）、EAER_TEXT_ANALYZER（smartcn / icu_analyzer / jieba 等，
默认 smartcn）；Embedding 服务 EAER_EMBED_*（见 eaer/adapters/online.py）。
处理索引任务时这几项缺一不可：缺少即启动报错，不会让索引任务默默排队（开发文档 12.1“部署时报错”）；
也不能用测试替身向量化。
转入后台、或执行者中断的提问（开发文档 8.5）也由本进程接着执行，需要与接口进程相同的查询配置与 EAER_TOKENS_FILE
（见 eaer/runtime.py）；不开放提问时设置 EAER_DISABLE_QUERY=1。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import psycopg
from psycopg.rows import dict_row

from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.runtime import ConfigError, query_runner, segmenter_for
from eaer.storage import LocalFileStore

__all__ = ["ConfigError", "build_worker", "index_parts", "main", "segmenter_for"]


def index_parts(env: dict[str, str]):
    """返回 (OpenSearch 客户端, 别名, 分词器名, 应用侧分词器, Embedder)；缺配置时抛 ConfigError。"""
    from eaer.adapters.online import OnlineModelClient
    from eaer.retrieval.opensearch import connect

    if not env.get("EAER_OPENSEARCH_URL"):
        raise ConfigError("未设置 EAER_OPENSEARCH_URL，无法处理索引任务")
    model = OnlineModelClient.from_env(env)
    if model.emb is None:
        raise ConfigError("未配置 Embedding 服务（EAER_EMBED_*），无法处理索引任务")
    analyzer = env.get("EAER_TEXT_ANALYZER", "smartcn")
    return (connect(env["EAER_OPENSEARCH_URL"]), env.get("EAER_INDEX_ALIAS", "eaer"), analyzer,
            segmenter_for(analyzer), model.embedder())


def build_worker(conn: psycopg.Connection, store, env: dict[str, str], *, with_index: bool = True) -> Worker:
    ing = Ingestion(conn, store)
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
    p.add_argument("command", nargs="?", default="worker", choices=["worker", "rebuild", "drop-retired", "purge"])
    p.add_argument("--no-index", action="store_true", help="只做解析与抽取，不处理索引任务")
    p.add_argument("--build-id", help="rebuild：新索引的构建编号")
    p.add_argument("--retain-hours", type=float, default=24.0, help="drop-retired：旧索引保留时长")
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
        if args.command == "purge":
            from datetime import datetime, timezone

            from eaer.orchestrator.tasks import purge
            print("已删除：", purge(conn, datetime.now(timezone.utc)))
            return 0
        if args.command == "rebuild":
            from eaer.ingestion.indexer import rebuild
            if not args.build_id:
                raise ConfigError("rebuild 需要 --build-id")
            client, alias, analyzer, seg, embedder = index_parts(env)
            print(rebuild(conn, client, alias, build_id=args.build_id, embedder=embedder, analyzer=analyzer,
                          segmenter=seg))
            return 0
        store = LocalFileStore(env["EAER_STORAGE_DIR"])
        worker = build_worker(conn, store, env, with_index=not args.no_index)
        runner = None if args.no_index else query_runner(env, store)
    except ConfigError as e:
        print(f"配置错误：{e}", file=sys.stderr)
        return 2
    poll = float(env.get("EAER_POLL_SECONDS", "2"))
    purge_every, last_purge = float(env.get("EAER_PURGE_SECONDS", "3600")), 0.0
    while True:
        done = worker.run_until_idle(limit=1000) + (runner.run_pending(limit=10) if runner else 0)
        if time.monotonic() - last_purge >= purge_every:          # Trace 清理（10.2、AC-10-10）
            from datetime import datetime, timezone

            from eaer.orchestrator.tasks import purge
            purge(conn, datetime.now(timezone.utc))
            last_purge = time.monotonic()
        if done == 0:
            time.sleep(poll)


if __name__ == "__main__":
    sys.exit(main())
