"""分词器对比（开发文档 3.3“中文分词”、AC-5-4）。

同一份事实表、同一批合成文档，在真实 OpenSearch 上每种分词器各建一个索引，按日期完整回放一遍检索评测，
分“仅关键词”与“关键词 + 向量”两种方式计分。每次运行用一个新的、已迁移的数据库，互不影响。

候选 jieba 不是引擎插件，在应用侧切词（eaer/retrieval/segment.py），其余候选是 OpenSearch 的分词器。

    EAER_ENV=eval python -m eaer.evalrun.analyzers 事实表.xlsx \
        --admin-url postgresql://postgres:postgres@localhost:5432/postgres \
        --opensearch http://localhost:9200 [--analyzers standard,cjk,smartcn,icu_analyzer,jieba] [--format docx|html]

设置了 EAER_EMBED_* 时向量化用配置的 Embedding 服务（查询带指令前缀），否则用测试替身 HashingEmbedder——
那时“关键词 + 向量”一栏只说明流程能跑通，不能作为选型结论。按 AC-5-4 选定分词与模型组合需要更大的评测集。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg.rows import dict_row

from eaer.evalrun.retrieval import Report, run
from eaer.ingestion.indexer import fresh_index
from eaer.retrieval.embed import Embedder, HashingEmbedder
from eaer.retrieval.index import Filters, SearchIndex
from eaer.retrieval.opensearch import connect
from eaer.retrieval.segment import JiebaSegmenter
from eaer.storage import LocalFileStore
from factset.model import FactSet

DEFAULT_ANALYZERS = ("standard", "cjk", "smartcn", "icu_analyzer", "jieba")
APP_SEGMENTERS = {"jieba": JiebaSegmenter}        # 在应用侧切词的候选


class LexicalOnly:
    """只保留关键词一路：向量检索返回空，用来单独比较分词器本身的效果。"""

    def __init__(self, index: SearchIndex):
        self._index = index

    def __getattr__(self, name):
        return getattr(self._index, name)

    def search_vector(self, vector: list[float], f: Filters, k: int) -> list[tuple[str, float]]:
        return []


@dataclass
class Row:
    analyzer: str
    mode: str                      # lexical / hybrid
    report: Report

    @property
    def scored(self) -> int:
        return sum(r.outcome in ("hit", "miss") for r in self.report.results)

    @property
    def hits(self) -> int:
        return sum(r.outcome == "hit" for r in self.report.results)

    @property
    def refused_ok(self) -> int:
        return sum(r.outcome == "refused" for r in self.report.results)

    @property
    def refusals(self) -> int:
        return sum(r.outcome in ("refused", "refusal_leak") for r in self.report.results)

    @property
    def mrr(self) -> float:
        return self.report.mrr

    @property
    def misses(self) -> list[str]:
        return [r.query_id for r in self.report.results if r.outcome in ("miss", "refusal_leak")]


def compare(fs: FactSet, corpus: Path, manifest: dict, *, client, new_db: Callable[[], str],
            analyzers: tuple[str, ...] = DEFAULT_ANALYZERS, embedder: Embedder | None = None,
            work_dir: Path) -> list[Row]:
    embedder = embedder or HashingEmbedder()
    rows: list[Row] = []
    for analyzer in analyzers:
        for mode in ("lexical", "hybrid"):
            alias = f"eaer-cmp-{uuid.uuid4().hex[:8]}"
            seg = APP_SEGMENTERS[analyzer]() if analyzer in APP_SEGMENTERS else None
            with fresh_index(client, alias, build_id=analyzer.replace("_", "-"), dims=embedder.dims,
                             analyzer=analyzer, segmenter=seg) as ix:
                index: SearchIndex = ix if mode == "hybrid" else LexicalOnly(ix)
                conn = psycopg.connect(new_db(), autocommit=True, row_factory=dict_row)
                try:
                    rep = run(conn, fs, corpus, manifest, store=LocalFileStore(work_dir / alias), index=index,
                              embedder=embedder, build_id=analyzer)
                finally:
                    conn.close()
            rows.append(Row(analyzer, mode, rep))
    return rows


def table(rows: list[Row]) -> str:
    emb = rows[0].report.embedder if rows else ""
    lines = [f"分词器对比（向量化 {emb}，前 {rows[0].report.top_m if rows else 0} 条）",
             "| 分词器 | 方式 | 命中 / 可计分 | MRR | 拒答正确 | 越权 | 对账差异 | 未命中 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r.analyzer} | {'仅关键词' if r.mode == 'lexical' else '关键词 + 向量'} | "
                     f"{r.hits} / {r.scored} | {r.mrr:.3f} | {r.refused_ok} / {r.refusals} | {r.report.leaks} | "
                     f"{r.report.reconcile_diffs} | {'、'.join(r.misses) or '—'} |")
    return "\n".join(lines)


@contextmanager
def scratch_databases(admin_url: str) -> Iterator[Callable[[], str]]:
    """按需创建已迁移的临时数据库，结束时全部删除。"""
    from eaer.db import migrate

    admin = psycopg.connect(admin_url, autocommit=True)
    created: list[str] = []
    p = urlsplit(admin_url)

    def url(name: str) -> str:
        return urlunsplit((p.scheme, p.netloc, "/" + name, p.query, p.fragment))

    template = f"eaer_cmp_tpl_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{template}"')
    created.append(template)
    migrate(url(template))

    def make() -> str:
        name = f"eaer_cmp_{uuid.uuid4().hex[:8]}"
        admin.execute(f'CREATE DATABASE "{name}" TEMPLATE "{template}"')
        created.append(name)
        return url(name)

    try:
        yield make
    finally:
        for name in reversed(created):
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


def main(argv: list[str] | None = None) -> int:
    from factset.load import load
    from factset.render import Renderer
    from factset.validate import validate

    p = argparse.ArgumentParser(prog="eaer.evalrun.analyzers", description="在真实 OpenSearch 上对比中文分词器")
    p.add_argument("xlsx")
    p.add_argument("--admin-url", required=True, help="PostgreSQL 管理连接串，用来创建临时数据库")
    p.add_argument("--opensearch", required=True)
    p.add_argument("--analyzers", default=",".join(DEFAULT_ANALYZERS))
    p.add_argument("--format", choices=["docx", "html"], default="docx")
    p.add_argument("--split", choices=["标定", "测试"], help="只用该集合的查询（14.3：指标在测试集上报）")
    p.add_argument("--per-type", type=int, help="每类查询最多取几条（按编号哈希抽样，可复现）")
    args = p.parse_args(argv)
    if os.environ.get("EAER_ENV") != "eval":
        print("分词器对比只在评测环境运行（EAER_ENV=eval）", file=sys.stderr)
        return 2
    from factset.subset import subset
    fs = subset(load(args.xlsx), split=args.split, per_type=args.per_type)
    issues = validate(fs)
    if issues:
        print(f"事实表校验未通过：{issues[:3]}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp, scratch_databases(args.admin_url) as new_db:
        work = Path(tmp)
        manifest = Renderer(fs).render(work / "corpus", args.format)
        embedder = None
        if os.environ.get("EAER_EMBED_BASE_URL"):                # 配置了 Embedding 服务就用它，否则用测试替身
            from eaer.adapters.online import OnlineModelClient
            embedder = OnlineModelClient.from_env().embedder()
        rows = compare(fs, work / "corpus", manifest, client=connect(args.opensearch), new_db=new_db,
                       analyzers=tuple(a.strip() for a in args.analyzers.split(",") if a.strip()), work_dir=work,
                       embedder=embedder)
    print(table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
