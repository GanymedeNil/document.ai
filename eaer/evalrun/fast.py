"""Fast 模式试跑（开发文档 8.3、14.4；为 AC-8-9、AC-8-10 的基线做准备）。

按日期回放事实表样例，每个查询在它的提问日用 Fast 回答，与 factset 计算的标准答案比对：
- 应答类：标准答案中的每个取值（按数值比较，忽略单位写法）都出现在最终回答中才算正确；
- 拒答类：拒答，或回答中不出现目标事实键的任何取值，才算正确；
- 标准答案要求“未验证”的，另记回答是否带了“未验证”标注；
- “验证器”类查询（如 Q16）给的是待验证的结论而不是问题，不适用 Fast，单列。
同时记录每个查询的耗时与 token 数，给出 P50、P95。

    EAER_ENV=eval python -m eaer.evalrun.fast 事实表.xlsx --admin-url postgresql://… [--format docx|html]

生成服务按 EAER_GEN_* 环境变量连接（见 eaer/adapters/online.py）。注意：Router、Deep、Verified 尚未实现，
期望由 Deep 或 Verified 回答的问题这里也只用 Fast；Embedding 与重排服务未配置时，检索的向量一路用测试替身
HashingEmbedder、不重排，结果只说明流程，不能作为产品基线（CLAUDE.md“测试替身的数字不是产品结论”）。
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from eaer.adapters.authorizer import LocalAuthorizer
from eaer.adapters.model import ModelClient
from eaer.evalrun.retrieval import run
from eaer.orchestrator.fast import FastAnswer, FastPipeline
from eaer.retrieval.embed import Embedder, HashingEmbedder
from eaer.retrieval.index import MemoryIndex, SearchIndex
from eaer.retrieval.service import Scope
from eaer.storage import LocalFileStore
from eaer.verifier.deterministic import extract
from factset.answer import Calculator
from factset.model import FactSet


@dataclass
class FastResult:
    query_id: str
    type: str
    question: str
    expected: str
    answer: str
    outcome: str                   # correct / wrong / refused_ok / leaked / not_applicable / error
    unverified_expected: bool
    unverified_labelled: bool
    forced_verify: list[str]
    seconds: float
    tokens: int
    error: str = ""


def _quantities(text: str):
    return {q for q, _ in extract(text).quantities}


def score(fs: FactSet, calc: Calculator, q, a: FastAnswer | None, error: str = "") -> FastResult:
    exp = calc.compute(q)
    base = dict(query_id=q.id, type=q.type, question=q.question, expected=exp.answer,
                unverified_expected=exp.unverified)
    if q.type == "验证器":
        return FastResult(**base, answer="", outcome="not_applicable", unverified_labelled=False, forced_verify=[],
                          seconds=0.0, tokens=0)
    if a is None:
        return FastResult(**base, answer="", outcome="error", unverified_labelled=False, forced_verify=[],
                          seconds=0.0, tokens=0, error=error)
    have = _quantities(a.answer)
    if exp.refuse:
        leaked = any(next(iter(_quantities(str(f.value))), None) in have
                     for f in fs.facts.values() if f.key == q.target_key)
        outcome = "leaked" if leaked and not a.abstained else "refused_ok"
    else:
        want = [f for f in (fs.facts[i] for i in exp.basis)]
        ok = all(next(iter(_quantities(str(f.value))), None) in have for f in want)
        outcome = "correct" if ok and not a.abstained else "wrong"
    return FastResult(**base, answer=a.answer, outcome=outcome, unverified_labelled="未验证" in a.answer,
                      forced_verify=a.forced_verify, seconds=a.seconds,
                      tokens=sum(a.usage.values()) if a.usage else 0)


def run_fast(conn, fs: FactSet, corpus: Path, manifest: dict, *, model: ModelClient, store, index: SearchIndex,
             embedder: Embedder) -> list[FastResult]:
    calc = Calculator(fs)

    def ask(conn, fs, q, retriever, ids, tenant_id):
        if q.type == "验证器":
            return score(fs, calc, q, None)
        p = FastPipeline(conn, retriever, model, LocalAuthorizer(conn, {}), store=store)
        try:
            a = p.answer(Scope(tenant_id, frozenset({q.role})), q.question, today=q.asked_on,
                         as_of_valid=q.as_of_valid, as_of_system=q.as_of_system)
        except Exception as e:  # noqa: BLE001 — 试跑记录错误并继续
            return score(fs, calc, q, None, f"{type(e).__name__}: {e}")
        return score(fs, calc, q, a)

    return run(conn, fs, corpus, manifest, store=store, index=index, embedder=embedder, ask=ask).results


def report(results: list[FastResult], *, model: str, embedder: str) -> str:
    scored = [r for r in results if r.outcome in ("correct", "wrong", "refused_ok", "leaked", "error")]
    good = [r for r in scored if r.outcome in ("correct", "refused_ok")]
    secs = sorted(r.seconds for r in scored if r.outcome != "error")
    p95 = secs[min(len(secs) - 1, int(round(0.95 * (len(secs) - 1))))] if secs else float("nan")
    lines = [f"Fast 试跑：生成 {model}；向量化 {embedder}；{len(scored)} 个可计分查询",
             f"正确 {len(good)} / {len(scored)}；耗时 P50 {statistics.median(secs):.2f} 秒、P95 {p95:.2f} 秒；"
             f"token 合计 {sum(r.tokens for r in results)}",
             "| 查询 | 类型 | 结果 | 未验证标注 | 强制验证 | 耗时 | 回答 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        mark = ("应有，" + ("有" if r.unverified_labelled else "缺")) if r.unverified_expected else ""
        lines.append(f"| {r.query_id} | {r.type} | {r.outcome} | {mark} | {'、'.join(r.forced_verify)} | "
                     f"{r.seconds:.2f} | {(r.answer or r.error)[:60]} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from eaer.adapters.online import OnlineModelClient
    from eaer.evalrun.analyzers import scratch_databases
    from factset.load import load
    from factset.render import Renderer

    p = argparse.ArgumentParser(prog="eaer.evalrun.fast", description="在事实表样例上试跑 Fast 模式")
    p.add_argument("xlsx")
    p.add_argument("--admin-url", required=True)
    p.add_argument("--format", choices=["docx", "html"], default="docx")
    args = p.parse_args(argv)
    if os.environ.get("EAER_ENV") != "eval":
        print("只在评测环境运行（EAER_ENV=eval）", file=sys.stderr)
        return 2
    model = OnlineModelClient.from_env()
    missing = model.missing_for_fast()
    if "生成" in missing:
        print("未配置生成服务（EAER_GEN_*）", file=sys.stderr)
        return 2
    if model.emb is not None:
        from eaer.adapters.model import ModelEmbedder
        embedder: Embedder = ModelEmbedder(model, os.environ.get("EAER_EMBED_MODEL", "embedding"))
    else:
        embedder = HashingEmbedder()
    fs = load(args.xlsx)
    with tempfile.TemporaryDirectory() as tmp, scratch_databases(args.admin_url) as new_db:
        import psycopg
        from psycopg.rows import dict_row

        work = Path(tmp)
        manifest = Renderer(fs).render(work / "corpus", args.format)
        conn = psycopg.connect(new_db(), autocommit=True, row_factory=dict_row)
        results = run_fast(conn, fs, work / "corpus", manifest, model=model, store=LocalFileStore(work / "files"),
                           index=MemoryIndex(), embedder=embedder)
        conn.close()
    print(report(results, model=model.gen.model, embedder=embedder.name))
    if missing:
        print(f"\n注意：未配置 {'、'.join(missing)}，结果只说明流程，不能作为产品基线。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
