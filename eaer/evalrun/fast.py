"""Fast 模式试跑（开发文档 8.3、14.4；为 AC-8-9、AC-8-10 的基线做准备）。

按日期回放事实表样例，每个查询在它的提问日用 Fast 回答，与 factset 计算的标准答案比对：
- 应答类：标准答案中的每个取值（按数值比较，忽略单位写法）都出现在最终回答中才算正确；
- 拒答类：拒答，或回答中不出现目标事实键的任何取值，才算正确；
- 标准答案要求“未验证”的，另记回答是否带了“未验证”标注；
- “验证器”类查询（如 Q16）给的是待验证的结论而不是问题，不适用 Fast，单列。
同时记录每个查询的耗时与 token 数，给出 P50、P95。

    EAER_ENV=eval python -m eaer.evalrun.fast 事实表.xlsx --admin-url postgresql://… [--format docx|html]

模型服务按 EAER_GEN_* / EAER_EMBED_* / EAER_RERANK_* 环境变量连接（见 eaer/adapters/online.py）。
不带 --engine 时只用 Fast（期望由 Deep 或 Verified 回答的问题也只用 Fast）；带 --engine 时经完整编排器
（Router、Deep、Verified）。Embedding 与重排服务未配置时，检索的向量一路用测试替身 HashingEmbedder、不重排，结果只说明流程，不能作为产品基线
（CLAUDE.md“测试替身的数字不是产品结论”）。索引用内存参考实现，关键词一路是它的单字加二字切分。
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
from eaer.evalrun.license import add_external_flag, emit, provenance_of, refuse_early
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
    mode_path: list[str] | None = None
    router: str = ""
    expected_mode: str | None = None


def _quantities(text: str):
    return {q for q, _ in extract(text).quantities}


def mentions(answer: str, f) -> bool:
    """回答里是否出现了这条事实的取值：数值按规范化后的数量比较，文本取值（如“二等座”）按原文。"""
    if isinstance(f.value, (int, float)):
        want = next(iter(_quantities(f"{f.value}{f.unit or ''}")), None)    # 带单位：450 万元是 4,500,000，不是 450
        return want is not None and want in _quantities(answer)
    return str(f.value) in answer


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
    answer = a.answer or ""
    if exp.refuse:
        leaked = any(mentions(answer, f) for f in fs.facts.values() if f.key == q.target_key)
        outcome = "leaked" if leaked and not a.abstained else "refused_ok"
    else:
        ok = all(mentions(answer, fs.facts[i]) for i in exp.basis)
        outcome = "correct" if ok and not a.abstained else "wrong"
    return FastResult(**base, answer=a.answer, outcome=outcome, unverified_labelled="未验证" in a.answer,
                      forced_verify=a.forced_verify, seconds=a.seconds,
                      tokens=sum(a.usage.values()) if a.usage else 0)


def verify_claim(conn, retriever, model, tenant_id: str, q) -> tuple[str, str]:
    """“验证器”类查询（如 Q16）：给的是待验证的结论。按 Verified 的做法检索、确定性检查、语义检查，返回 (状态, 原因)。"""
    from eaer.orchestrator.fast import Steps, evidence_block, target_of
    from eaer.verifier.semantic import SemanticChecker

    claim = q.question.removeprefix("结论：")
    scope = Scope(tenant_id, frozenset({q.role}))
    steps = Steps(conn, retriever, model, LocalAuthorizer(conn, {}))
    hits = steps.retrieve(scope, claim, today=q.asked_on, as_of_valid=None, as_of_system=None)
    meta = steps.meta(hits, None)
    v = SemanticChecker(model).judge(claim, [(h.evidence_id, evidence_block(h, meta[h.version_id])) for h in hits],
                                     q.asked_on)
    if not v.supporting_ids:
        return v.status, v.reason
    c = steps.check(scope, [{"text": claim, "evidence_ids": v.supporting_ids, "core": True}], hits, meta,
                    target_of(q.asked_on, None))[0]
    worst = max([c.status, v.status], key=["supported", "unverified", "unsupported"].index)
    return worst, "；".join(x["reason"] for x in c.checks.values() if x["reason"]) or v.reason


def run_engine(conn, fs: FactSet, corpus: Path, manifest: dict, *, model: ModelClient, store, index: SearchIndex,
               embedder: Embedder, reranker=None) -> list[FastResult]:
    """经完整的编排器回答（Router、升级、Deep、Verified）；不传时间参数，时间点由编排器从问题中识别（7.5）。"""
    from eaer.orchestrator.engine import Engine
    calc = Calculator(fs)

    def ask(conn, fs, q, retriever, ids, tenant_id):
        if q.type == "验证器":
            status, reason = verify_claim(conn, retriever, model, tenant_id, q)
            exp = calc.compute(q)
            want = "supported" if exp.answer == "支持" else "unsupported"      # 标准答案：支持 / 不支持
            # 依据的版本只有推断或未知的生效日时，验证器按设计给“未验证”（9.2），也算对
            ok = status == want or (want == "supported" and status == "unverified"
                                    and exp.grade not in (None, "explicit"))
            return FastResult(q.id, q.type, q.question, exp.answer, f"{status}：{reason}"[:200],
                              "correct" if ok else "wrong", False, False, [], 0.0, 0,
                              expected_mode=q.expected_mode)
        eng = Engine(conn, retriever, model, LocalAuthorizer(conn, {}), store=store)
        try:
            r = eng.ask(Scope(tenant_id, frozenset({q.role})), q.question, today=q.asked_on)
        except Exception as e:  # noqa: BLE001
            return score(fs, calc, q, None, f"{type(e).__name__}: {e}")
        a = FastAnswer(r.answer, r.claims, r.abstained, r.notices, r.forced_verify, r.time_basis, r.mode, r.mode_path,
                       r.model, {k: v for k, v in r.cost.get("by_kind", {}).get("generate", {}).items()
                                 if k in ("input_tokens", "output_tokens")}, r.seconds)
        res = score(fs, calc, q, a)
        res.mode_path, res.router, res.expected_mode = r.mode_path, r.router.get("decided", ""), q.expected_mode
        return res

    return run(conn, fs, corpus, manifest, store=store, index=index, embedder=embedder, ask=ask,
               reranker=reranker).results


def run_fast(conn, fs: FactSet, corpus: Path, manifest: dict, *, model: ModelClient, store, index: SearchIndex,
             embedder: Embedder, reranker=None) -> list[FastResult]:
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

    return run(conn, fs, corpus, manifest, store=store, index=index, embedder=embedder, ask=ask,
               reranker=reranker).results


def report(results: list[FastResult], *, model: str, embedder: str) -> str:
    scored = [r for r in results if r.outcome in ("correct", "wrong", "refused_ok", "leaked", "error")]
    good = [r for r in scored if r.outcome in ("correct", "refused_ok")]
    secs = sorted(r.seconds for r in scored if r.outcome != "error")
    p95 = secs[min(len(secs) - 1, int(round(0.95 * (len(secs) - 1))))] if secs else float("nan")
    engine = any(r.mode_path is not None for r in results)
    title = "编排器试跑" if engine else "Fast 试跑"
    lines = [f"{title}：生成 {model}；向量化 {embedder}；{len(scored)} 个可计分查询",
             f"正确 {len(good)} / {len(scored)}；耗时 P50 {statistics.median(secs):.2f} 秒、P95 {p95:.2f} 秒；"
             f"token 合计 {sum(r.tokens for r in results)}"]
    if engine:
        routed = [r for r in results if r.expected_mode and r.mode_path is not None]
        hit = sum(r.expected_mode.lower() in r.mode_path + [r.router] for r in routed)
        lines.append(f"模式：预期模式出现在 Router 决定或实际路径中的 {hit} / {len(routed)}")
        lines += ["| 查询 | 类型 | 结果 | 预期模式 | Router | 路径 | 未验证标注 | 耗时 | 回答 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    else:
        lines += ["| 查询 | 类型 | 结果 | 未验证标注 | 强制验证 | 耗时 | 回答 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        mark = ("应有，" + ("有" if r.unverified_labelled else "缺")) if r.unverified_expected else (
            "多标" if r.unverified_labelled else "")
        if engine:
            lines.append(f"| {r.query_id} | {r.type} | {r.outcome} | {r.expected_mode or ''} | {r.router} | "
                         f"{'→'.join(r.mode_path or [])} | {mark} | {r.seconds:.2f} | {(r.answer or r.error)[:70]} |")
        else:
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
    p.add_argument("--engine", action="store_true", help="经完整编排器（Router、升级、Deep、Verified）回答")
    p.add_argument("--split", choices=["标定", "测试"], help="只用该集合的查询（14.3：指标在测试集上报）")
    p.add_argument("--per-type", type=int, help="每类查询最多取几条（按编号哈希抽样，可复现）")
    add_external_flag(p)
    args = p.parse_args(argv)
    if os.environ.get("EAER_ENV") != "eval":
        print("只在评测环境运行（EAER_ENV=eval）", file=sys.stderr)
        return 2
    prov = [provenance_of(args.xlsx)]
    if (code := refuse_early(prov, external=args.external)) is not None:
        return code
    model = OnlineModelClient.from_env()
    missing = model.missing_for_fast()
    if "生成" in missing:
        print("未配置生成服务（EAER_GEN_*）", file=sys.stderr)
        return 2
    embedder: Embedder = model.embedder() if model.emb is not None else HashingEmbedder()
    reranker = model if model.rr is not None else None
    from factset.subset import subset
    fs = subset(load(args.xlsx), split=args.split, per_type=args.per_type)
    with tempfile.TemporaryDirectory() as tmp, scratch_databases(args.admin_url) as new_db:
        import psycopg
        from psycopg.rows import dict_row

        work = Path(tmp)
        manifest = Renderer(fs).render(work / "corpus", args.format)
        conn = psycopg.connect(new_db(), autocommit=True, row_factory=dict_row)
        runner = run_engine if args.engine else run_fast
        results = runner(conn, fs, work / "corpus", manifest, model=model, store=LocalFileStore(work / "files"),
                         index=MemoryIndex(), embedder=embedder, reranker=reranker)
        conn.close()
    text = report(results, model=model.gen.model,
                  embedder=embedder.name + (f"，重排 {model.rr.model}" if reranker else "，不重排"))
    if missing:
        text += f"\n\n注意：未配置 {'、'.join(missing)}，结果只说明流程，不能作为产品基线。"
    return emit(text, prov, external=args.external)


if __name__ == "__main__":
    sys.exit(main())
