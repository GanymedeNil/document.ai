"""三方对比（开发文档 8.7 AC-8-11、16.3 交付物）：传统 Hybrid RAG、Always-Agent、Adaptive 的正确率、耗时、成本。

    EAER_ENV=eval python -m eaer.evalrun.compare3 事实表.xlsx --admin-url postgresql://… [--split 测试 --per-type 5] \\
        [--out 报告.md]

三种执行方式用同一份语料、同一套检索（混合检索加重排）、同一个生成模型，按日期回放后在各自的提问日回答同一批查询：
- 传统 Hybrid RAG：混合检索、重排后一次生成，带引用。不经 Router，不从问题中识别时间点（检索按提问日当前生效），
  不升级、不做语义验证（只有 Fast 自带的确定性检查）；
- Always-Agent：每个问题都走最重的路径——从 Deep 开始（计划、分步检索），再经 Verified（语义检查与定向补检索）；
- Adaptive：本产品的做法——Router 按问题选模式，Fast 检索后按信号升级到 Deep，命中强制验证清单的经 Verified。
计分与 python -m eaer.evalrun.fast 相同（标准答案中的取值都出现、拒答类不泄露取值）；“验证器”类不适用，不计。
成本按生成调用的 token 数计（含 Verified 的语义检查与 Deep 的规划），乘以所用模型的单价即为金额（单价随服务商
调整，不写进工具）。耗时是单个查询从提问到回答的时间（不含回放）。
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from eaer.adapters.authorizer import LocalAuthorizer
from eaer.evalrun.fast import FastResult, run_engine, score
from eaer.evalrun.license import add_external_flag, emit, provenance_of, refuse_early
from eaer.evalrun.retrieval import run
from eaer.retrieval.service import Scope

CONFIGS = [("hybrid", "传统 Hybrid RAG"), ("agent", "Always-Agent"), ("adaptive", "Adaptive（本产品）")]
GOOD = ("correct", "refused_ok")
SCORED = ("correct", "wrong", "refused_ok", "leaked", "error")


@dataclass
class Row:
    result: FastResult
    calls: dict = field(default_factory=dict)       # 各类模型调用次数


def _calls(cost: dict) -> dict:
    return {k: v.get("calls", 0) for k, v in (cost.get("by_kind") or {}).items()}


def run_config(name: str, conn, fs, corpus: Path, manifest: dict, *, model, store, index, embedder, reranker) -> list[Row]:
    from factset.answer import Calculator

    from eaer.adapters.online import collect_calls, summarize
    from eaer.orchestrator.engine import Engine
    from eaer.orchestrator.fast import FastPipeline
    from eaer.orchestrator.forced_verify import add_rule

    if name == "adaptive":
        return [Row(r) for r in run_engine(conn, fs, corpus, manifest, model=model, store=store, index=index,
                                           embedder=embedder, reranker=reranker)]
    calc = Calculator(fs)
    rows: list[Row] = []
    ruled: set[str] = set()

    def ask(conn, fs, q, retriever, ids, tenant_id):
        if q.type == "验证器":
            res = score(fs, calc, q, None)
            rows.append(Row(res))
            return res
        scope = Scope(tenant_id, frozenset({q.role}))
        with collect_calls() as calls:
            try:
                if name == "hybrid":
                    a = FastPipeline(conn, retriever, model, LocalAuthorizer(conn, {}), store=store).answer(
                        scope, q.question, today=q.asked_on)
                    res = score(fs, calc, q, a)
                    res.mode_path = ["fast"]
                else:
                    if tenant_id not in ruled:          # 强制验证清单加一条“全部”：每个问题都经 Verified
                        from datetime import datetime, timezone
                        add_rule(conn, tenant_id, "all", None, "compare3", datetime.now(timezone.utc))
                        ruled.add(tenant_id)
                    # 不论 Router 怎么判断，都从 Deep 开始（模式提示只升不降，Router 判为 Verified 的问题不会经 Deep，
                    # 所以这里直接改起始模式；只用于评测）；强制验证清单的“全部”保证随后经 Verified
                    eng = Engine(conn, retriever, model, LocalAuthorizer(conn, {}), store=store)
                    eng.scope = scope
                    st = eng.prepare(q.question, today=q.asked_on)
                    st["mode_path"] = ["deep"]
                    r = eng.result(eng.graph().invoke(st), "compare3")
                    from eaer.orchestrator.fast import FastAnswer
                    a = FastAnswer(r.answer, r.claims, r.abstained, r.notices, r.forced_verify, r.time_basis, r.mode,
                                   r.mode_path, r.model, {k: v for k, v in r.cost.get("by_kind", {}).get("generate", {}).items()
                                                          if k in ("input_tokens", "output_tokens")}, r.seconds)
                    res = score(fs, calc, q, a)
                    res.mode_path = r.mode_path
            except Exception as e:  # noqa: BLE001 — 记录错误并继续
                res = score(fs, calc, q, None, f"{type(e).__name__}: {e}")
        if name == "hybrid" and res.outcome != "error":
            gen = summarize(calls).get("by_kind", {}).get("generate", {})
            res.tokens = gen.get("input_tokens", 0) + gen.get("output_tokens", 0)
        rows.append(Row(res, _calls(summarize(calls))))
        return res

    run(conn, fs, corpus, manifest, store=store, index=index, embedder=embedder, ask=ask, reranker=reranker)
    return rows


def _p(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


def summarize_config(rows: list[Row]) -> dict:
    # “验证器”类给的是待验证的结论而不是问题，三种方式都不计（Adaptive 的运行器会对它计分，这里排除，分母一致）
    scored = [r.result for r in rows if r.result.outcome in SCORED and r.result.type != "验证器"]
    good = [r for r in scored if r.outcome in GOOD]
    secs = [r.seconds for r in scored if r.outcome != "error"]
    tokens = [r.tokens for r in scored]
    by_type: dict[str, list[int]] = {}
    for r in scored:
        by_type.setdefault(r.type, [0, 0])
        by_type[r.type][0] += r.outcome in GOOD
        by_type[r.type][1] += 1
    out = {"n": len(scored), "correct": len(good), "accuracy": len(good) / len(scored) if scored else float("nan"),
           "errors": sum(r.outcome == "error" for r in scored),
           "p50": statistics.median(secs) if secs else float("nan"), "p95": _p(secs, 0.95),
           "tokens_mean": statistics.mean(tokens) if tokens else 0.0, "by_type": by_type,
           "paths": {}}
    for r in scored:
        key = "→".join(r.mode_path or [])
        out["paths"][key] = out["paths"].get(key, 0) + 1
    return out


def report(stats: dict[str, dict], *, model: str, embedder: str, reranker: str | None, dataset: str) -> str:
    lines = ["# 三方对比报告（AC-8-11）", "",
             f"生成 {model}；向量化 {embedder}；重排 {reranker or '无'}；索引为内存参考实现；评测集 {dataset}。"
             "三种方式用同一份语料、同一套检索与同一个生成模型，按日期回放后回答同一批查询。", "",
             "| 方式 | 正确 | 正确率 | 耗时 P50（秒） | 耗时 P95（秒） | 每问生成 token | 执行路径 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for key, title in CONFIGS:
        s = stats[key]
        paths = "；".join(f"{p} {n}" for p, n in sorted(s["paths"].items(), key=lambda x: -x[1]))
        lines.append(f"| {title} | {s['correct']} / {s['n']} | {s['accuracy']:.1%} | {s['p50']:.2f} | {s['p95']:.2f} | "
                     f"{s['tokens_mean']:.0f} | {paths} |")
    types = sorted({t for s in stats.values() for t in s["by_type"]})
    lines += ["", "按查询类型的正确数：", "", "| 类型 | " + " | ".join(t for _, t in CONFIGS) + " |",
              "| --- |" + " --- |" * len(CONFIGS)]
    for t in types:
        cells = [f"{stats[k]['by_type'].get(t, [0, 0])[0]} / {stats[k]['by_type'].get(t, [0, 0])[1]}" for k, _ in CONFIGS]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", "说明：样本量小，差异需结合按类型的数字看；合成数据，结论待真实语料复核；耗时受本机模型服务负载影响。"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from eaer.adapters.online import OnlineModelClient
    from eaer.evalrun.analyzers import scratch_databases
    from eaer.retrieval.embed import HashingEmbedder
    from eaer.retrieval.index import MemoryIndex
    from eaer.storage import LocalFileStore
    from factset.load import load
    from factset.render import Renderer
    from factset.subset import subset

    p = argparse.ArgumentParser(prog="eaer.evalrun.compare3", description="三方对比：传统 Hybrid RAG、Always-Agent、Adaptive")
    p.add_argument("xlsx")
    p.add_argument("--admin-url", required=True)
    p.add_argument("--format", choices=["docx", "html"], default="html")
    p.add_argument("--split", choices=["标定", "测试"])
    p.add_argument("--per-type", type=int)
    p.add_argument("--only", choices=[k for k, _ in CONFIGS], action="append", help="只跑其中几种（可重复）")
    p.add_argument("--out", help="把报告写入文件")
    add_external_flag(p)
    args = p.parse_args(argv)
    if os.environ.get("EAER_ENV") != "eval":
        print("只在评测环境运行（EAER_ENV=eval）", file=sys.stderr)
        return 2
    prov = [provenance_of(args.xlsx)]
    if (code := refuse_early(prov, external=args.external)) is not None:
        return code
    model = OnlineModelClient.from_env()
    if model.missing_for_fast():
        print(f"三方对比需要真实的模型服务：缺 {'、'.join(model.missing_for_fast())}", file=sys.stderr)
        return 2
    embedder = model.embedder() if model.emb is not None else HashingEmbedder()
    fs = subset(load(args.xlsx), split=args.split, per_type=args.per_type)
    stats: dict[str, dict] = {}
    keys = args.only or [k for k, _ in CONFIGS]
    with tempfile.TemporaryDirectory() as tmp, scratch_databases(args.admin_url) as new_db:
        import psycopg
        from psycopg.rows import dict_row

        for key in keys:
            work = Path(tmp) / key
            manifest = Renderer(fs).render(work / "corpus", args.format)
            conn = psycopg.connect(new_db(), autocommit=True, row_factory=dict_row)
            print(f"—— {dict(CONFIGS)[key]} ——", file=sys.stderr)
            rows = run_config(key, conn, fs, work / "corpus", manifest, model=model, store=LocalFileStore(work / "files"),
                              index=MemoryIndex(), embedder=embedder, reranker=model)
            conn.close()
            stats[key] = summarize_config(rows)
    for key, _ in CONFIGS:
        stats.setdefault(key, {"n": 0, "correct": 0, "accuracy": float("nan"), "errors": 0, "p50": float("nan"),
                               "p95": float("nan"), "tokens_mean": 0.0, "by_type": {}, "paths": {}})
    text = report(stats, model=model.gen.model, embedder=embedder.name, reranker=model.rr.model if model.rr else None,
                  dataset=prov[0].detail or prov[0].dataset)
    code = emit(text, prov, external=args.external, out=args.out)
    return code


if __name__ == "__main__":
    sys.exit(main())
