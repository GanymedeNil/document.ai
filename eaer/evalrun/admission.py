"""准入检查与能力矩阵（开发文档 12.3、9.3、9.5、AC-12-3、AC-12-6）。

客户在自己的环境中用这一套评测集跑自己的模型，数据不出域：
1. 验证器（语义检查所用的模型）：由事实表生成条目，每个事实有正例（复述）与反例（改数值、换版本、换主体、
   改单位或周期、改结论）。正例、反例各自按条目编号的哈希排序后交替分入标定集与评测集，两半互不交叉、规模相当
   （AC-9-5，划分写进报告）。先在标定集上为 semantic_min_confidence 选出误通过率不超过门槛的最低阈值，再在
   评测集上按该阈值算误通过率与误拒率，两者都在门槛内、且正例反例数够，才准入 Verified。随机作答或一律判支持的
   模型误通过率高，一律判未验证的模型误拒率高，都过不了；
2. Deep：在装载好的事实表语料上，以最后一个日期为提问日，问版本比较（修订前后、第 N 版与第 M 版）与多对象比较
   （不用图表里的事实，也不用有冲突的取值），标准答案的取值都出现在回答里算对，正确率达标才准入 Deep；
3. 事实表的全部查询照常经编排器回答，得到正确率与各路径的耗时；
4. 汇总为能力矩阵：各模式是否可用、正确率、误通过率与误拒率、响应时间、运行环境。
指定 --database-url 与 --tenant 时，把结论写入产品库（model_admission，只追加），并把标定出的阈值写入
tenant_config.thresholds；部署时据此开关 Verified 与 Deep（eaer/orchestrator/admission.py）。

    EAER_ENV=eval python -m eaer.evalrun.admission 事实表.xlsx --admin-url postgresql://…（建临时评测库用）\\
        [--database-url postgresql://…（产品库） --tenant t1[,t2]] [--out 能力矩阵.md]

样本只有一份事实表，结果说明的是这份样例上的表现；门槛是初值，基线评测后与用户一起回填（16.2）。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path

from eaer.evalrun.license import add_external_flag, emit, provenance_of, refuse_early
from eaer.adapters.authorizer import LocalAuthorizer
from eaer.orchestrator.admission import DEFAULT_CRITERIA
from eaer.retrieval.service import Scope
from eaer.verifier.deterministic import extract
from eaer.verifier.semantic import SemanticChecker

CANDIDATE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
UNIT_SWAP = {"元/晚": "元/月", "元/天": "元/月", "元/月": "元/天", "元": "万元", "万元": "元"}


# ---------------------------------------------------------------- 验证器

@dataclass(frozen=True)
class VerifierItem:
    id: str
    kind: str              # 复述 / 改数值 / 换版本 / 换主体 / 改单位 / 改结论
    claim: str
    evidence: str
    supported: bool


def _num(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def _changed(v):
    if isinstance(v, (int, float)):
        w = round(v * 1.2)
        return w if w != v else v + 1
    return f"{v}（另）"


def verifier_items(fs, max_facts: int = 40) -> list[VerifierItem]:
    """每个事实生成一组正反例；事实很多时按编号哈希取 max_facts 个（大评测集上准入不必逐条问模型）。"""
    facts = sorted(fs.facts.values(), key=lambda f: f.id)
    if len(facts) > max_facts:
        keep = set(sorted((f.id for f in facts), key=lambda i: hashlib.sha1(i.encode()).hexdigest())[:max_facts])
        others = facts
        facts = [f for f in facts if f.id in keep]
    else:
        others = facts
    out: list[VerifierItem] = []
    for i, f in enumerate(facts):
        v = fs.versions[f.version_id]
        d = fs.documents[v.document_id]
        unit = f.unit or ""
        val = f"{_num(f.value)}{unit}"
        ev = f"《{d.title}》第 {v.version_no} 版规定：{f.key}为{val}。"

        def add(kind, claim, ok):
            out.append(VerifierItem(f"{f.id}-{kind}", kind, claim, ev, ok))

        add("复述", f"{f.key}为{val}。", True)
        add("转述", f"根据《{d.title}》，{f.key}是{val}。", True)
        add("改数值", f"{f.key}为{_num(_changed(f.value))}{unit}。", False)
        other = next((g for g in facts[i + 1:] + facts[:i] + others if g.key != f.key), None)
        if other is not None:
            add("换主体", f"{other.key}为{val}。", False)
        if unit in UNIT_SWAP:
            add("改单位", f"{f.key}为{_num(f.value)}{UNIT_SWAP[unit]}。", False)
        add("改结论", f"{f.key}没有统一标准，由各部门自行决定。", False)
        sibling = next((g for g in others if g.key == f.key and g.version_id != f.version_id and g.value != f.value
                        and fs.versions[g.version_id].document_id == v.document_id), None)
        if sibling is not None:
            add("换版本", f"《{d.title}》第 {v.version_no} 版规定{f.key}为{_num(sibling.value)}{sibling.unit or ''}。", False)
    return out


def split(items: list[VerifierItem]) -> tuple[list[VerifierItem], list[VerifierItem]]:
    """标定集与评测集：正例、反例各自按编号哈希排序后交替分配，互不交叉、规模相当。"""
    calib, evaluation = [], []
    for label in (True, False):
        group = sorted((i for i in items if i.supported is label), key=lambda i: hashlib.sha1(i.id.encode()).hexdigest())
        for k, it in enumerate(group):
            (calib if k % 2 == 0 else evaluation).append(it)
    return calib, evaluation


@dataclass
class Judged:
    item: VerifierItem
    status: str
    confidence: float


def judge_all(model, items: list[VerifierItem], on: date) -> list[Judged]:
    """语义检查原始结论（不套阈值），阈值在标定时离线套用。"""
    sem = SemanticChecker(model, min_confidence=0.0)
    out = []
    for n, it in enumerate(items):
        v = sem.judge(it.claim, [(f"ev_{n:016x}", it.evidence)], on)
        out.append(Judged(it, v.status, v.confidence))
    return out


def passes(j: Judged, t: float) -> bool:
    return j.status == "supported" and j.confidence >= t


def rates(results: list[Judged], t: float) -> dict:
    pos = [j for j in results if j.item.supported]
    neg = [j for j in results if not j.item.supported]
    return {"threshold": t, "positives": len(pos), "negatives": len(neg),
            "false_pass": round(sum(passes(j, t) for j in neg) / len(neg), 4) if neg else None,
            "false_reject": round(sum(not passes(j, t) for j in pos) / len(pos), 4) if pos else None}


def calibrate(results: list[Judged], target_false_pass: float) -> float:
    """误通过率不超过目标的最低阈值（召回最高）；都达不到时取最严的。"""
    ok = [t for t in CANDIDATE_THRESHOLDS if (rates(results, t)["false_pass"] or 0.0) <= target_false_pass]
    return min(ok) if ok else max(CANDIDATE_THRESHOLDS)


def check_verifier(model, fs, criteria: dict, on: date) -> dict:
    if not model.capabilities().structured_output:
        return {"ok": False, "reason": "所用模型不支持结构化输出", "threshold": None}
    calib, evaluation = split(verifier_items(fs))
    jc, je = judge_all(model, calib, on), judge_all(model, evaluation, on)
    t = calibrate(jc, criteria["verified_max_false_pass"])
    ev = rates(je, t)
    by_kind = defaultdict(lambda: [0, 0])
    for j in je:
        by_kind[j.item.kind][0] += passes(j, t) == j.item.supported
        by_kind[j.item.kind][1] += 1
    problems = []
    if ev["positives"] < criteria["verified_min_positive"] or ev["negatives"] < criteria["verified_min_negative"]:
        problems.append(f"评测集太小（正例 {ev['positives']}、反例 {ev['negatives']}）")
    if ev["false_pass"] is not None and ev["false_pass"] > criteria["verified_max_false_pass"]:
        problems.append(f"误通过率 {ev['false_pass']:.0%} 超过 {criteria['verified_max_false_pass']:.0%}")
    if ev["false_reject"] is not None and ev["false_reject"] > criteria["verified_max_false_reject"]:
        problems.append(f"误拒率 {ev['false_reject']:.0%} 超过 {criteria['verified_max_false_reject']:.0%}")
    return {"ok": not problems, "reason": "；".join(problems) or "达标", "threshold": t,
            "calibration": rates(jc, t), "evaluation": ev,
            "by_kind": {k: f"{a}/{b}" for k, (a, b) in sorted(by_kind.items())},
            "split": {"calibration": sorted(i.id for i in calib), "evaluation": sorted(i.id for i in evaluation)}}


# ---------------------------------------------------------------- Deep

@dataclass(frozen=True)
class DeepItem:
    kind: str
    question: str
    expected: tuple[str, ...]
    role: str


def deep_items(fs, max_pairs: int = 4, max_items: int = 20) -> list[DeepItem]:
    """只用正文与表格里的事实：图表里的取值要视觉模型解析（12.1），不该算作 Deep 的错。"""
    facts = [f for f in fs.facts.values() if f.carrier != "图表"]
    versions = defaultdict(list)
    for v in fs.versions.values():
        versions[v.document_id].append(v)
    fact = {(f.version_id, f.key): f for f in facts}

    def val(f):
        return f"{_num(f.value)}{f.unit or ''}"

    def role(*doc_ids):
        common = frozenset.intersection(*(fs.documents[d].roles for d in doc_ids))
        return sorted(common)[0] if common else None

    out: list[DeepItem] = []
    for doc_id, vs in sorted(versions.items()):
        vs.sort(key=lambda v: v.version_no)
        r = role(doc_id)
        for a, b in zip(vs, vs[1:]):
            for (vid, key), fa in sorted(fact.items()):
                fb = fact.get((b.id, key))
                if vid == a.id and fb and fb.value != fa.value and b.valid_from:
                    out.append(DeepItem("修订前后", f"{b.valid_from.year} 年 {b.valid_from.month} 月修订前后，{key}有什么变化？",
                                        (val(fa), val(fb)), r))
        for a, b in combinations(vs, 2):
            for (vid, key), fa in sorted(fact.items()):
                fb = fact.get((b.id, key))
                if vid == a.id and fb and fb.value != fa.value:
                    out.append(DeepItem("版本号", f"第 {a.version_no} 版和第 {b.version_no} 版的{key}有什么区别？",
                                        (val(fa), val(fb)), r))
    current = {}
    for doc_id, vs in versions.items():
        last = max(vs, key=lambda v: v.version_no)
        for f in facts:
            if f.version_id == last.id:
                current.setdefault(f.key, []).append((doc_id, f))
    unique = sorted((k, docs[0]) for k, docs in current.items() if len(docs) == 1)   # 有冲突的取值不拿来比较
    for (k1, (d1, f1)), (k2, (d2, f2)) in list(combinations(unique, 2))[:max_pairs]:
        r = role(d1, d2)
        if r:
            out.append(DeepItem("多对象", f"{k1}和{k2}分别是多少？", (val(f1), val(f2)), r))
    if len(out) > max_items:                     # 大评测集上按问题哈希取 max_items 题，各类型都留一些
        out = sorted(out, key=lambda d: hashlib.sha1(d.question.encode()).hexdigest())[:max_items]
    return out


def _quantities(text: str):
    return {q for q, _ in extract(text).quantities}


def deep_correct(answer: str, abstained: bool, expected: tuple[str, ...]) -> bool:
    have = _quantities(answer or "")
    return not abstained and all(_quantities(e) <= have for e in expected)


# ---------------------------------------------------------------- 汇总

def _p(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 2)


def environment(model) -> dict:
    mem = None
    try:
        mem = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    endpoints = {}
    for name in ("gen", "emb", "rr"):
        ep = getattr(model, name, None)
        if ep is not None:
            endpoints[name] = {"model": ep.model, "host": ep.base_url.split("/")[2] if "//" in ep.base_url else ep.base_url}
    return {"platform": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count(), "memory_gb": mem,
            "python": platform.python_version(), "endpoints": endpoints}


def run_admission(conn, fs, corpus: Path, manifest: dict, *, model, store, index, embedder, reranker=None,
                  criteria: dict | None = None, tenant_id: str = "eval") -> dict:
    from eaer.evalrun.fast import FastAnswer, score
    from eaer.evalrun.retrieval import run
    from eaer.orchestrator.engine import Engine
    from eaer.retrieval.service import Retriever
    from factset.answer import Calculator

    crit = {**DEFAULT_CRITERIA, **(criteria or {})}
    calc = Calculator(fs)
    dates = [q.asked_on for q in fs.queries] + [e.occurred_on for e in fs.events] + \
            [v.approved_on for v in fs.versions.values()]
    final = max(dates)
    started = time.monotonic()
    verified = check_verifier(model, fs, crit, final)

    def engine(c, retriever):
        return Engine(c, retriever, model, LocalAuthorizer(c, {}), store=store)

    rows = []

    def ask(c, fs_, q, retriever, ids, tid):
        if q.type == "验证器":
            return None
        try:
            r = engine(c, retriever).ask(Scope(tid, frozenset({q.role})), q.question, today=q.asked_on)
        except Exception as e:  # noqa: BLE001 — 单个查询失败记为错误，继续评测
            rows.append({"query": q.id, "outcome": "error", "path": [], "seconds": 0.0, "error": f"{type(e).__name__}: {e}"})
            return None
        res = score(fs, calc, q, FastAnswer(r.answer, r.claims, r.abstained, r.notices, r.forced_verify, r.time_basis,
                                            seconds=r.seconds))
        rows.append({"query": q.id, "outcome": res.outcome, "path": r.mode_path, "seconds": r.seconds})
        return None

    run(conn, fs, corpus, manifest, store=store, index=index, embedder=embedder, ask=ask, reranker=reranker,
        tenant_id=tenant_id)
    retriever = Retriever(conn, index, embedder, reranker, top_m=10)
    deep_rows = []
    caps = model.capabilities()
    deep_capable = caps.structured_output
    for it in deep_items(fs) if deep_capable else []:
        t0 = time.monotonic()
        try:
            r = engine(conn, retriever).ask(Scope(tenant_id, frozenset({it.role})), it.question, today=final,
                                            mode_hint="deep")
        except Exception as e:  # noqa: BLE001 — 记为答错，并留下原因
            deep_rows.append({"kind": it.kind, "question": it.question, "expected": list(it.expected), "correct": False,
                              "path": [], "seconds": round(time.monotonic() - t0, 2),
                              "answer": f"出错：{type(e).__name__}: {e}"[:120]})
            continue
        deep_rows.append({"kind": it.kind, "question": it.question, "expected": list(it.expected),
                          "correct": deep_correct(r.answer, r.abstained, it.expected), "path": r.mode_path,
                          "seconds": round(time.monotonic() - t0, 2), "answer": (r.answer or "")[:120]})
    n = len(deep_rows)
    acc = round(sum(d["correct"] for d in deep_rows) / n, 4) if n else None
    deep_problems = []
    if not deep_capable:
        deep_problems.append("所用模型不支持结构化输出")
    elif n < crit["deep_min_questions"]:
        deep_problems.append(f"评测题太少（{n}）")
    elif acc < crit["deep_min_accuracy"]:
        deep_problems.append(f"正确率 {acc:.0%} 低于 {crit['deep_min_accuracy']:.0%}")
    scored = [x for x in rows if x["outcome"] != "not_applicable"]
    errors = [x for x in rows if x["outcome"] == "error"] + [d for d in deep_rows if d["answer"].startswith("出错：")]
    good = sum(x["outcome"] in ("correct", "refused_ok") for x in scored)
    paths = defaultdict(list)
    for x in scored:
        paths["→".join(x["path"])].append(x["seconds"])
    gen = getattr(model, "gen", None)
    return {"model": {"generation": getattr(gen, "model", type(model).__name__),
                      "embedding": getattr(getattr(model, "emb", None), "model", None),
                      "rerank": getattr(getattr(model, "rr", None), "model", None),
                      "capabilities": asdict(caps)},
            "criteria": crit, "asked_on": final.isoformat(),
            "verified": verified,
            "deep": {"ok": not deep_problems, "reason": "；".join(deep_problems) or "达标", "accuracy": acc,
                     "questions": n, "items": deep_rows,
                     "p50": _p([d["seconds"] for d in deep_rows], 0.5), "p95": _p([d["seconds"] for d in deep_rows], 0.95)},
            "queries": {"correct": good, "scored": len(scored), "items": rows,
                        "p50": _p([x["seconds"] for x in scored], 0.5), "p95": _p([x["seconds"] for x in scored], 0.95),
                        "by_path": {k: {"n": len(v), "p50": _p(v, 0.5), "p95": _p(v, 0.95)} for k, v in sorted(paths.items())}},
            "errors": len(errors), "environment": environment(model), "seconds": round(time.monotonic() - started, 1)}


def matrix(report: dict) -> str:
    v, d, q, m = report["verified"], report["deep"], report["queries"], report["model"]
    ev = v.get("evaluation") or {}

    def pct(x):
        return "—" if x is None else f"{x:.0%}"

    lines = [f"# 能力矩阵：{m['generation']}",
             "",
             f"Embedding {m['embedding'] or '未配置'}；重排 {m['rerank'] or '未配置'}；提问日 {report['asked_on']}；"
             f"耗时 {report['seconds']} 秒；模型调用出错 {report.get('errors', 0)} 次。样本是事实表样例，门槛为初值，不是产品基线。",
             "",
             "| 模式 | 是否开启 | 指标 | 响应时间 | 说明 |",
             "| --- | --- | --- | --- | --- |",
             f"| Fast | 是 | 事实表查询正确 {q['correct']}/{q['scored']} | P50 {q['p50']} 秒、P95 {q['p95']} 秒 | 全部查询经编排器回答 |",
             f"| Verified | {'是' if v['ok'] else '否'} | 误通过率 {pct(ev.get('false_pass'))}、误拒率 {pct(ev.get('false_reject'))}"
             f"（评测集正例 {ev.get('positives', 0)}、反例 {ev.get('negatives', 0)}；阈值 {v.get('threshold')}） | 见各路径 | {v['reason']} |",
             f"| Deep | {'是' if d['ok'] else '否'} | 正确率 {pct(d['accuracy'])}（{d['questions']} 题） | "
             f"P50 {d['p50']} 秒、P95 {d['p95']} 秒 | {d['reason']} |",
             "",
             "## 各路径耗时", "", "| 路径 | 次数 | P50 | P95 |", "| --- | --- | --- | --- |"]
    lines += [f"| {k} | {x['n']} | {x['p50']} | {x['p95']} |" for k, x in q["by_path"].items()]
    if v.get("by_kind"):
        lines += ["", "## 验证器（评测集，按类型判对数）", "", "| 类型 | 判对 |", "| --- | --- |"]
        lines += [f"| {k} | {x} |" for k, x in v["by_kind"].items()]
    if d["items"]:
        lines += ["", "## Deep 评测题", "", "| 类型 | 问题 | 标准答案 | 结果 | 路径 |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| {x['kind']} | {x['question']} | {'、'.join(x['expected'])} | {'对' if x['correct'] else '错'} | "
                  f"{'→'.join(x['path'])} |" for x in d["items"]]
    e = report["environment"]
    lines += ["", "## 运行环境", "",
              f"{e['platform']}；{e['machine']}；CPU {e['cpus']} 核；内存 {e['memory_gb']} GB；Python {e['python']}。",
              "模型服务：" + "；".join(f"{k} {x['model']}（{x['host']}）" for k, x in e["endpoints"].items()) + "。",
              "硬件要求以此为参考：线上模型的算力在服务方；本地 Embedding 与重排服务（modelserver）的机器另计。"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from eaer.adapters.online import OnlineModelClient
    from eaer.evalrun.analyzers import scratch_databases
    from eaer.retrieval.embed import HashingEmbedder
    from eaer.retrieval.index import MemoryIndex
    from eaer.storage import LocalFileStore
    from factset.load import load
    from factset.render import Renderer

    p = argparse.ArgumentParser(prog="eaer.evalrun.admission", description="模型准入检查与能力矩阵")
    p.add_argument("xlsx")
    p.add_argument("--admin-url", required=True, help="建临时评测库用的管理连接")
    p.add_argument("--database-url", help="产品库：写入准入结论与标定的阈值")
    p.add_argument("--tenant", help="写入哪些租户（逗号分隔），与 --database-url 一起用")
    p.add_argument("--out", help="能力矩阵写到这个文件（Markdown）")
    p.add_argument("--format", choices=["docx", "html"], default="docx")
    p.add_argument("--per-type", type=int, default=10, help="能力矩阵中事实表查询每类最多取几条（默认 10）")
    add_external_flag(p)
    args = p.parse_args(argv)
    if os.environ.get("EAER_ENV") != "eval":
        print("只在评测环境运行（EAER_ENV=eval）", file=sys.stderr)
        return 2
    prov = [provenance_of(args.xlsx)]
    if (code := refuse_early(prov, external=args.external)) is not None:
        return code
    if bool(args.database_url) != bool(args.tenant):
        print("--database-url 与 --tenant 要一起给", file=sys.stderr)
        return 2
    model = OnlineModelClient.from_env()
    if "生成" in model.missing_for_fast():
        print("未配置生成服务（EAER_GEN_*）", file=sys.stderr)
        return 2
    embedder = model.embedder() if model.emb is not None else HashingEmbedder()
    reranker = model if model.rr is not None else None
    from factset.subset import subset
    fs = subset(load(args.xlsx), split="测试" if any(q.split for q in load(args.xlsx).queries) else None,
                per_type=args.per_type)
    with tempfile.TemporaryDirectory() as tmp, scratch_databases(args.admin_url) as new_db:
        import psycopg
        from psycopg.rows import dict_row

        work = Path(tmp)
        manifest = Renderer(fs).render(work / "corpus", args.format)
        with psycopg.connect(new_db(), autocommit=True, row_factory=dict_row) as conn:
            report = run_admission(conn, fs, work / "corpus", manifest, model=model,
                                   store=LocalFileStore(work / "files"), index=MemoryIndex(), embedder=embedder,
                                   reranker=reranker)
    report["dataset"] = {"file": Path(args.xlsx).name, "provenance": prov[0].dataset, "modified": prov[0].modified,
                         "sha256": hashlib.sha256(Path(args.xlsx).read_bytes()).hexdigest()}
    if (code := emit(matrix(report), prov, external=args.external, out=args.out)) != 0:
        return code
    if args.database_url:
        import psycopg
        from psycopg.rows import dict_row

        from eaer.orchestrator.admission import record
        with psycopg.connect(args.database_url, autocommit=True, row_factory=dict_row) as conn:
            for t in [x.strip() for x in args.tenant.split(",") if x.strip()]:
                aid = record(conn, t, report["model"]["generation"], verified_ok=report["verified"]["ok"],
                             deep_ok=report["deep"]["ok"], report=report, now=datetime.now(timezone.utc),
                             thresholds={"semantic_min_confidence": report["verified"]["threshold"]}
                             if report["verified"].get("threshold") is not None else None)
                print(f"已写入租户 {t} 的准入结论：{aid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
