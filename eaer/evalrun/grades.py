"""来源等级如实（开发文档 7、9.2，AC-7-3 硬性）：对合成集逐条比对，不一致数应为 0。

按日期回放事实表（与检索评测相同，见 retrieval.run），每个查询在它的提问日经完整编排器回答，然后逐条核对：
- 回答中引用依据事实所在证据时，标注的来源等级（响应里引用的 valid_grade）等于事实表在系统认知时间的等级
  （factset 的 Timeline.grade：生效日与作废日中较弱的一个，记录类为空）；
- 规范类文档只由 inferred 或 unknown（含没有生效日）支撑的结论标为“未验证”（结论状态 unverified，回答文字带“（未验证）”），
  由 explicit 支撑的不标，记录类不标。

比的是等级与标注，不是召回，所以用两个替身把与等级无关的部分固定下来：
- 检索：在真实检索结果前面加上依据事实所在的证据。它们照样经过数据库核对（状态、权限、有效期、回溯时间点）——
  核对不通过就不会出现，计为“依据未进入回答”；只是不依赖排序能否把它们排进前几条。
- 模型：每条看到的依据证据各写一条结论（原文照抄，数值核对必然一致），语义检查一律判支持；
  Deep 比较题的对象留空（按版本比较），其余规划取默认。
时间参数直接取事实表中的生效时间点与回溯时间点（时间识别的准确率另由 AC-7-6、AC-7-7 评测）。
不调用任何模型服务，结果可复现。拒答类、图表类（需要视觉模型）与验证器类查询不在此列。

    EAER_ENV=eval python -m eaer.evalrun.grades docs/评测集/合成集-v2.xlsx --admin-url postgresql://… [--split 测试]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from eaer.adapters.authorizer import LocalAuthorizer
from eaer.adapters.model import Capabilities, Completion, Usage
from eaer.evalrun.license import add_external_flag, emit, provenance_of, refuse_early
from eaer.evalrun.retrieval import norm, run
from eaer.lifecycle.asof import end_of_day
from eaer.retrieval.embed import HashingEmbedder
from eaer.retrieval.index import MemoryIndex
from eaer.retrieval.service import Scope
from factset.answer import Calculator
from factset.model import FactSet

LABEL = "（未验证）"
SKIP_TYPES = {"验证器"}


@dataclass
class GradeCheck:
    query_id: str
    type: str
    fact_id: str
    expected_grade: str | None
    actual_grade: str | None       # 回答引用中标注的等级；依据没有进入回答时为 None
    expect_unverified: bool
    status: str | None             # supported / unverified；依据没有进入回答时为 None
    labelled: bool                 # 回答文字带“（未验证）”
    outcome: str                   # ok / grade_mismatch / label_mismatch / uncovered


class _OracleRetriever:
    """真实检索结果前面加上依据证据；它们同样经过数据库核对（_recheck），核对规则与真实检索完全相同。"""

    def __init__(self, inner, targets: list[str]):
        self.inner, self.targets = inner, targets

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def search(self, scope, question, *, today, as_of_valid=None, as_of_system=None):
        hits = self.inner.search(scope, question, today=today, as_of_valid=as_of_valid, as_of_system=as_of_system)
        system_t = end_of_day(as_of_system) if as_of_system else None
        on = as_of_valid or as_of_system or today
        forced = self.inner._recheck(scope, [(e, 1.0) for e in self.targets], on, system_t)
        have = {h.evidence_id for h in forced}
        return forced + [h for h in hits if h.evidence_id not in have]


class _OracleModel:
    """每条看到的依据证据写一条结论（原文照抄）；语义检查判支持；规划取默认。"""

    def __init__(self, targets: list[str], key: str):
        self.targets, self.key = targets, key

    def capabilities(self):
        return Capabilities(True, 128000, False, 0)

    def generate(self, messages, schema):
        p = "\n".join(m.content for m in messages)
        seen = [t for t in self.targets if f"[{t}]" in p]
        props = (schema or {}).get("properties", {})

        def out(parsed):
            return Completion(json.dumps(parsed, ensure_ascii=False), parsed, Usage(1, 1), "oracle")

        if "verdict" in props:
            return out({"verdict": "supported", "confidence": 1.0, "supporting_ids": seen, "reason": "依据证据"})
        if "as_of_valid" in props:
            return out({"as_of_valid": ""})
        if "field" in props:
            return out({"field": self.key, "objects": []})
        if "first_hop" in props:
            return out({"first_hop": self.key})
        if "phenomenon" in props:
            return out({"phenomenon": self.key, "change_query": self.key})
        if "closed" in props:
            return out({"closed": True})
        if "queries" in props:
            return out({"queries": []})
        if not seen:
            return out({"abstain": True, "claims": [], "conflicts": [], "missing": self.key})
        content = {m.group(1): m.group(2) for m in re.finditer(r"\[(ev_[0-9a-f]{16})\][^\n]*\n(.*?)\n<<<证据结束", p, re.S)}
        return out({"abstain": False, "conflicts": [], "missing": "", "claims": [
            {"text": content.get(t, self.key), "evidence_ids": [t], "core": True} for t in seen]})


def run_grades(conn, fs: FactSet, corpus: Path, manifest: dict, *, store, index=None, embedder=None) -> list[GradeCheck]:
    from eaer.orchestrator.engine import Engine
    calc = Calculator(fs)
    checks: list[GradeCheck] = []

    def ask(conn, fs, q, retriever, ids, tenant_id):
        if q.type in SKIP_TYPES:
            return None
        exp = calc.compute(q)
        if exp.refuse:
            return None
        facts = [fs.facts[i] for i in exp.basis]
        if not facts or any(f.carrier == "图表" for f in facts):
            return None
        known_at = q.as_of_system or q.asked_on
        by_fact: dict[str, list[str]] = {}
        for f in facts:
            want = norm(f"{f.value}{f.unit or ''}")
            rows = conn.execute("SELECT id, content FROM evidence WHERE version_id = %s ORDER BY id",
                                (ids[f.version_id],)).fetchall()
            by_fact[f.id] = [r["id"] for r in rows if want in norm(r["content"])]
        targets = list(dict.fromkeys(e for es in by_fact.values() for e in es))
        eng = Engine(conn, _OracleRetriever(retriever, targets), _OracleModel(targets, q.target_key),
                     LocalAuthorizer(conn, {}), store=store)
        r = eng.ask(Scope(tenant_id, frozenset({q.role})), q.question, today=q.asked_on,
                    as_of_valid=q.as_of_valid, as_of_system=q.as_of_system)
        shown = r.response()["claims"]
        labels = _labels(r.answer, shown)
        for f in facts:
            v = fs.versions[f.version_id]
            grade = calc.tl.grade(v, known_at)
            expect_unv = fs.documents[v.document_id].normative and grade != "explicit"
            mine = [k for k, c in enumerate(shown) if any(x["evidence_id"] in by_fact[f.id] for x in c["citations"])]
            if not mine:
                checks.append(GradeCheck(q.id, q.type, f.id, grade, None, expect_unv, None, False, "uncovered"))
                continue
            for k in mine:
                c, labelled = shown[k], labels[k]
                cited = next(x for x in c["citations"] if x["evidence_id"] in by_fact[f.id])
                if cited["valid_grade"] != grade:
                    outcome = "grade_mismatch"
                elif (c["status"] == "unverified") != expect_unv or labelled != expect_unv:
                    outcome = "label_mismatch"
                else:
                    outcome = "ok"
                checks.append(GradeCheck(q.id, q.type, f.id, grade, cited["valid_grade"], expect_unv, c["status"],
                                         labelled, outcome))
        return None

    run(conn, fs, corpus, manifest, store=store, index=index or MemoryIndex(), embedder=embedder or HashingEmbedder(),
        ask=ask)
    return checks


def _labels(answer: str, claims: list[dict]) -> list[bool]:
    """回答文字中每条结论后是否紧跟“（未验证）”。按结论在回答中的先后逐条定位：
    两条结论文字相同（例如两个版本的同一条款没改）时，不能只按文字查找。"""
    pos, out = 0, []
    for c in claims:
        i = answer.find(c["text"], pos)
        if i < 0:
            out.append(False)
            continue
        end = i + len(c["text"])
        out.append(answer.startswith(LABEL, end))
        pos = end
    return out


def report(checks: list[GradeCheck]) -> str:
    c = Counter(x.outcome for x in checks)
    bad = c["grade_mismatch"] + c["label_mismatch"]
    queries = len({x.query_id for x in checks})
    lines = [f"来源等级核对（AC-7-3）：{queries} 个查询、{len(checks)} 条依据",
             f"不一致：{bad}（等级不符 {c['grade_mismatch']}，“未验证”标注不符 {c['label_mismatch']}）——硬性项要求为 0",
             f"一致：{c['ok']}；依据未进入回答：{c['uncovered']}"]
    by_grade = Counter((x.expected_grade, x.outcome) for x in checks)
    for g in ("explicit", "inferred", "unknown", None):
        n = sum(v for (gg, _), v in by_grade.items() if gg == g)
        if n:
            lines.append(f"  预期等级 {g or '不适用（记录类）'}：{n} 条，一致 {by_grade[(g, 'ok')]}")
    for x in checks:
        if x.outcome != "ok":
            lines.append(f"  · {x.query_id}（{x.type}）{x.fact_id}：{x.outcome}，预期 {x.expected_grade}"
                         f"{'、未验证' if x.expect_unverified else ''}，实际 {x.actual_grade}、{x.status}"
                         f"{'、带标注' if x.labelled else ''}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from eaer.evalrun.analyzers import scratch_databases
    from eaer.storage import LocalFileStore
    from factset.load import load
    from factset.render import Renderer
    from factset.subset import subset

    p = argparse.ArgumentParser(prog="eaer.evalrun.grades", description="来源等级如实（AC-7-3）：对合成集逐条比对")
    p.add_argument("xlsx")
    p.add_argument("--admin-url", required=True)
    p.add_argument("--format", choices=["docx", "html"], default="html")
    p.add_argument("--split", choices=["标定", "测试"])
    p.add_argument("--per-type", type=int)
    add_external_flag(p)
    args = p.parse_args(argv)
    if os.environ.get("EAER_ENV") != "eval":
        print("只在评测环境运行（EAER_ENV=eval）", file=sys.stderr)
        return 2
    prov = [provenance_of(args.xlsx)]
    if (code := refuse_early(prov, external=args.external)) is not None:
        return code
    fs = subset(load(args.xlsx), split=args.split, per_type=args.per_type)
    with tempfile.TemporaryDirectory() as tmp, scratch_databases(args.admin_url) as new_db:
        import psycopg
        from psycopg.rows import dict_row

        work = Path(tmp)
        manifest = Renderer(fs).render(work / "corpus", args.format)
        with psycopg.connect(new_db(), autocommit=True, row_factory=dict_row) as conn:
            checks = run_grades(conn, fs, work / "corpus", manifest, store=LocalFileStore(work / "files"))
    code = emit(report(checks), prov, external=args.external)
    return code or (1 if any(x.outcome in ("grade_mismatch", "label_mismatch") for x in checks) else 0)


if __name__ == "__main__":
    sys.exit(main())
