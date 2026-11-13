"""从事实表计算每个查询的标准答案，并与人工填写的预期答案比对（开发文档 AC-14-3）。

规则与开发文档第 6–9 节一致：
- 系统认知时间：回溯查询取回溯时间点，其余取提问日；
- 判断有效的日期：填了生效时间点取它，否则取系统认知时间；
- 归档状态与可访问角色按提问日判断，回溯查询也不例外；
- 冲突时 explicit 优先；最高等级的候选取值仍不一致时不代为选择。
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date

from .model import GRADE_RANK, Fact, FactSet, Query, weakest
from .replay import Timeline


@dataclass
class Expected:
    query_id: str
    refuse: bool
    basis: tuple[str, ...] = ()
    grade: str | None = None
    values: tuple[str, ...] = ()       # 回答中必须出现的取值，已带单位
    answer: str = "拒答"               # 规范化的答案文本
    unverified: bool = False           # 时效结论需标注“未验证”
    conflict: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def fmt(f: Fact) -> str:
    v = f"{f.value:,}" if isinstance(f.value, (int, float)) else str(f.value)
    return f"{v} {f.unit}" if f.unit else v


class Calculator:
    def __init__(self, fs: FactSet):
        self.fs = fs
        self.tl = Timeline(fs)

    def _version(self, f: Fact):
        return self.fs.versions[f.version_id]

    def _visible(self, q: Query, known_at: date) -> list[Fact]:
        out = []
        for f in self.fs.facts.values():
            if f.key != q.target_key:
                continue
            v = self._version(f)
            doc = self.fs.documents[v.document_id]
            if (self.tl.approved(v, known_at)
                    and not self.tl.archived(v, q.asked_on)
                    and q.role in self.tl.roles(doc, q.asked_on)):
                out.append(f)
        return out

    def compute(self, q: Query) -> Expected:
        known_at = q.as_of_system or q.asked_on
        on = q.as_of_valid or known_at
        cands = self._visible(q, known_at)
        if not cands:
            return Expected(q.id, True, reason="没有可见的候选事实（不存在、无权限、已归档或尚未审核通过）")
        if q.type == "版本差异":
            return self._diff(q, cands, known_at)

        valid = [f for f in cands if self.tl.in_force(self._version(f), on, known_at)]
        if not valid:
            return Expected(q.id, True, reason=f"{on} 没有有效的候选事实")
        ranked = sorted(valid, key=lambda f: self._rank(f, known_at), reverse=True)
        grades = {f.id: self.tl.grade(self._version(f), known_at) for f in ranked}
        distinct = {f.value for f in ranked}
        basis = tuple(sorted(f.id for f in ranked))
        top = ranked[0]
        conflict = len(distinct) > 1
        if conflict:
            top_rank = self._rank(top, known_at)[0]
            tops = [f for f in ranked if self._rank(f, known_at)[0] == top_rank]
            if len({f.value for f in tops}) > 1:
                # 同等可信的候选互相矛盾：全部列出，不代为选择
                return Expected(q.id, False, basis, grades[top.id], tuple(fmt(f) for f in ranked),
                                "存在冲突：" + "；".join(fmt(f) for f in ranked) + "，并标注“未验证”",
                                unverified=True, conflict=True, reason="同等可信的候选取值不一致")
        grade = grades[top.id]
        doc = self.fs.documents[self._version(top).document_id]
        unverified = doc.normative and grade != "explicit"

        if q.type == "验证器":
            claimed = re.findall(r"\d[\d,]*(?:\.\d+)?", q.question)
            claimed_v = float(claimed[-1].replace(",", "")) if claimed else None
            ok = claimed_v is not None and isinstance(top.value, (int, float)) and claimed_v == float(top.value)
            return Expected(q.id, False, (top.id,), grade, (), "支持" if ok else "不支持",
                            reason=f"结论取值 {claimed_v}，事实取值 {top.value}")

        values = tuple(fmt(f) for f in ([top] + [f for f in ranked if f.value != top.value]))
        answer = fmt(top)
        if unverified:
            answer += "，并标注“未验证”"
        if conflict:
            answer += "，并提示另有文件写 " + "、".join(values[1:])
        return Expected(q.id, False, basis, grade, values, answer, unverified, conflict)

    def _rank(self, f: Fact, known_at: date) -> tuple[int, date]:
        v = self._version(f)
        g = self.tl.grade(v, known_at)
        rank = GRADE_RANK[g] if g is not None else GRADE_RANK["explicit"]
        valid_from, _ = self.tl.validity(v, known_at)
        return rank, (v.doc_date or valid_from or v.approved_on)

    def _diff(self, q: Query, cands: list[Fact], known_at: date) -> Expected:
        by_version = {f.version_id: f for f in cands}
        pairs = []
        for f in cands:
            v = self._version(f)
            s = self.tl.successor(v, known_at)
            if s is not None and s.id in by_version:
                pairs.append((f, by_version[s.id]))
        if not pairs:
            return Expected(q.id, True, reason="没有可比较的前后两个版本")
        if q.as_of_valid:
            pairs = [p for p in pairs if self.tl.validity(self._version(p[1]), known_at)[0] == q.as_of_valid] or pairs
        old, new = max(pairs, key=lambda p: self._version(p[1]).version_no)
        grade = weakest(self.tl.grade(self._version(old), known_at), self.tl.grade(self._version(new), known_at))
        unverified = grade not in (None, "explicit")
        return Expected(q.id, False, tuple(sorted((old.id, new.id))), grade, (fmt(old), fmt(new)),
                        f"由 {fmt(old)} 调整为 {fmt(new)}" + ("，并标注“未验证”" if unverified else ""), unverified=unverified)


def compare(q: Query, e: Expected) -> list[str]:
    """人工答案与计算结果的差异；空列表表示一致。"""
    diffs = []
    if q.should_refuse != e.refuse:
        diffs.append(f"是否应拒答：人工 {q.should_refuse}，计算 {e.refuse}（{e.reason}）")
        return diffs
    if e.refuse:
        return diffs
    if set(q.basis) != set(e.basis):
        diffs.append(f"依据事实：人工 {'; '.join(q.basis)}，计算 {'; '.join(e.basis)}")
    if (q.expected_grade or None) != e.grade:
        diffs.append(f"预期来源等级：人工 {q.expected_grade}，计算 {e.grade}")
    if q.type == "验证器":
        if e.answer not in (q.expected_answer or ""):
            diffs.append(f"预期答案应为“{e.answer}”")
        return diffs
    for v in e.values:
        if v not in (q.expected_answer or ""):
            diffs.append(f"预期答案中缺少取值“{v}”")
    marked = "未验证" in (q.expected_answer or "")
    if e.unverified and not marked:
        diffs.append("结论只有 inferred 或 unknown 等级支撑，预期答案应标注“未验证”")
    if marked and not e.unverified:
        diffs.append("结论有 explicit 等级支撑，预期答案不应标注“未验证”")
    return diffs
