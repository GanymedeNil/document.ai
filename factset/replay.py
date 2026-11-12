"""按时间回放：系统在某一天知道什么。

与开发文档第 4、6、7 节一致：
- 版本在审核通过日起对系统可见；
- 有效期更正自事件发生日起生效，此前的回溯看不到；
- 旧版本的作废日由取代它的新版本推导（7.3）；
- 归档、取消归档和权限变更按提问日判断。
"""
from __future__ import annotations

from datetime import date

from .model import Document, FactSet, Version, weakest


class Timeline:
    def __init__(self, fs: FactSet):
        self.fs = fs
        self._successor = {v.supersedes: v for v in fs.versions.values() if v.supersedes}

    def approved(self, v: Version, t: date) -> bool:
        return v.approved_on <= t

    def validity(self, v: Version, t: date) -> tuple[date | None, str | None]:
        """生效日及其来源等级，取系统在 t 时记录的值。"""
        valid_from, grade = v.valid_from, v.grade
        for e in self.fs.events:
            if e.occurred_on > t:
                break
            if e.type == "有效期更正" and e.target == v.id:
                if e.new_valid_from is not None:
                    valid_from = e.new_valid_from
                    grade = e.new_grade or "explicit"  # 审核人补录记为 explicit
                elif e.new_grade is not None:
                    grade = e.new_grade
        return valid_from, grade

    def successor(self, v: Version, t: date) -> Version | None:
        s = self._successor.get(v.id)
        return s if s is not None and self.approved(s, t) else None

    def valid_to(self, v: Version, t: date) -> tuple[date | None, str | None]:
        """作废日（不再有效的第一天）及其等级；没有已知的新版本时为 (None, None)。"""
        s = self.successor(v, t)
        if s is None:
            return None, None
        s_from, s_grade = self.validity(s, t)
        if s_from is not None and s_grade == "explicit":
            return s_from, "explicit"
        return s.approved_on, "inferred"

    def archived(self, v: Version, t: date) -> bool:
        state = False
        for e in self.fs.events:
            if e.occurred_on > t:
                break
            if e.type in ("归档", "取消归档") and e.target in (v.id, v.document_id):
                state = e.type == "归档"
        return state

    def roles(self, d: Document, t: date) -> frozenset[str]:
        roles = d.roles
        for e in self.fs.events:
            if e.occurred_on > t:
                break
            if e.type == "权限变更" and e.target == d.id and e.new_roles:
                roles = e.new_roles
        return roles

    def in_force(self, v: Version, on: date, known_at: date) -> bool:
        """版本在 on 这一天是否有效，按系统在 known_at 时的认知判断。"""
        doc = self.fs.documents[v.document_id]
        if not doc.normative:
            return v.doc_date is not None and v.doc_date <= on
        valid_from, _ = self.validity(v, known_at)
        valid_to, _ = self.valid_to(v, known_at)
        # 空值不过滤（开发文档 7.4）
        return (valid_from is None or valid_from <= on) and (valid_to is None or valid_to > on)

    def grade(self, v: Version, known_at: date) -> str | None:
        """支撑该版本有效性的来源等级：生效日与作废日中较弱的一个；记录类为 None。"""
        doc = self.fs.documents[v.document_id]
        if not doc.normative:
            return None
        _, g_from = self.validity(v, known_at)
        valid_to, g_to = self.valid_to(v, known_at)
        return weakest(g_from, g_to if valid_to is not None else None)
