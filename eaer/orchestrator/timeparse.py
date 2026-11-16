"""时间点的识别与回显（开发文档 7.5、AC-7-6）。

先用规则从问题中解析，接口参数 as_of_valid、as_of_system 优先于解析结果：
- “截至 D …系统（中）…记录 / 系统所知”：回溯查询，D 是系统认知时间；问题里若另有日期，那个是生效时间点，
  否则生效时间点也取 D（“截至 7 月 2 日系统中记录的上限”问的就是 7 月 2 日当时的有效值）；
- 其余完整日期（年月日）：生效时间点；
- 只有年、季度、月份（“2026 年第二季度”）不是时间点，是记录类文档的主题；
- “当时”“那时”“去年”“上个月”等相对说法规则无法确定，交给模型解析（resolver 回调，可不提供）；
  模型也确定不了时不给日期，原文保留在 unresolved 中（7.1“无法换算的相对表述保留原文”）。
解析出的时间点写在回答开头回显，如“按 2026 年 6 月 2 日生效的版本回答”。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

_DATE = re.compile(r"(\d{4})\s*(?:年|-|/|\.)\s*(\d{1,2})\s*(?:月|-|/|\.)\s*(\d{1,2})\s*[日号]?")
_SYSTEM = re.compile(r"截[至止]\s*$")
_SYSTEM_AFTER = re.compile(r"^[，,\s]*[^，,。；？?]{0,12}(系统|记录|所知)")
RELATIVE = re.compile(r"当时|那时|彼时|去年|前年|上个?月|上周|上季度|此前|之前|以前|\d+\s*(?:天|个月|年)前")


@dataclass
class TimeBasis:
    as_of_valid: date | None = None
    as_of_system: date | None = None
    source: str = "today"                 # parameter / parsed / model / today
    unresolved: list[str] = field(default_factory=list)

    @property
    def has_point(self) -> bool:
        return self.as_of_valid is not None or self.as_of_system is not None

    def echo(self) -> list[str]:
        out = []
        if self.as_of_system:
            d = self.as_of_system
            out.append(f"按截至 {d.year} 年 {d.month} 月 {d.day} 日的系统记录回答")
        if self.as_of_valid:
            d = self.as_of_valid
            out.append(f"按 {d.year} 年 {d.month} 月 {d.day} 日生效的版本回答")
        if self.unresolved:
            out.append(f"问题中的“{'、'.join(self.unresolved)}”无法换算为具体日期，按当前生效的版本回答")
        return out

    def to_json(self) -> dict:
        return {"as_of_valid": self.as_of_valid and self.as_of_valid.isoformat(),
                "as_of_system": self.as_of_system and self.as_of_system.isoformat(), "source": self.source,
                "unresolved": self.unresolved}


def _dates(question: str) -> list[tuple[date, int, int]]:
    out = []
    for m in _DATE.finditer(question):
        try:
            out.append((date(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.start(), m.end()))
        except ValueError:
            continue
    return out


def parse(question: str) -> TimeBasis:
    """只用规则解析；无法确定的相对说法放在 unresolved。"""
    tb = TimeBasis()
    valid: list[date] = []
    for d, start, end in _dates(question):
        if _SYSTEM.search(question[:start]) and _SYSTEM_AFTER.search(question[end:]):
            tb.as_of_system = d
        elif _SYSTEM.search(question[:start]) and "系统" in question[:start][-8:]:
            tb.as_of_system = d
        else:
            valid.append(d)
    if valid:
        tb.as_of_valid = valid[0]
    elif tb.as_of_system:
        tb.as_of_valid = tb.as_of_system
    if tb.has_point:
        tb.source = "parsed"
    elif RELATIVE.search(question):
        tb.unresolved = sorted(set(RELATIVE.findall(question)))
    return tb


Resolver = Callable[[str, date], tuple[date | None, date | None]]


def resolve(question: str, *, today: date, as_of_valid: date | None = None, as_of_system: date | None = None,
            resolver: Resolver | None = None) -> TimeBasis:
    """接口参数优先；其次规则；规则确定不了的相对说法再交给模型（resolver）。"""
    if as_of_valid or as_of_system:
        return TimeBasis(as_of_valid or as_of_system, as_of_system, "parameter")
    tb = parse(question)
    if tb.unresolved and resolver is not None:
        v, s = resolver(question, today)
        if v or s:
            return TimeBasis(v or s, s, "model")
    return tb
