"""Router（开发文档 8.1）：按顺序判断入口模式。

1. 问题含明确日期、时间点或版本用语（截至、当时、修订前、旧版、第 N 版），或接口传入了时间参数 → Verified；
2. 问题含比较、因果或多跳用语（区别、对比、变化、为什么、分别、哪些……的……）→ Deep；
3. 其他 → Fast。
同时含版本用语与比较用语的“版本比较”（如“修订前后有什么变化”）入口为 Deep：Deep 的比较模板分别检索新旧版本；
它同时命中强制验证清单的“版本比较”，检索后照样转入 Verified（AC-8-1）。
接口的 mode_hint 只能把模式提高，不能降低（AC-8-4）。检索之后的再判断与升级在 engine 中完成（8.1、8.2）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from eaer.orchestrator.timeparse import TimeBasis

RANK = {"fast": 0, "deep": 1, "verified": 2}
VERSION = re.compile(r"截至|截止|当时|修订前|修订后|修订前后|旧版|新版|历史版本|之前的版本|第\s*[0-9一二三四五六七八九十]+\s*版")
COMPARE = re.compile(r"区别|对比|比较|变化|差异|异同|为什么|原因|分别|前后|哪些[^？?]{0,20}的")


@dataclass(frozen=True)
class Route:
    mode: str
    reason: str
    hinted: bool = False

    def to_json(self) -> dict:
        return {"decided": self.mode, "reason": self.reason, "hinted": self.hinted}


def decide(question: str, tb: TimeBasis, mode_hint: str | None = None) -> Route:
    version, compare = bool(VERSION.search(question)), bool(COMPARE.search(question))
    timed = tb.has_point or bool(tb.unresolved)
    if compare and (version or timed):
        route = Route("deep", "版本比较")
    elif timed or version:
        route = Route("verified", "含时间点或版本用语" + ("（接口参数）" if tb.source == "parameter" else ""))
    elif compare:
        route = Route("deep", "含比较、因果或多跳用语")
    else:
        route = Route("fast", "一般问题")
    if mode_hint in RANK and RANK[mode_hint] > RANK[route.mode]:
        return Route(mode_hint, f"{route.reason}；按 mode_hint 提高为 {mode_hint}", True)
    return route
