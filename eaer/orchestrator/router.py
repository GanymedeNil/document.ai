"""Router（开发文档 8.1）：按顺序判断入口模式。

1. 问题含明确日期、时间点或版本用语（截至、当时、修订前、旧版、第 N 版），或接口传入了时间参数 → Verified；
2. 问题含比较、因果或多跳用语（区别、对比、变化、不同、为什么、分别、哪些……的……）→ Deep；英文问题的同类用语
   （compare、difference、change、consistent、versus、both、respectively、why……）同样 → Deep；
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
VERSION = re.compile(r"截至|截止|当时|修订|旧版|新版|历史版本|之前的版本|第\s*[0-9一二三四五六七八九十]+\s*版")
COMPARE = re.compile(r"区别|对比|比较|变化|差异|异同|不同|为什么|原因|分别|前后|改成|改为|调整为|从[^？?]{0,20}[改调]"
                     r"|哪些[^？?]{0,20}的")
# 英文问题的比较与多跳用语（2026-11-15，MultiHop-RAG：“Between the … report published on …, and the … report …, was there
# a change …”“consistent”原先都进 Fast，只取到其中一篇就整题拒答）。不分大小写；“climate change”是名词，不算。
EN_COMPARE = re.compile(r"(?i)\b(?:compar(?:e|es|ed|ing|ison|isons)|differ(?:s|ed|ent|ently|ence|ences)?|"
                        r"(?<!climate )chang(?:e|es|ed|ing)|versus|vs\.?|contrast(?:s|ed|ing)?|"
                        r"(?:in)?consisten(?:t|tly|cy|cies)|respectively|both)\b")
# 英文问题的因果用语（Deep 的因果模板）
EN_CAUSAL = re.compile(r"(?i)\bwhy\b|\bfor\s+what\s+reasons?\b|\bwhat\s+(?:caused|causes|led\s+to)\b"
                       r"|\breasons?\s+(?:for|behind)\b")


@dataclass(frozen=True)
class Route:
    mode: str
    reason: str
    hinted: bool = False

    def to_json(self) -> dict:
        return {"decided": self.mode, "reason": self.reason, "hinted": self.hinted}


def decide(question: str, tb: TimeBasis, mode_hint: str | None = None) -> Route:
    version = bool(VERSION.search(question))
    compare = bool(COMPARE.search(question) or EN_COMPARE.search(question) or EN_CAUSAL.search(question))
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
