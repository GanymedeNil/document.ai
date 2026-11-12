"""分类与时间抽取（开发文档 6.2 第 3 阶段、7.2）。规则优先；规则无法判断的留给模型建议或审核人。

每个抽取结果都带原句位置；没有原句位置的结果不采用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

CN_DIGITS = {"〇": 0, "零": 0, "○": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}

_ARABIC_CN = r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
_CHINESE_CN = r"[〇○零一二三四五六七八九]{4}\s*年\s*[一二三四五六七八九十]{1,3}\s*月\s*[一二三四五六七八九十]{1,3}\s*日"
_ISO = r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
DATE = rf"(?:{_ARABIC_CN}|{_CHINESE_CN}|{_ISO})"

EFFECTIVE = re.compile(rf"(?:自|从)?\s*({DATE})\s*起\s*(?:施行|执行|生效|实施|启用)")
EXPIRE = re.compile(rf"有效期(?:至|到)\s*({DATE})|({DATE})\s*起\s*(?:废止|停止执行|失效|不再执行)")
RELATIVE = re.compile(r"自(?:发布|印发|公布)之日起\s*(?:(\d+|[一二三四五六七八九十百]+)\s*(?:日|天)后)?\s*(?:施行|执行|生效|实施)")
ABOLISH = re.compile(r"原?《([^》]+)》\s*(?:[（(]\s*第\s*(\d+)\s*版\s*[)）])?\s*(?:同时)?(?:废止|不再执行|停止执行)")
DOC_DATE = re.compile(rf"(?:报告日期|日期|印发日期|发布日期)\s*[:：]\s*({DATE})")
LONE_DATE = re.compile(rf"^\s*({DATE})\s*$")

NORMATIVE_WORDS = ("办法", "规定", "制度", "规范", "标准", "细则", "合同", "协议", "规程", "守则", "准则")
RECORD_WORDS = ("报告", "纪要", "总结", "分析", "方案", "记录", "简报", "周报", "月报")


def _cn_small(s: str) -> int:
    """一到三十一的中文数字。"""
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + CN_DIGITS[s[1]]
    if s.endswith("十"):
        return CN_DIGITS[s[0]] * 10
    if "十" in s:
        a, b = s.split("十")
        return CN_DIGITS[a] * 10 + CN_DIGITS[b]
    return CN_DIGITS[s]


def parse_date(text: str) -> date | None:
    s = re.sub(r"\s+", "", text)
    m = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", s) or re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    try:
        if m:
            return date(int(m[1]), int(m[2]), int(m[3]))
        m = re.fullmatch(r"([〇○零一二三四五六七八九]{4})年([一二三四五六七八九十]{1,3})月([一二三四五六七八九十]{1,3})日", s)
        if m:
            year = int("".join(str(CN_DIGITS[c]) for c in m[1]))
            return date(year, _cn_small(m[2]), _cn_small(m[3]))
    except (ValueError, KeyError):
        return None
    return None


def classify(title: str | None) -> tuple[str | None, str]:
    """按标题判断规范类或记录类；判断不了返回 None，交给模型建议或审核人。"""
    t = (title or "").strip()
    if any(w in t for w in RECORD_WORDS):
        return "record", "标题含记录类用词"
    if any(t.endswith(w) or w in t for w in NORMATIVE_WORDS) or t.endswith("通知"):
        return "normative", "标题含规范类用词"
    return None, "规则无法判断"


@dataclass
class Found:
    value: object
    locator: dict
    sentence: str


@dataclass
class Extraction:
    effective: Found | None = None
    expires: Found | None = None
    relative: Found | None = None          # 无法换算的相对表述
    doc_date: Found | None = None
    abolishes: list[Found] = field(default_factory=list)   # value 为 (标题, 版本号或 None)

    def to_json(self) -> dict:
        def f(x: Found | None):
            if x is None:
                return None
            v = x.value.isoformat() if isinstance(x.value, date) else x.value
            return {"value": v, "locator": x.locator, "sentence": x.sentence}
        return {"effective": f(self.effective), "expires": f(self.expires), "relative": f(self.relative),
                "doc_date": f(self.doc_date),
                "abolishes": [{"title": a.value[0], "version_no": a.value[1], "locator": a.locator,
                               "sentence": a.sentence} for a in self.abolishes]}


def _sentence(text: str, start: int, end: int) -> str:
    left = max(text.rfind(c, 0, start) for c in "。；;！？\n") + 1
    rights = [i for i in (text.find(c, end) for c in "。；;！？\n") if i != -1]
    return text[left:(min(rights) + 1 if rights else len(text))].strip()


def extract(paragraphs: list[tuple[dict, str]]) -> Extraction:
    """paragraphs：按文档顺序的 (定位, 段落文字)。同类结果取第一处出现的。"""
    ex = Extraction()
    for loc, text in paragraphs:
        if ex.effective is None and (m := EFFECTIVE.search(text)):
            d = parse_date(m[1])
            if d:
                ex.effective = Found(d, loc, _sentence(text, m.start(), m.end()))
        if ex.expires is None and (m := EXPIRE.search(text)):
            d = parse_date(m[1] or m[2])
            if d:
                ex.expires = Found(d, loc, _sentence(text, m.start(), m.end()))
        if ex.relative is None and (m := RELATIVE.search(text)):
            ex.relative = Found(m[0], loc, _sentence(text, m.start(), m.end()))
        if ex.doc_date is None and (m := DOC_DATE.search(text)):
            d = parse_date(m[1])
            if d:
                ex.doc_date = Found(d, loc, m[0])
        for m in ABOLISH.finditer(text):
            ex.abolishes.append(Found((m[1], int(m[2]) if m[2] else None), loc, _sentence(text, m.start(), m.end())))
    # 落款日期：最后一段只有一个日期
    if ex.doc_date is None and paragraphs:
        loc, text = paragraphs[-1]
        if (m := LONE_DATE.match(text)) and (d := parse_date(m[1])):
            ex.doc_date = Found(d, loc, text.strip())
    return ex
