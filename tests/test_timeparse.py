"""时间点识别与回显（开发文档 7.5；AC-7-6 的规则部分）：事实表样例全部问题的时间点按规则解析正确，
接口参数优先于解析结果，无法换算的相对说法不给日期并回显原文，交给模型解析的回调在规则无果时才调用。"""
from __future__ import annotations

from datetime import date

import pytest

from eaer.orchestrator.timeparse import TimeBasis, parse, resolve
from fixture import SAMPLE

D = date.fromisoformat
TODAY = D("2026-11-11")


@pytest.mark.parametrize("row", SAMPLE["查询"][1:], ids=[r[0] for r in SAMPLE["查询"][1:]])
def test_sample_questions(row):
    qid, question, want_valid, want_system = row[0], row[2], row[7], row[8]
    tb = parse(question)
    assert (tb.as_of_valid, tb.as_of_system) == (want_valid, want_system), qid


@pytest.mark.parametrize("q, valid, system", [
    ("2026-03-15 一线城市住宿上限", D("2026-03-15"), None),
    ("2026/3/15 一线城市住宿上限", D("2026-03-15"), None),
    ("截至2026年9月15日系统所知，住宿上限是多少", D("2026-09-15"), D("2026-09-15")),
    ("截止 2026 年 9 月 15 日的记录中，2025 年 6 月 15 日的上限", D("2025-06-15"), D("2026-09-15")),
    ("截至 2026 年 9 月 15 日，住宿上限是多少", D("2026-09-15"), None),        # 没说“系统/记录”：普通时间点
    ("2026 年 2 月 30 日的上限", None, None),                                 # 不存在的日期
])
def test_more_forms(q, valid, system):
    tb = parse(q)
    assert (tb.as_of_valid, tb.as_of_system) == (valid, system)


def test_parameters_take_precedence_over_parsed():
    tb = resolve("2026 年 3 月 15 日的上限", today=TODAY, as_of_valid=D("2025-01-01"))
    assert (tb.as_of_valid, tb.source) == (D("2025-01-01"), "parameter")
    tb = resolve("2026 年 3 月 15 日的上限", today=TODAY, as_of_system=D("2026-09-15"))
    assert (tb.as_of_valid, tb.as_of_system) == (D("2026-09-15"), D("2026-09-15"))


def test_relative_wording_goes_to_the_model_resolver_only_when_rules_fail():
    calls = []

    def resolver(q, today):
        calls.append(q)
        return D("2025-11-11"), None

    tb = resolve("去年这个时候的上限是多少", today=TODAY, resolver=resolver)
    assert (tb.as_of_valid, tb.source) == (D("2025-11-11"), "model") and len(calls) == 1
    resolve("2026 年 3 月 15 日的上限", today=TODAY, resolver=resolver)
    assert len(calls) == 1                                                    # 规则能定，不调用模型
    tb = resolve("当时的上限是多少", today=TODAY, resolver=lambda q, t: (None, None))
    assert not tb.has_point and tb.unresolved == ["当时"] and "无法换算" in tb.echo()[0]


def test_echo():
    assert TimeBasis(D("2026-06-02")).echo() == ["按 2026 年 6 月 2 日生效的版本回答"]
    assert TimeBasis(D("2025-06-15"), D("2026-09-15")).echo() == [
        "按截至 2026 年 9 月 15 日的系统记录回答", "按 2025 年 6 月 15 日生效的版本回答"]
    assert TimeBasis().echo() == []
