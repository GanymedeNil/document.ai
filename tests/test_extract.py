"""抽取规则单元测试（开发文档 7.2）。"""
from __future__ import annotations

from datetime import date

import pytest

from eaer.ingestion.extract import classify, extract, parse_date


@pytest.mark.parametrize("text, expected", [
    ("2026年7月1日", date(2026, 7, 1)),
    ("2026 年 7 月 1 日", date(2026, 7, 1)),
    ("二〇二六年七月一日", date(2026, 7, 1)),
    ("二〇二五年十二月三十一日", date(2025, 12, 31)),
    ("二〇二六年十月二十日", date(2026, 10, 20)),
    ("2026-03-01", date(2026, 3, 1)),
    ("2026/3/1", date(2026, 3, 1)),
    ("2026年2月30日", None),
    ("October 30, 2023", date(2023, 10, 30)),
    ("30 October 2023", date(2023, 10, 30)),
    ("1st Feb. 2024", date(2024, 2, 1)),
    ("Sept 9, 2023", date(2023, 9, 9)),
    ("31 September 2023", None),
    ("october 30, 2023", None),           # 月份名要首字母大写（“may”是情态动词）
])
def test_parse_date(text, expected):
    assert parse_date(text) == expected


def para(*texts):
    return [({"para": i}, t) for i, t in enumerate(texts)]


def test_effective_date_and_abolition():
    ex = extract(para("第一条　总则。", "第五条　本办法自2026年7月1日起施行。原《差旅报销管理办法》（第2版）同时废止。"))
    assert ex.effective.value == date(2026, 7, 1) and ex.effective.locator == {"para": 1}
    assert ex.effective.sentence == "第五条　本办法自2026年7月1日起施行。"
    assert [a.value for a in ex.abolishes] == [("差旅报销管理办法", 2)]


@pytest.mark.parametrize("text, field, value", [
    ("本规定自二〇二六年三月一日起执行。", "effective", date(2026, 3, 1)),
    ("2026年1月1日起实施。", "effective", date(2026, 1, 1)),
    ("本办法有效期至2027年12月31日。", "expires", date(2027, 12, 31)),
    ("本通知自2026年9月1日起废止。", "expires", date(2026, 9, 1)),
    ("报告日期：2026年7月10日", "doc_date", date(2026, 7, 10)),
])
def test_single_patterns(text, field, value):
    assert getattr(extract(para(text)), field).value == value


def test_effective_not_confused_with_expiry():
    ex = extract(para("本办法自2026年7月1日起施行，原办法同时废止。"))
    assert ex.effective.value == date(2026, 7, 1) and ex.expires is None


def test_relative_wording_is_kept_unresolved():
    ex = extract(para("本办法自发布之日起30日后施行。"))
    assert ex.effective is None and ex.relative is not None
    assert "30日后" in ex.relative.sentence


def test_signature_date_as_doc_date():
    ex = extract(para("一、概述", "以上。", "2026年7月10日"))
    assert ex.doc_date.value == date(2026, 7, 10)


@pytest.mark.parametrize("texts, expected", [
    (("Published: October 30, 2023", "Body."), date(2023, 10, 30)),
    (("Body.", "Issued on 5 March 2024 by the Commission."), date(2024, 3, 5)),
    (("Report date: 2024-03-05",), date(2024, 3, 5)),
    (("Date: 1 May 2024", "Body."), date(2024, 5, 1)),
    (("Body.", "October 30, 2023"), date(2023, 10, 30)),                 # 落款
    (("Annual report", "30 October 2023", "Body."), date(2023, 10, 30)),  # 标题下的日期
    (("Due date: 1 May 2024",), None),                                    # 不是文档日期
    (("Expiry date: 2024-03-05",), None),
    (("Published in October 2023.",), None),                              # 只有年月，不是日期
    (("A", "B", "C", "30 October 2023", "D"), None),                      # 不在前三段、也不是最后一段
    (("一、概述", "2026年7月10日", "正文。"), None),                     # 中文日期在落款，标题下的不按这条
])
def test_english_doc_date(texts, expected):
    """英文报告与新闻的文档日期（2026-11-15，MultiHop-RAG 与 ViDoRe 的英文文档原先一份也抽不到）：“Published/Issued/
    Released (on)”“Dated”后、“Publication date:”“Report date:”后、段首“Date:”后的日期，落款日期，以及前三段中
    只有一个英文日期的段落。"""
    ex = extract(para(*texts))
    assert (ex.doc_date and ex.doc_date.value) == expected


def test_abolition_without_version_and_other_wording():
    ex = extract(para("原《采购审批权限规定》不再执行。"))
    assert [a.value for a in ex.abolishes] == [("采购审批权限规定", None)]


def test_no_dates_found():
    ex = extract(para("第一条　为规范相关工作，制定本规定。", "第二条　部门经理单笔采购审批上限为50,000元。"))
    assert (ex.effective, ex.expires, ex.relative, ex.doc_date, ex.abolishes) == (None, None, None, None, [])


@pytest.mark.parametrize("title, expected", [
    ("差旅报销管理办法", "normative"),
    ("采购审批权限规定", "normative"),
    ("关于差旅住宿标准的补充通知", "normative"),
    ("2026 年第二季度差旅费用分析报告", "record"),
    ("产品评审会议纪要", "record"),
    ("新员工手册", None),
])
def test_classify(title, expected):
    assert classify(title)[0] == expected
