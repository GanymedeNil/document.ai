"""AC-9-1（硬性）：四项确定性检查各有不少于 20 个正例和 20 个反例（开发文档 9.2）。

另含 AC-9-2 的确定性部分：事实表样例 Q16（把 650 改成 680 的扰动结论）在真实入库、审核、检索得到的
证据上被 value_match 判为不支持；同一结论写 650 时通过。完整的 Verified 冒烟需要模型做结论拆分，待模型接入。
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

import psycopg
import pytest
from psycopg.rows import dict_row

from eaer.adapters.authorizer import LocalAuthorizer
from eaer.domain.evidence import content_hash, evidence_id
from eaer.domain.identity import Scope
from eaer.verifier.deterministic import DeterministicChecker, Target, locator_exists, time_valid, value_match

D = date.fromisoformat


# ---------------------------------------------------------------- value_match

VALUE_POS = [
    ("一线城市住宿上限为650元/晚", ["一线城市｜标准：650元/晚"]),
    ("上限为６５０元", ["住宿上限 650 元"]),                                       # 全角
    ("审批上限为10万元", ["审批上限为100,000元"]),                                  # 万与千分位
    ("审批上限为100000元", ["审批上限为10万元"]),
    ("差旅总费用为128万元", ["第二季度差旅总费用 1,280,000 元"]),
    ("本办法自2026年1月1日起施行", ["本办法自2026年1月1日起施行。"]),
    ("自2026-01-01起施行", ["本办法自2026年1月1日起施行"]),                        # 日期写法
    ("2026/3/1 起生效", ["自2026年3月1日起施行"]),
    ("2026年3月起执行", ["自2026年3月1日起施行"]),                                 # 只到月
    ("2026年起执行新标准", ["自2026年1月1日起施行"]),                              # 只到年
    ("费用增长5%", ["同比增长5%"]),
    ("伙食补贴每天80元，交通补贴每天50元", ["伙食补贴｜标准：80元/天", "交通补贴｜标准：50元/天"]),  # 多条证据合起来
    ("根据第三条，住宿上限为650元", ["第三条　一线城市住宿上限为每晚650元"]),
    ("第2版规定上限为600元", ["一线城市住宿上限为600元"]),                         # 序号不是取值
    ("一线城市住宿有上限", ["一线城市住宿上限为650元"]),                            # 没有数值
    ("上限为1.5万元", ["上限为15,000元"]),
    ("上限为5千元", ["上限为5000元"]),
    ("报销比例为0.5", ["报销比例为0.50"]),
    ("P002-v2 的上限为100000元", ["上限为100,000元"]),                            # 编号中的数字不是取值
    ("2025年12月18日审核通过，上限600元", ["2025-12-18 审核通过；上限 600 元"]),
    ("１２０元", ["120元"]),
    ("根据第5份通知，上限为650元", ["上限为650元"]),                                # “第N份”也是序号
]
VALUE_NEG = [
    ("一线城市住宿上限为680元/晚", ["一线城市｜标准：650元/晚"]),                    # Q16 扰动
    ("审批上限为20万元", ["审批上限为100,000元"]),
    ("审批上限为10万元", ["审批上限为10,000元"]),
    ("自2026年2月1日起施行", ["自2026年1月1日起施行"]),
    ("自2025年1月1日起施行", ["自2026年1月1日起施行"]),
    ("2026年3月起执行", ["自2026年1月1日起施行"]),
    ("2027年起执行", ["自2026年1月1日起施行"]),
    ("增长5%", ["增长了 5 元"]),                                                   # 百分比与数值不同
    ("伙食补贴每天80元，交通补贴每天60元", ["伙食补贴｜标准：80元/天", "交通补贴｜标准：50元/天"]),
    ("上限为650元", []),
    ("上限为650元", ["上限为6500元"]),
    ("上限为65元", ["上限为650元"]),
    ("上限为1.5万元", ["上限为1,500元"]),
    ("上限为5千元", ["上限为500元"]),
    ("总费用128万元", ["总费用 128 元"]),
    ("2026-01-02 起施行", ["2026年1月1日起施行"]),
    ("每晚650元，自2026年7月1日起", ["每晚650元，自2026年1月1日起"]),
    ("每晚650元，自2026年1月1日起", ["每晚600元，自2026年1月1日起"]),
    ("报销比例为0.5", ["报销比例为0.05"]),
    ("上限为１００元", ["上限为10元"]),
    ("第三条规定上限为3元", ["第三条　上限为650元"]),
]


@pytest.mark.parametrize("claim, ev", VALUE_POS)
def test_value_match_positive(claim, ev):
    assert value_match(claim, ev).outcome == "pass"


@pytest.mark.parametrize("claim, ev", VALUE_NEG)
def test_value_match_negative(claim, ev):
    r = value_match(claim, ev)
    assert r.outcome == "unsupported" and r.reason.startswith("证据中没有")


# ---------------------------------------------------------------- locator_exists

VID = "11111111-1111-1111-1111-111111111111"


def row(modality: str, locator, content: str = "一线城市住宿上限为650元") -> dict:
    h = content_hash(content)
    eid = evidence_id(VID, modality, locator, h) if isinstance(locator, dict) else "ev_x"
    return {"id": eid, "version_id": VID, "modality": modality, "locator": locator, "content": content,
            "content_hash": h}


LOC_POS = [
    ("text", {"section": [], "para": 0}), ("text", {"section": ["总则"], "para": 3}),
    ("text", {"section": [], "para": 1, "page": 1}), ("text", {"section": ["附则"], "para": 40, "page": 12}),
    ("text", {"section": ["第一章", "第一节"], "para": 7}), ("text", {"section": [], "para": 999}),
    ("text", {"section": ["总则"], "para": 2, "sentence": "第二句"}), ("text", {"section": ["差旅与住宿"], "para": 5}),
    ("table", {"section": [], "table": 0, "row": 1, "column": 1}), ("table", {"section": [], "table": 0, "row": 5, "column": 3}),
    ("table", {"section": [], "table": 1, "row": 2, "column": 1, "page": 2}),
    ("table", {"section": ["附件"], "table": 3, "row": 1, "column": 2}),
    ("table", {"section": ["附件", "表一"], "table": 0, "row": 100, "column": 10, "page": 30}),
    ("table", {"section": [], "table": 7, "row": 1, "column": 1}),
    ("figure", {"section": [], "figure": 0}), ("figure", {"section": [], "figure": 1, "page": 1}),
    ("figure", {"section": [], "figure": 0, "bbox": [0, 0, 1, 1]}),
    ("figure", {"section": [], "figure": 2, "bbox": [0.1, 0.2, 0.5, 0.9]}),
    ("figure", {"section": ["附图"], "figure": 3, "page": 4, "bbox": [0.25, 0.25, 0.75, 0.75]}),
    ("figure", {"section": [], "figure": 9, "page": 99}),
]


def _tamper(kind: str) -> dict:
    r = row("text", {"section": [], "para": 1})
    if kind == "content":
        r["content"] = "一线城市住宿上限为680元"
    elif kind == "locator":
        r["locator"] = {"section": [], "para": 2}
    elif kind == "id":
        r["id"] = "ev_" + "0" * 16
    elif kind == "hash":
        r["content_hash"] = content_hash("别的内容")
    return r


LOC_NEG = [
    ("不存在", None, True),
    ("文本缺段落", row("text", {"section": []}), True),
    ("段落为负", row("text", {"section": [], "para": -1}), True),
    ("段落是字符串", row("text", {"section": [], "para": "3"}), True),
    ("段落是布尔", row("text", {"section": [], "para": True}), True),
    ("页码为 0", row("text", {"section": [], "para": 1, "page": 0}), True),
    ("页码是字符串", row("text", {"section": [], "para": 1, "page": "1"}), True),
    ("章节不是列表", row("text", {"section": "总则", "para": 1}), True),
    ("章节含数字", row("text", {"section": ["总则", 2], "para": 1}), True),
    ("表格第 0 行是表头", row("table", {"section": [], "table": 0, "row": 0, "column": 1}), True),
    ("表格第 0 列是行名", row("table", {"section": [], "table": 0, "row": 1, "column": 0}), True),
    ("表格缺序号", row("table", {"section": [], "row": 1, "column": 1}), True),
    ("图表缺序号", row("figure", {"section": [], "page": 1}), True),
    ("区域只有三个数", row("figure", {"section": [], "figure": 0, "bbox": [0, 0, 1]}), True),
    ("区域左右颠倒", row("figure", {"section": [], "figure": 0, "bbox": [0.6, 0, 0.4, 1]}), True),
    ("区域超出页面", row("figure", {"section": [], "figure": 0, "bbox": [0, 0, 1.2, 1]}), True),
    ("未知模态", row("audio", {"section": [], "para": 1}), True),
    ("定位不是对象", row("text", ["para", 1]), True),
    ("内容被改", _tamper("content"), True),
    ("定位被改", _tamper("locator"), True),
    ("id 被改", _tamper("id"), True),
    ("哈希被改", _tamper("hash"), True),
    ("原件已删除", row("text", {"section": [], "para": 1}), False),
]


@pytest.mark.parametrize("modality, locator", LOC_POS)
def test_locator_exists_positive(modality, locator):
    assert locator_exists(row(modality, locator)) .outcome == "pass"


@pytest.mark.parametrize("name, r, raw", LOC_NEG, ids=[n[0] for n in LOC_NEG])
def test_locator_exists_negative(name, r, raw):
    assert locator_exists(r, raw_exists=raw).outcome == "unsupported"


# ---------------------------------------------------------------- time_valid

def V(vf=None, fg="unknown", vt=None, tg="unknown"):
    return {"valid_from": vf and D(vf), "valid_from_grade": fg, "valid_to": vt and D(vt), "valid_to_grade": tg}


ON = D("2026-06-01")
TIME_POS = [
    ("normative", None, V("2026-01-01", "explicit"), ON),
    ("normative", None, V("2026-06-01", "explicit"), ON),                         # 生效当天
    ("normative", None, V("2026-01-01", "explicit", "2026-06-02", "explicit"), ON),  # 作废前一天
    ("normative", None, V("2025-01-01", "explicit", "2027-01-01", "explicit"), ON),
    ("normative", None, V("2000-01-01", "explicit"), ON),
    ("normative", None, V("2026-05-31", "explicit", "2026-06-02", "explicit"), ON),
    (None, None, V("2026-01-01", "explicit"), ON),                                  # 类型未定按规范类
    ("normative", None, V("2026-01-01", "explicit"), D("2030-12-31")),
    ("normative", None, V("2026-01-01", "explicit", "2026-07-01", "explicit"), D("2026-06-30")),
    ("normative", None, V("2026-01-01", "explicit", "2026-07-01", "explicit"), D("2026-01-01")),
    ("record", D("2026-05-01"), None, ON),
    ("record", D("2026-06-01"), None, ON),                                          # 记录类当天
    ("record", D("2020-01-01"), None, ON),
    ("record", D("2026-05-31"), V(), ON),                                           # 记录类不看有效期
    ("normative", None, V("2025-03-01", "explicit", "2026-03-01", "explicit"), D("2025-03-01")),
    ("normative", None, V("2025-03-01", "explicit", "2026-03-01", "explicit"), D("2026-02-28")),
    ("normative", None, V("2026-07-01", "explicit"), D("2026-07-01")),
    ("normative", None, V("2026-07-01", "explicit"), D("2026-11-11")),
    ("normative", None, V("2024-02-29", "explicit", "2028-02-29", "explicit"), D("2026-02-28")),
    ("normative", None, V("2026-01-01", "explicit", "2099-12-31", "explicit"), ON),
]
TIME_NEG = [
    ("normative", None, V("2026-01-01", "explicit", "2026-06-01", "explicit"), ON, "unsupported"),   # 作废当天起无效
    ("normative", None, V("2025-01-01", "explicit", "2026-01-01", "explicit"), ON, "unsupported"),
    ("normative", None, V("2026-07-01", "explicit"), ON, "unsupported"),                             # 尚未生效
    ("normative", None, V("2026-06-02", "explicit"), ON, "unsupported"),
    ("normative", None, V("2030-01-01", "explicit", "2031-01-01", "explicit"), ON, "unsupported"),
    ("normative", None, None, ON, "unsupported"),                                                   # 那时没有记录
    ("record", D("2026-06-02"), None, ON, "unsupported"),                                           # 文档日期在后
    ("record", D("2027-01-01"), None, ON, "unsupported"),
    ("normative", None, V("2026-01-01", "explicit", "2026-05-01", "inferred"), ON, "unverified"),    # 推断作废
    ("normative", None, V("2026-01-01", "explicit", "2026-06-01", "inferred"), ON, "unverified"),
    ("normative", None, V("2026-07-01", "inferred"), ON, "unverified"),                              # 推断未生效
    ("normative", None, V(), ON, "unverified"),                                                     # 生效日为空
    ("normative", None, V(None, "unknown", "2027-01-01", "explicit"), ON, "unverified"),
    ("normative", None, V("2026-01-01", "inferred"), ON, "unverified"),                              # 仅由推断支撑
    ("normative", None, V("2026-01-01", "inferred", "2027-01-01", "explicit"), ON, "unverified"),
    ("normative", None, V("2026-01-01", "explicit", "2027-01-01", "inferred"), ON, "unverified"),
    (None, None, V("2026-01-01", "inferred"), ON, "unverified"),
    ("record", None, None, ON, "unverified"),                                                       # 记录类无日期
    ("normative", None, V("2025-01-01", "explicit", "2025-12-18", "inferred"), ON, "unverified"),
    ("normative", None, V("2026-01-01", "explicit", "2026-03-01", "explicit"), D("2026-03-01"), "unsupported"),
]


@pytest.mark.parametrize("doc_type, doc_date, val, on", TIME_POS)
def test_time_valid_positive(doc_type, doc_date, val, on):
    assert time_valid(doc_type, doc_date, val, on).outcome == "pass"


@pytest.mark.parametrize("doc_type, doc_date, val, on, want", TIME_NEG)
def test_time_valid_negative(doc_type, doc_date, val, on, want):
    assert time_valid(doc_type, doc_date, val, on).outcome == want


# ---------------------------------------------------------------- authorized（经真实授权接口与数据库）

@pytest.fixture(scope="module")
def corpus(template_db):
    from conftest import _admin, url_for
    from eaer.lifecycle.service import Lifecycle

    admin = _admin()
    name = f"eaer_v_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}" TEMPLATE "{template_db}"')
    conn = psycopg.connect(url_for(name), autocommit=True, row_factory=dict_row)
    lc = Lifecycle(conn)
    pos, neg = [], []

    def make(tenant, principals, state, text):
        lc.ensure_tenant(tenant)
        acl = lc.create_acl_policy(tenant, principals)
        doc = lc.create_document(tenant, f"d-{uuid.uuid4().hex[:6]}", text[:8], "normative")
        vid = lc.add_version(doc, content_hash=f"sha256:{uuid.uuid4().hex}", raw_object_uri="file://x",
                             acl_policy_id=acl, parser_version="t")
        lc.set_draft_validity(vid, valid_from=D("2026-01-01"), valid_from_grade="explicit")
        ev = lc.add_evidence(vid, "text", {"section": [], "para": 0}, text)
        if state != "pending_review":
            lc.publish(vid, "t")
        if state == "archived":
            lc.archive_document(doc, "t")
        return ev

    roles = [["R_A"], ["R_A", "R_B"], ["R_A", "R_C"], ["R_B", "R_A"], ["R_A", "R_B", "R_C"]]
    for i in range(20):
        pos.append(make("t1", roles[i % 5], "searchable", f"第{i}份文件规定上限为{100 + i}元"))
    for i in range(6):
        neg.append(make("t1", [["R_B"], ["R_C"], ["R_B", "R_C"]][i % 3], "searchable", f"无权文件{i}，上限{900 + i}元"))
    for i in range(5):
        neg.append(make("t1", ["R_A"], "pending_review", f"待审核文件{i}"))
    for i in range(5):
        neg.append(make("t1", ["R_A"], "archived", f"已归档文件{i}"))
    for i in range(3):
        neg.append(make("t2", ["R_A"], "searchable", f"其他租户文件{i}"))
    neg.append("ev_" + "f" * 16)
    checker = DeterministicChecker(conn, LocalAuthorizer(conn, {}), Scope("t1", frozenset({"R_A"})))
    yield checker, pos, neg
    conn.close()
    admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.close()


@pytest.mark.parametrize("i", range(20))
def test_authorized_positive(corpus, i):
    checker, pos, _ = corpus
    r = checker.check("该文件规定了上限", [pos[i]], retrieved={pos[i]}, target=Target(D("2026-06-01")))
    assert r.checks["authorized"].outcome == "pass" and r.removed == [] and r.alerts == []
    assert r.outcome == "pass"


@pytest.mark.parametrize("i", range(20))
def test_authorized_negative(corpus, i):
    checker, _, neg = corpus
    r = checker.check("该文件规定了上限", [neg[i]], retrieved={neg[i]}, target=None)
    assert r.checks["authorized"].outcome == "unsupported" and r.removed == [neg[i]]
    assert r.outcome == "unsupported" and len(r.alerts) == 1 and "安全告警" in r.alerts[0]


def test_unauthorized_evidence_is_removed_and_claim_rechecked_on_the_rest(corpus):
    checker, pos, neg = corpus
    r = checker.check("上限为100元", [neg[0], pos[0]], retrieved={neg[0], pos[0]}, target=None)
    assert r.removed == [neg[0]] and r.evidence_ids == [pos[0]] and r.outcome == "pass"
    r = checker.check("上限为900元", [neg[0], pos[0]], retrieved={neg[0], pos[0]}, target=None)
    assert r.outcome == "unsupported" and r.checks["value_match"].outcome == "unsupported"   # 900 元只在无权证据里


def test_citing_evidence_outside_this_retrieval_is_unsupported(corpus):
    checker, pos, _ = corpus
    r = checker.check("上限为100元", [pos[0]], retrieved={pos[1]}, target=None)
    assert r.outcome == "unsupported" and "检索结果以外" in r.checks["locator_exists"].reason
    assert checker.check("上限为100元", [], retrieved={pos[0]}, target=None).outcome == "unsupported"


# ---------------------------------------------------------------- AC-9-2 的确定性部分：事实表样例 Q16

def test_q16_perturbed_claim_is_unsupported_on_real_evidence(conn, tmp_path, monkeypatch):
    from eaer.evalrun.retrieval import run
    from eaer.retrieval.embed import HashingEmbedder
    from eaer.retrieval.index import MemoryIndex
    from eaer.retrieval.service import Retriever
    from eaer.storage import LocalFileStore
    from factset import load
    from factset.render import Renderer
    from fixture import sample, write

    monkeypatch.setenv("EAER_ENV", "eval")
    fs = load(write(sample(), tmp_path / "s.xlsx"))
    manifest = Renderer(fs).render(tmp_path / "corpus", "docx")
    index, emb = MemoryIndex(), HashingEmbedder()
    run(conn, fs, tmp_path / "corpus", manifest, store=LocalFileStore(tmp_path / "files"), index=index, embedder=emb)
    scope = Scope("eval", frozenset({"R_STAFF"}))
    asked = D("2026-11-11")
    hits = Retriever(conn, index, emb).search(scope, "一线城市的住宿报销上限", today=asked)
    basis = [h.evidence_id for h in hits if "650" in h.content]
    assert basis, [h.content for h in hits]
    checker = DeterministicChecker(conn, LocalAuthorizer(conn, {}), scope)
    retrieved = {h.evidence_id for h in hits}
    q16 = next(q for q in fs.queries if q.id == "Q16").question.removeprefix("结论：")
    bad = checker.check(q16, basis, retrieved=retrieved, target=Target(asked))
    assert bad.outcome == "unsupported" and bad.checks["value_match"].reason == "证据中没有：680"
    good = checker.check(q16.replace("680", "650"), basis, retrieved=retrieved, target=Target(asked))
    assert good.outcome == "pass", good.checks


# ---------------------------------------------------------------- 目标日期的复述（Fast 试跑中发现）

@pytest.mark.parametrize("claim, ev, targets, want", [
    ("2026年7月2日实际生效的一线城市住宿上限是650元/晚", ["一线城市｜标准：650元/晚"], (D("2026-07-02"),), "pass"),
    ("截至 2026 年 7 月 2 日，系统中记录的上限为 600元/晚", ["一线城市｜标准：600元/晚"], (D("2026-07-02"),), "pass"),
    ("2026年7月的上限是650元", ["上限650元"], (D("2026-07-02"),), "pass"),                     # 只到月，同一个月
    ("2026年7月2日的上限是650元，自2026年8月1日起执行", ["上限650元"], (D("2026-07-02"),), "unsupported"),  # 别的日期仍须有出处
    ("2026年7月2日的上限是650元", ["上限650元"], (), "unsupported"),                            # 没有目标日期时照常核对
    ("2026年7月2日的上限是680元", ["上限650元"], (D("2026-07-02"),), "unsupported"),            # 数值不受影响
])
def test_value_match_exempts_only_the_query_target_dates(claim, ev, targets, want):
    assert value_match(claim, ev, targets).outcome == want
