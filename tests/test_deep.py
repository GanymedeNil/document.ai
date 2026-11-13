"""Deep 模式（开发文档 8.3 的三个计划模板、8.2 的 Fast → Deep、8.5 的预算、10.3 的动作限制）。离线：脚本化模型。

- 比较模板：版本比较（按修订时间、按版本号）在各自版本的有效期内分别检索，旧版本的结论按旧版本的日期核对；
  时间点比较；模型拆出比较对象的一般比较；找不到要比较的版本时拒答；
- 多跳模板：用上一跳的结果构造下一跳，问题闭合即停；
- 因果模板：检索现象 → 检索变更说明与现象所在文档的上一版本 → 关联生成；
- AC-8-2（硬性，Deep 部分）：缺口一直补不上时，检索轮数不超过 Deep 的上限（含转入前 Fast 已用的），以拒答结束；
- AC-8-3（Deep 部分）：8.2 中 Fast → Deep 的三个信号各转入一次，mode_path 为 fast → deep（→ verified）；
- 12.1：模型没有结构化输出或上下文不够长时 Deep 关闭，由 Fast 回答并说明；
- 10.3：Deep 的动作只有检索与取版本，授权范围不进入模型上下文；取版本同样只列有权查看的可搜索版本；
- 计划、分组与动作写入 query_log（8.4 的 plan、gaps、history）。
"""
from __future__ import annotations

import itertools
import re
from datetime import date

import pytest

from eaer.adapters.authorizer import LocalAuthorizer
from eaer.orchestrator.deep import VersionSpec, pick_versions, topic, version_spec
from eaer.orchestrator.engine import log_query
from eaer.orchestrator.timeparse import parse
from eaer.retrieval.index import MissingScope
from eaer.retrieval.service import Retriever, Scope, VersionInfo
from eaer.verifier.deterministic import DeterministicChecker, Target
from test_engine import LowReranker, Scripted, claim, ids_with
from test_ingestion import env, html  # noqa: F401
from test_retrieval import ix, publish  # noqa: F401

SCOPE = Scope("t1", frozenset({"R_STAFF"}))
TODAY = date(2026, 11, 11)

N1 = html("差旅报销管理办法", "第一条　为规范差旅报销，制定本办法。", "第二条　一线城市住宿上限为每晚650元。",
          "第三条　出差伙食补贴每天80元。", "第四条　二线城市住宿标准按《差旅住宿补充通知》执行。",
          "第五条　本办法自2026年1月1日起施行。")
N2 = html("差旅报销管理办法", "第一条　为规范差旅报销，制定本办法。", "第二条　一线城市住宿上限为每晚700元。",
          "第三条　出差伙食补贴每天80元。", "第四条　二线城市住宿标准按《差旅住宿补充通知》执行。",
          "第五条　本办法自2026年7月1日起施行。")
NOTICE = html("关于调整一线城市住宿标准的说明", "因一线城市酒店价格上涨，一线城市住宿标准提高至每晚700元。",
              "本说明自2026年7月1日起施行。")
SUPPLEMENT = html("差旅住宿补充通知", "二线城市住宿上限为每晚450元。", "本通知自2026年1月1日起施行。")


@pytest.fixture
def deep(ix):
    publish(ix, N1, "n1.html", source_ref="N")
    publish(ix, N2, "n2.html", source_ref="N")
    publish(ix, NOTICE, "s.html", source_ref="S")
    publish(ix, SUPPLEMENT, "b.html", source_ref="B")
    ix.conn.execute("UPDATE document SET doc_type = 'normative'")
    ix.worker.run_until_idle()
    calls = {"n": 0}
    real = ix.retriever.search

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    ix.retriever.search = counting
    ix.search_calls = calls
    ix.version_of = lambda eid: str(ix.conn.execute(
        "SELECT v.version_no FROM evidence e JOIN document_version v ON v.id = e.version_id WHERE e.id = %s",
        (eid,)).fetchone()["version_no"])

    def make(model, retriever=None, **kw):
        from eaer.orchestrator.engine import Engine
        return Engine(ix.conn, retriever or ix.retriever, model, LocalAuthorizer(ix.conn, {}), store=ix.store, **kw)
    ix.make = make
    return ix


def plan(**slots):
    def deep(kind, prompt):
        if kind.startswith("plan_"):
            return slots
        return {"hop": {"closed": True}, "replan": {"queries": []}}[kind]
    return deep


def compare_draft(p):
    old, new = ids_with(p, "650元")[:1], ids_with(p, "700元")[:1]
    return {"abstain": False, "claims": [
        {"text": "修订前（第 1 版）一线城市住宿上限为每晚650元。", "evidence_ids": old, "core": False},
        {"text": "修订后（第 2 版）一线城市住宿上限为每晚700元。", "evidence_ids": new, "core": False},
        {"text": "一线城市住宿上限由每晚650元调整为每晚700元。", "evidence_ids": old + new, "core": True}]}


def assert_safe(model, r):
    """10.3：动作只有检索与取版本；授权范围（主体标识）不出现在任何发给模型的内容里。"""
    assert {a["kind"] for a in r.plan["actions"]} <= {"search", "versions"}
    assert not any("R_STAFF" in p for p in model.prompts)


# ---------- 比较：版本比较 ----------

def test_version_comparison_by_revision_point(deep):
    m = Scripted(compare_draft, deep=plan(field="一线城市住宿上限", objects=[]))
    r = deep.make(m).ask(SCOPE, "2026 年 7 月修订前后，一线城市住宿上限有什么变化？", today=TODAY)
    assert r.router["decided"] == "deep" and r.mode_path == ["deep", "verified"] and r.plan["template"] == "compare"
    before, after = r.plan["groups"]
    assert before["label"].startswith("修订前（第 1 版）") and after["label"].startswith("修订后（第 2 版）")
    assert before["on"] == "2026-06-30" and after["on"] == "2026-07-01"
    assert {deep.version_of(i) for i in before["evidence_ids"]} == {"1"}         # 每组只含本版本的证据
    assert {deep.version_of(i) for i in after["evidence_ids"]} == {"2"}
    assert not r.abstained and "650元" in r.answer and "700元" in r.answer
    assert [c.status for c in r.claims] == ["supported"] * 3
    assert r.claims[0].checks["time_valid"]["outcome"] == "pass"                # 旧版本按 6 月 30 日核对
    assert r.retrieval_rounds == 2                                              # 定位文档 1 + 修订前版本 1
    assert "versions" in [a["kind"] for a in r.plan["actions"]]
    assert_safe(m, r)


def test_version_comparison_by_version_numbers(deep):
    m = Scripted(compare_draft, deep=plan(field="一线城市住宿上限", objects=[]))
    r = deep.make(m).ask(SCOPE, "第 1 版和第 2 版的一线城市住宿上限有什么区别", today=TODAY)
    labels = [g["label"] for g in r.plan["groups"]]
    assert labels[0].startswith("第 1 版") and labels[1].startswith("第 2 版") and not r.abstained


def test_version_comparison_without_such_versions_refuses_without_generating(deep):
    m = Scripted(compare_draft, deep=plan(field="一线城市住宿上限", objects=[]))
    r = deep.make(m).ask(SCOPE, "第 1 版和第 5 版的一线城市住宿上限有什么区别", today=TODAY)
    assert r.abstained and "没有找到可以比较的第 1、5 版" in r.answer and "draft" not in m.calls
    assert r.retrieval_rounds == 1


# ---------- 比较：时间点比较与一般比较 ----------

def test_time_point_comparison_replans_then_refuses_within_budget(deep):
    n = itertools.count(1)

    def d(kind, prompt):
        if kind == "replan":
            label = re.search(r"- (.*?)：已用检索词", prompt).group(1)
            return {"queries": [{"label": label, "query": f"住宿标准{next(n)}"}]}
        return {"field": "一线城市住宿上限", "objects": []} if kind.startswith("plan_") else {"closed": True}

    m = Scripted(compare_draft, deep=d)
    r = deep.make(m).ask(SCOPE, "2020 年 1 月 1 日和现在的一线城市住宿上限有什么不同", today=TODAY)
    assert r.router["decided"] == "deep" and r.plan["template"] == "compare"
    assert [g["on"] for g in r.plan["groups"]] == ["2020-01-01", "2026-11-11"]
    assert r.retrieval_rounds == 3 and m.calls.count("replan") == 2                # 1 轮分别检索 + 2 轮再规划
    assert deep.search_calls["n"] == 4 and r.abstained and "2020 年 1 月 1 日" in r.answer
    assert "draft" not in m.calls and r.plan["gaps"]


def test_general_comparison_uses_objects_from_the_model(deep):
    objects = [{"label": "住宿上限", "query": "一线城市住宿上限"}, {"label": "伙食补贴", "query": "出差伙食补贴"}]

    def draft(p):
        return {"abstain": False, "claims": [claim("一线城市住宿上限为每晚700元。", p, "700元", core=True),
                                             claim("出差伙食补贴每天80元。", p, "80元", core=False)]}

    m = Scripted(draft, deep=plan(field="标准", objects=objects))
    r = deep.make(m).ask(SCOPE, "一线城市住宿上限和出差伙食补贴分别是多少", today=TODAY)
    assert r.plan["template"] == "compare" and [g["label"] for g in r.plan["groups"]] == ["住宿上限", "伙食补贴"]
    assert r.retrieval_rounds == 1 and deep.search_calls["n"] == 2              # 一轮内对每个对象各检索一次
    assert not r.abstained and "700元" in r.answer and "80元" in r.answer


# ---------- 多跳 ----------

def test_multihop_builds_the_next_hop_from_the_first(deep):
    hops = iter([{"closed": False, "next_query": "差旅住宿补充通知 二线城市住宿上限", "missing": "补充通知的标准"},
                 {"closed": True}])

    def d(kind, prompt):
        return {"first_hop": "二线城市住宿标准"} if kind == "plan_multihop" else next(hops)

    def draft(p):
        return {"abstain": False, "claims": [claim("二线城市住宿标准按《差旅住宿补充通知》执行。", p, "补充通知》执行", core=False),
                                             claim("二线城市住宿上限为每晚450元。", p, "450元", core=True)]}

    m = Scripted(draft, deep=d)
    r = deep.make(m).ask(SCOPE, "二线城市住宿按哪份文件执行，标准是多少", today=TODAY, mode_hint="deep")
    assert r.router["hinted"] and r.mode_path[0] == "deep" and r.plan["template"] == "multihop"
    assert [g["label"] for g in r.plan["groups"]] == ["第 1 跳", "第 2 跳"] and r.retrieval_rounds == 2
    assert m.calls.count("hop") == 2 and not r.abstained and "450元" in r.answer
    assert_safe(m, r)


def never_closes():
    n = itertools.count(1)
    return lambda kind, prompt: ({"first_hop": "二线城市住宿标准"} if kind == "plan_multihop" else
                                 {"closed": False, "next_query": f"补充规定{next(n)}", "missing": "没有找到所需的标准"})


CANNOT = lambda p: {"abstain": True, "missing": "证据中没有三线城市的标准", "claims": []}  # noqa: E731


def test_unsatisfiable_multihop_stays_within_the_deep_budget(deep):
    """预算用尽仍未闭合：照常生成（8.3），仍缺的要点交给生成步骤；核心答不了即拒答（8.2）。"""
    m = Scripted(CANNOT, deep=never_closes())
    r = deep.make(m).ask(SCOPE, "三线城市住宿标准是多少", today=TODAY, mode_hint="deep")
    assert r.retrieval_rounds == 3 and deep.search_calls["n"] == 3 and r.abstained
    assert "证据中没有三线城市的标准" in r.answer and m.calls.count("draft") == 1
    assert "以下要点仍没有找到证据：没有找到所需的标准" in m.prompts[-1] and r.plan["unresolved"]
    deep.search_calls["n"] = 0
    r = deep.make(Scripted(CANNOT, deep=never_closes()), budgets={"deep_rounds": 2}).ask(
        SCOPE, "三线城市住宿标准是多少", today=TODAY, mode_hint="deep")
    assert r.retrieval_rounds == 2 and deep.search_calls["n"] == 2 and r.abstained


def test_unresolved_points_are_reported_when_the_core_is_answered(deep):
    m = Scripted(lambda p: {"abstain": False, "claims": [claim("一线城市住宿上限为每晚700元。", p, "上限为每晚700元")]},
                 deep=never_closes())
    r = deep.make(m).ask(SCOPE, "一线城市住宿上限和三线城市标准", today=TODAY, mode_hint="deep")
    assert not r.abstained and "700元" in r.answer and r.retrieval_rounds == 3
    assert any(n.startswith("以下要点没有找到证据") and "没有找到所需的标准" in n for n in r.notices)


# ---------- 因果 ----------

def test_causal_template_retrieves_change_notes_and_the_previous_version(deep):
    def draft(p):
        return {"abstain": False, "claims": [claim("一线城市住宿上限现为每晚700元。", p, "上限为每晚700元", core=False),
                                             claim("调整的原因是一线城市酒店价格上涨。", p, "酒店价格上涨", core=True)]}

    m = Scripted(draft, deep=plan(phenomenon="一线城市住宿上限", change_query="一线城市住宿标准 调整 说明"))
    r = deep.make(m).ask(SCOPE, "为什么一线城市住宿上限调整了", today=TODAY)
    assert r.router["decided"] == "deep" and r.plan["template"] == "causal" and r.retrieval_rounds == 2
    labels = [g["label"] for g in r.plan["groups"]]
    assert labels[:2] == ["现象", "相关变更记录与说明"] and labels[2].startswith("变更前（《差旅报销管理办法》第 1 版）")
    assert {deep.version_of(i) for i in r.plan["groups"][2]["evidence_ids"]} == {"1"}
    assert not r.abstained and "酒店价格上涨" in r.answer and r.mode_path == ["deep", "verified"]
    assert_safe(m, r)


# ---------- AC-8-3：Fast → Deep 的三个信号 ----------

def test_low_rerank_score_escalates_fast_to_deep(deep):
    retr = Retriever(deep.conn, deep.index, deep.retriever.embedder, LowReranker())
    m = Scripted(lambda p: {"abstain": False, "claims": [claim("一线城市住宿上限为每晚700元。", p, "上限为每晚700元")]})
    r = deep.make(m, retriever=retr).ask(SCOPE, "一线城市住宿上限是多少", today=TODAY)
    assert r.mode_path == ["fast", "deep", "verified"] and r.escalations[0] == "转入 Deep：重排后最高分低于阈值"
    assert r.plan["template"] == "multihop" and r.plan["reason"] == "由 Fast 升级"
    assert r.plan["groups"][0]["label"] == "第 1 跳（首轮检索）" and r.retrieval_rounds == 1   # Fast 的结果即第一跳
    assert m.calls.count("draft") == 2 and not r.abstained


def test_failed_citation_and_missing_points_escalate_to_deep(deep):
    calls = iter([lambda p: {"abstain": False, "claims": [claim("一线城市住宿上限为每晚999元。", p, "上限为每晚700元")]},
                  lambda p: {"abstain": False, "claims": [claim("一线城市住宿上限为每晚700元。", p, "上限为每晚700元")]}])
    r = deep.make(Scripted(lambda p: next(calls)(p))).ask(SCOPE, "一线城市住宿上限是多少", today=TODAY)
    assert r.mode_path == ["fast", "deep", "verified"] and "引用检查失败" in r.escalations[0]
    assert not r.abstained and "700元" in r.answer and "999" not in r.answer

    missing = iter([{"abstain": False, "missing": "没有找到二线城市的标准",
                     "claims": [{"text": "一线城市住宿上限为每晚700元。", "evidence_ids": [], "core": True}]}])

    def draft(p):
        first = next(missing, None)
        if first is not None:
            first["claims"][0]["evidence_ids"] = ids_with(p, "上限为每晚700元")[:1]
            return first
        return {"abstain": False, "claims": [claim("一线城市住宿上限为每晚700元。", p, "上限为每晚700元"),
                                             claim("二线城市住宿上限为每晚450元。", p, "450元", core=False)]}

    hops = iter([{"closed": False, "next_query": "二线城市住宿上限", "missing": "二线城市的标准"}, {"closed": True}])
    m = Scripted(draft, deep=lambda kind, p: next(hops))
    r = deep.make(m).ask(SCOPE, "一线和二线城市住宿上限各是多少", today=TODAY)
    assert r.mode_path[:2] == ["fast", "deep"] and "问题中有部分要点没有证据覆盖" in r.escalations[0]
    assert r.retrieval_rounds == 2 and "450元" in r.answer and "上一步说明还缺：没有找到二线城市的标准" in m.prompts[1]


def test_unsatisfiable_question_through_deep_stays_within_budget(deep):
    """AC-8-2：Fast 1 轮 + Deep 补到上限 3 轮（含 Fast 的 1 轮）+ Verified 补检索 1 轮，以拒答结束。"""
    wrong = lambda p: {"abstain": False, "claims": [claim("一线城市住宿上限为每晚999元。", p, "上限为每晚700元")]}  # noqa: E731
    m = Scripted(wrong, deep=never_closes())
    r = deep.make(m).ask(SCOPE, "一线城市住宿上限是多少", today=TODAY)
    assert r.mode_path == ["fast", "deep", "verified"] and r.abstained and "999" not in r.answer
    assert r.retrieval_rounds == 4 and deep.search_calls["n"] == 4


# ---------- 12.1：Deep 的能力要求 ----------

def test_deep_closed_without_structured_output_or_enough_context(deep):
    q = "2026 年 7 月修订前后，一线城市住宿上限有什么变化？"
    r = deep.make(Scripted(compare_draft, context_tokens=8000)).ask(SCOPE, q, today=TODAY)
    assert r.mode_path[0] == "fast" and r.plan is None and any("上下文长度 8000 不足" in n for n in r.notices)
    plain = Scripted(lambda p: f"一线城市住宿上限为每晚700元。[{ids_with(p, '上限为每晚700元')[0]}]", structured=False)
    r = deep.make(plain).ask(SCOPE, q, today=TODAY)
    assert r.mode_path == ["fast"] and any("不支持结构化输出，Deep 不可用" in n for n in r.notices)


# ---------- 取版本的权限（10.1、10.3） ----------

def test_versions_lists_only_searchable_versions_the_user_may_see(deep):
    doc = deep.conn.execute("SELECT id FROM document WHERE source_ref = 'N'").fetchone()["id"]
    assert [v.version_no for v in deep.retriever.versions(SCOPE, str(doc))] == [1, 2]
    other = deep.lc.create_acl_policy("t1", ["R_FIN"])
    n3 = N2.replace("700".encode(), "800".encode()).replace("2026年7月1日".encode(), "2026年10月1日".encode())
    r = deep.ing.receive("t1", n3, "n3.html", acl_policy_id=other, source_ref="N")
    deep.worker.run_until_idle()
    task = str(deep.conn.execute("SELECT id FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()["id"])
    pending = deep.ing.receive("t1", N2.replace("700".encode(), "900".encode()), "n4.html", acl_policy_id=deep.acl,
                               source_ref="N")
    assert pending and [v.version_no for v in deep.retriever.versions(SCOPE, str(doc))] == [1, 2]   # 待审核的不列
    assert deep.review.decide("t1", [{"task_id": task, "decision": "approve"}], "alice")[0]["ok"]
    assert [v.version_no for v in deep.retriever.versions(SCOPE, str(doc))] == [1, 2]               # 无权的不列
    assert [v.version_no for v in deep.retriever.versions(Scope("t1", frozenset({"R_FIN"})), str(doc))] == [3]
    assert deep.retriever.versions(Scope("t2", frozenset({"R_STAFF"})), str(doc)) == []
    with pytest.raises(MissingScope):
        deep.retriever.versions(Scope("t1", frozenset()), str(doc))
    deep.lc.archive_document(doc, "alice")
    assert deep.retriever.versions(SCOPE, str(doc)) == []                                           # 归档后不列


# ---------- 按版本的目标时间核对 ----------

def test_old_version_claims_are_checked_against_the_old_versions_dates(deep):
    hits = deep.retriever.search(SCOPE, "一线城市住宿上限", today=TODAY, as_of_valid=date(2026, 6, 30))
    old = next(h for h in hits if "650元" in h.content)
    checker = DeterministicChecker(deep.conn, LocalAuthorizer(deep.conn, {}), SCOPE, deep.store)
    claim_text = "修订前一线城市住宿上限为每晚650元。"
    r = checker.check(claim_text, [old.evidence_id], retrieved={old.evidence_id}, target=Target(TODAY))
    assert r.outcome == "unsupported" and "作废" in r.checks["time_valid"].reason
    r = checker.check(claim_text, [old.evidence_id], retrieved={old.evidence_id}, target=Target(TODAY),
                      version_targets={old.version_id: Target(date(2026, 6, 30))})
    assert r.outcome == "pass"


# ---------- query_log ----------

def test_plan_and_actions_are_logged(deep):
    r = deep.make(Scripted(compare_draft, deep=plan(field="一线城市住宿上限", objects=[]))).ask(
        SCOPE, "2026 年 7 月修订前后，一线城市住宿上限有什么变化？", today=TODAY)
    qid = log_query(deep.conn, r, scope=SCOPE, user_id="alice", question="q")
    cost = deep.conn.execute("SELECT cost, mode_path FROM query_log WHERE id = %s", (qid,)).fetchone()
    assert cost["mode_path"] == ["deep", "verified"] and cost["cost"]["plan"]["template"] == "compare"
    assert cost["cost"]["plan"]["actions"][0]["kind"] == "search" and cost["cost"]["retrieval_rounds"] == 2


# ---------- 纯函数 ----------

def V(no, vf=None, vt=None, approved=None):
    from datetime import datetime, timezone
    return VersionInfo(f"v{no}", "d", "办法", no, vf, vt, approved and datetime(*approved, tzinfo=timezone.utc))


def test_version_spec_and_pick_versions():
    t = lambda q: version_spec(q, parse(q), TODAY)  # noqa: E731
    assert t("第二版和第三版有什么区别") == VersionSpec("numbers", numbers=(2, 3))
    assert t("2026 年 7 月修订前后有什么变化") == VersionSpec("revision", point=(2026, 7, None))
    assert t("2025 年 6 月 15 日和现在有什么不同") == VersionSpec("dates", dates=(date(2025, 6, 15), TODAY))
    assert t("新旧版有什么区别") == VersionSpec("latest") and t("一线城市和二线城市有什么区别") is None
    vs = [V(1, date(2025, 1, 1), date(2026, 3, 1)), V(2, date(2026, 3, 1), date(2026, 7, 1)),
          V(3, None, None, (2026, 7, 5))]
    labels = lambda p: [(lbl[:3], v.version_no, d) for lbl, v, d in p]  # noqa: E731
    assert labels(pick_versions(vs, VersionSpec("revision", point=(2026, 3, None)), TODAY)) == [
        ("修订前", 1, date(2026, 2, 28)), ("修订后", 2, date(2026, 3, 1))]
    # 生效日未知时按审核通过的月份找修订后的版本
    assert labels(pick_versions(vs, VersionSpec("revision", point=(2026, 7, None)), TODAY)) == [
        ("修订前", 2, date(2026, 6, 30)), ("修订后", 3, TODAY)]
    assert labels(pick_versions(vs, VersionSpec("latest"), date(2026, 5, 1)))[1][1] == 2
    assert pick_versions(vs, VersionSpec("numbers", numbers=(1, 4)), TODAY) is None
    assert pick_versions(vs[:1], VersionSpec("latest"), TODAY) is None
    assert topic("2026 年 7 月修订前后，一线城市住宿上限有什么变化？") == "一线城市住宿上限"


def test_from_to_wording_is_a_version_comparison():
    """“最近一次修订把 X 从多少改成了多少”：Router 进 Deep，Deep 选比较模板、按版本比较（合成集 v2 试跑中发现的漏判）。"""
    from eaer.orchestrator.deep import choose_template
    from eaer.orchestrator.router import decide

    q = "《员工休假管理办法》最近一次修订把陪产假天数从多少改成了多少？"
    assert decide(q, parse(q)).mode == "deep" and choose_template(q)[0] == "compare"
    assert version_spec(q, parse(q), TODAY) == VersionSpec("latest")
