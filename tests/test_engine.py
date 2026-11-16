"""查询编排：Router、升级、Verified、预算与能力降级（开发文档 8.1–8.6、9.3–9.4、12.1）。离线：脚本化模型。

- AC-8-1（硬性）命中强制验证清单的问题全部经过 Verified；未命中的不经过；
- AC-8-2（硬性）无法满足的问题：检索轮数不超上限（Fast 1 + 补检索 1），以拒答结束；
- AC-8-3 8.2 的每个升级信号各一个用例，mode_path 与 escalations 符合预期（Fast → Deep 开放时的路径见 test_deep.py）；
- AC-8-4 mode_hint 只升不降；
- AC-8-5（硬性）unsupported 结论不出现，unverified 有标注；
- AC-7-6 时间点回显在回答开头，接口参数优先于解析结果；
- AC-12-1 模型不支持结构化输出：Verified 关闭，Fast 用纯文本照常回答并说明；
- 定向补检索把引错证据的结论修好；query_log 写入完整。
"""
from __future__ import annotations

import re
from datetime import date

import pytest

from eaer.adapters.authorizer import LocalAuthorizer
from eaer.adapters.model import Capabilities, Completion, Usage
from eaer.orchestrator.engine import Engine, log_query
from eaer.orchestrator.forced_verify import add_rule
from eaer.retrieval.service import Retriever, Scope
from test_ingestion import env, html  # noqa: F401
from test_retrieval import ix, publish  # noqa: F401

SCOPE = Scope("t1", frozenset({"R_STAFF"}))
TODAY = date(2026, 6, 1)


def ids_with(prompt: str, needle: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"\[(ev_[0-9a-f]+)\][^\n]*\n([^\n]*)", prompt) if needle in m.group(2)]


DEEP_KINDS = {"field": "plan_compare", "first_hop": "plan_multihop", "phenomenon": "plan_causal",
              "closed": "hop", "queries": "replan"}


def deep_default(kind, prompt):
    return {"hop": {"closed": True}, "replan": {"queries": []}}.get(kind, {})


class Scripted:
    """一个模型兼任生成、语义检查、时间解析与 Deep 的规划，按请求的 schema 区分。
    deep(kind, prompt)：kind 为 plan_compare / plan_multihop / plan_causal / hop / replan。"""

    def __init__(self, draft, judge=None, structured=True, deep=None, context_tokens=32000):
        self.draft, self.structured, self.deep, self.context_tokens = draft, structured, deep or deep_default, context_tokens
        self.judge = judge or (lambda claim, prompt: {"verdict": "supported", "confidence": 0.9,
                                                      "supporting_ids": re.findall(r"\[(ev_[0-9a-f]+)\]", prompt)})
        self.calls, self.prompts = [], []

    def capabilities(self):
        return Capabilities(self.structured, self.context_tokens, False, 0)

    def generate(self, messages, schema):
        prompt = "\n".join(m.content for m in messages)
        self.prompts.append(prompt)
        props = (schema or {}).get("properties", {})
        kind = next((k for key, k in DEEP_KINDS.items() if key in props), None)
        if kind:
            self.calls.append(kind)
            return Completion("{}", self.deep(kind, prompt), Usage(30, 10), "scripted")
        if "verdict" in props:
            self.calls.append("judge")
            claim = re.search(r"结论：(.*)", prompt).group(1)
            return Completion("{}", self.judge(claim, prompt), Usage(50, 10), "scripted")
        if "standalone" in props:                # 多轮记忆的改写（memory.condense）：默认当作完整问题，不改写
            self.calls.append("condense")
            return Completion("{}", {"standalone": prompt.rsplit("当前问题：", 1)[-1], "depends_on_history": False},
                              Usage(20, 5), "scripted")
        if "as_of_valid" in props:
            self.calls.append("time")
            return Completion("{}", {"as_of_valid": ""}, Usage(10, 2), "scripted")
        self.calls.append("draft")
        out = self.draft(prompt)
        return Completion(out if isinstance(out, str) else "{}", None if isinstance(out, str) else out,
                          Usage(100, 20), "scripted")


def claim(text, prompt, needle, core=True):
    return {"text": text, "evidence_ids": ids_with(prompt, needle)[:1], "core": core}


NORM = html("差旅报销管理办法", "第一条　为规范差旅报销，制定本办法。", "第二条　一线城市住宿上限为每晚650元。",
            "第三条　出差伙食补贴每天80元。", "第四条　本办法自2026年1月1日起施行。")
RECORD = html("2026年第二季度差旅费用分析报告", "报告日期：2026年7月10日", "第二季度差旅总费用为128万元。")


@pytest.fixture
def eng(ix):
    publish(ix, NORM, "n.html", source_ref="N")
    publish(ix, RECORD, "r.html", source_ref="R")
    ix.conn.execute("UPDATE document SET doc_type = CASE WHEN source_ref = 'R' THEN 'record' ELSE 'normative' END")
    ix.conn.execute("UPDATE document_version SET doc_date = '2026-07-10' FROM document d "
                    "WHERE d.id = document_version.document_id AND d.source_ref = 'R'")
    ix.worker.run_until_idle()
    calls = {"n": 0}
    real = ix.retriever.search

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    ix.retriever.search = counting
    ix.search_calls = calls

    def make(model, retriever=None, **kw):
        return Engine(ix.conn, retriever or ix.retriever, model, LocalAuthorizer(ix.conn, {}), store=ix.store, **kw)
    ix.make = make
    return ix


ANSWER_650 = lambda p: {"abstain": False, "claims": [claim("一线城市住宿上限为每晚650元。", p, "650元")]}  # noqa: E731
ANSWER_128 = lambda p: {"abstain": False, "claims": [claim("第二季度差旅总费用为128万元。", p, "128万元")]}  # noqa: E731


# ---------- AC-8-1 ----------

def test_normative_citation_forces_verified(eng):
    r = eng.make(Scripted(ANSWER_650)).ask(SCOPE, "一线城市住宿上限是多少", today=date(2026, 11, 11))
    assert r.router["decided"] == "fast" and r.mode_path == ["fast", "verified"] and r.mode == "verified"
    assert "builtin-normative" in r.forced_verify and r.claims[0].semantic["status"] == "supported"
    assert not r.abstained and "650元" in r.answer


def test_time_question_starts_in_verified(eng):
    r = eng.make(Scripted(ANSWER_650)).ask(SCOPE, "2026 年 3 月 15 日一线城市住宿上限是多少", today=date(2026, 11, 11))
    assert r.router["decided"] == "verified" and r.mode_path == ["verified"]
    assert r.time_basis["as_of_valid"] == "2026-03-15"


def test_tenant_rule_forces_verified_and_record_question_without_rules_stays_fast(eng):
    q = "第二季度差旅总费用是多少"
    r = eng.make(Scripted(ANSWER_128)).ask(SCOPE, q, today=date(2026, 11, 11))
    assert r.mode_path == ["fast"] and r.forced_verify == [] and "128万元" in r.answer
    rid, _ = add_rule(eng.conn, "t1", "keyword", "差旅总费用", "admin", r and __import__("datetime").datetime.now())
    r = eng.make(Scripted(ANSWER_128)).ask(SCOPE, q, today=date(2026, 11, 11))
    assert r.mode_path == ["fast", "verified"] and rid["id"] in r.forced_verify


# ---------- AC-8-2 ----------

def test_unsatisfiable_question_stays_within_budget_and_refuses(eng):
    wrong = lambda p: {"abstain": False, "claims": [claim("一线城市住宿上限为每晚999元。", p, "650元")]}  # noqa: E731
    deny = lambda c, p: {"verdict": "unsupported", "confidence": 0.95, "supporting_ids": []}  # noqa: E731
    r = eng.make(Scripted(wrong, deny), deep_enabled=False).ask(SCOPE, "一线城市住宿上限是多少", today=date(2026, 11, 11))
    assert r.abstained and "999" not in r.answer
    assert r.retrieval_rounds == 2 and eng.search_calls["n"] == 2      # 首轮 1 + 补检索 1，不再多
    # Deep 开放时的同一问题见 test_deep.py::test_unsatisfiable_question_through_deep_stays_within_budget


# ---------- AC-8-3：8.2 的每个信号 ----------

def test_signal_conflict(eng):
    def draft(p):
        ids = ids_with(p, "650元") + ids_with(p, "80元")
        return {"abstain": False, "claims": [claim("一线城市住宿上限为每晚650元。", p, "650元")], "conflicts": [ids]}
    r = eng.make(Scripted(draft)).ask(SCOPE, "第二季度差旅总费用是多少", today=date(2026, 11, 11))
    assert r.mode_path == ["fast", "verified"] and any("冲突" in e for e in r.escalations)


def test_signal_superseded_version(eng):
    publish(eng, html("差旅报销管理办法", "第二条　一线城市住宿上限为每晚700元。", "第四条　本办法自2026年7月1日起施行。"),
            "n2.html", source_ref="N")
    eng.worker.run_until_idle()
    r = eng.make(Scripted(ANSWER_650), budgets={}).ask(SCOPE, "一线城市住宿上限是多少", today=date(2026, 11, 11),
                                                       as_of_valid=date(2026, 3, 1))
    assert "命中已被取代或已过期的版本" in r.signals and r.escalations == []
    assert r.mode_path == ["verified"]                              # 时间参数直接进入 Verified，信号照样记录
    r2 = eng.make(Scripted(ANSWER_650)).ask(SCOPE, "一线城市住宿上限是多少", today=date(2026, 3, 1))
    assert r2.mode_path == ["fast", "verified"] and any("已被取代" in e for e in r2.escalations)


def test_signal_chart_value(eng):
    lc = eng.lc
    from eaer.lifecycle.service import Lifecycle  # noqa: F401
    vid = eng.conn.execute("SELECT v.id FROM document_version v JOIN document d ON d.id = v.document_id "
                           "WHERE d.source_ref = 'R'").fetchone()["id"]
    lc.add_evidence(vid, "figure", {"section": [], "figure": 0}, "图1　第二季度差旅总费用：128万元")
    eng.conn.execute("INSERT INTO ingest_job (id, version_id, stage, status, created_at, updated_at) "
                     "VALUES (gen_random_uuid(), %s, 'index_update', 'queued', now(), now())", (vid,))
    eng.worker.run_until_idle()
    draft = lambda p: {"abstain": False, "claims": [  # noqa: E731
        {"text": "第二季度差旅总费用为128万元。", "evidence_ids": ids_with(p, "图1")[:1], "core": True}]}
    r = eng.make(Scripted(draft)).ask(SCOPE, "第二季度差旅总费用是多少", today=date(2026, 11, 11))
    assert r.mode_path == ["fast", "verified"] and any("图表" in e for e in r.escalations)


class LowReranker:
    def rerank(self, query, passages):
        return [0.05] * len(passages)


def test_signals_towards_deep_are_recorded_while_deep_is_not_available(eng):
    """Deep 关闭（准入未通过）时只记录信号；Deep 开放时的升级见 test_deep.py。"""
    retr = Retriever(eng.conn, eng.index, eng.retriever.embedder, LowReranker())
    missing = lambda p: {"abstain": False, "missing": "没有找到二线城市的标准",  # noqa: E731
                         "claims": [claim("第二季度差旅总费用为128万元。", p, "128万元")]}
    r = eng.make(Scripted(missing), retriever=retr, deep_enabled=False).ask(
        SCOPE, "第二季度差旅总费用和二线城市标准", today=date(2026, 11, 11))
    deep = [e for e in r.escalations if e.startswith("应转入 Deep")]
    assert deep and "重排后最高分低于阈值" in deep[0] and "部分要点没有证据覆盖" in deep[0] and "Deep 尚未开放" in deep[0]
    r = eng.make(Scripted(lambda p: {"abstain": False, "claims": [
        claim("第二季度差旅总费用为130万元。", p, "128万元")]}), deep_enabled=False).ask(
        SCOPE, "第二季度差旅总费用是多少", today=date(2026, 11, 11))
    assert any("引用检查失败" in e for e in r.escalations)


def test_comparison_question_is_routed_to_deep_and_falls_back(eng):
    r = eng.make(Scripted(ANSWER_650), deep_enabled=False).ask(SCOPE, "2026 年 7 月修订前后住宿上限有什么变化", today=date(2026, 11, 11))
    assert r.router["decided"] == "deep" and "Deep 尚未开放" in " ".join(r.notices)
    assert r.mode_path == ["fast", "verified"]                      # 版本比较命中强制验证，仍经过 Verified


# ---------- AC-8-4 ----------

def test_mode_hint_only_raises(eng):
    m = Scripted(ANSWER_128)
    r = eng.make(m).ask(SCOPE, "第二季度差旅总费用是多少", today=date(2026, 11, 11), mode_hint="verified")
    assert r.router["hinted"] and r.mode_path == ["verified"]
    r = eng.make(m).ask(SCOPE, "2026 年 3 月 15 日一线城市住宿上限是多少", today=date(2026, 11, 11), mode_hint="fast")
    assert not r.router["hinted"] and r.mode_path == ["verified"]


# ---------- AC-8-5 ----------

def test_semantic_unsupported_side_claim_removed_and_low_confidence_labelled(eng):
    def draft(p):
        return {"abstain": False, "claims": [claim("一线城市住宿上限为每晚650元。", p, "650元"),
                                             claim("出差伙食补贴每天80元。", p, "80元", core=False),
                                             claim("本办法自2026年1月1日起施行。", p, "2026年1月1日", core=False)]}

    def judge(c, p):
        if "伙食" in c:
            return {"verdict": "unsupported", "confidence": 0.9, "supporting_ids": []}
        if "施行" in c:
            return {"verdict": "supported", "confidence": 0.4, "supporting_ids": re.findall(r"\[(ev_[0-9a-f]+)\]", p)}
        return {"verdict": "supported", "confidence": 0.95, "supporting_ids": re.findall(r"\[(ev_[0-9a-f]+)\]", p)}

    r = eng.make(Scripted(draft, judge), budgets={"verified_repair_rounds": 0}).ask(
        SCOPE, "一线城市住宿上限是多少", today=date(2026, 11, 11))
    assert not r.abstained and "伙食" not in r.answer and "施行。（未验证）" in r.answer
    assert [c.status for c in r.claims] == ["supported", "unsupported", "unverified"]
    assert "伙食" not in str(r.response()["claims"])


def test_targeted_retrieval_repairs_a_claim_that_cited_the_wrong_evidence(eng):
    def draft(p):                     # 结论对，但引用了伙食补贴那条证据
        return {"abstain": False, "claims": [{"text": "一线城市住宿上限为每晚650元。",
                                              "evidence_ids": ids_with(p, "80元")[:1], "core": True}]}

    def judge(c, p):
        good = ids_with(p, "650元")
        return ({"verdict": "supported", "confidence": 0.95, "supporting_ids": good} if good
                else {"verdict": "unsupported", "confidence": 0.9, "supporting_ids": []})

    # Deep 关闭：引用检查失败本应转入 Deep，这里只看 Verified 的定向补检索
    r = eng.make(Scripted(draft, judge), deep_enabled=False).ask(SCOPE, "一线城市住宿上限是多少", today=date(2026, 11, 11))
    c = r.claims[0]
    assert c.repaired and c.status == "supported" and not r.abstained and r.retrieval_rounds == 2
    assert "650" in eng.conn.execute("SELECT content FROM evidence WHERE id = %s",
                                     (c.citations[0].evidence_id,)).fetchone()["content"]


# ---------- AC-7-6 ----------

def test_time_point_echoed_and_parameters_override(eng):
    m = Scripted(ANSWER_650)
    r = eng.make(m).ask(SCOPE, "2026 年 3 月 15 日一线城市住宿上限是多少", today=date(2026, 11, 11))
    assert r.answer.startswith("按 2026 年 3 月 15 日生效的版本回答。")
    r = eng.make(m).ask(SCOPE, "2026 年 3 月 15 日一线城市住宿上限是多少", today=date(2026, 11, 11),
                        as_of_valid=date(2026, 2, 1))
    assert r.time_basis["source"] == "parameter" and r.answer.startswith("按 2026 年 2 月 1 日生效的版本回答。")


def test_relative_time_goes_to_the_model_then_is_reported_unresolved(eng):
    m = Scripted(ANSWER_650)
    r = eng.make(m).ask(SCOPE, "当时一线城市住宿上限是多少", today=date(2026, 11, 11))
    assert "time" in m.calls and r.time_basis["unresolved"] == ["当时"] and any("无法换算" in n for n in r.notices)


# ---------- AC-12-1 ----------

def test_no_structured_output_disables_verified_and_fast_still_answers(eng):
    plain = Scripted(lambda p: f"一线城市住宿上限为每晚650元。[{ids_with(p, '650元')[0]}]", structured=False)
    r = eng.make(plain).ask(SCOPE, "一线城市住宿上限是多少", today=date(2026, 11, 11))
    assert r.mode_path == ["fast"] and not r.abstained and r.answer == "一线城市住宿上限为每晚650元。（未验证）"
    assert any("验证模式不可用" in n for n in r.notices) and "judge" not in plain.calls


# ---------- query_log ----------

def test_log_query_writes_all_required_fields(eng):
    r = eng.make(Scripted(ANSWER_650)).ask(SCOPE, "一线城市住宿上限是多少", today=date(2026, 11, 11))
    qid = log_query(eng.conn, r, scope=SCOPE, user_id="alice", question="一线城市住宿上限是多少")
    row = eng.conn.execute("SELECT * FROM query_log WHERE id = %s", (qid,)).fetchone()
    assert row["mode_path"] == ["fast", "verified"] and row["model_id"] == "scripted"
    assert row["authz_decision"]["principals"] == ["R_STAFF"] and row["evidence_ids"] == r.evidence_ids
    assert row["index_build_id"] == "MemoryIndex" and row["cost"]["retrieval_rounds"] == 1
    assert row["claims"][0]["semantic"]["status"] == "supported"
