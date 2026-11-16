"""严格模式与宽松模式（迁移 0015，设计变更记录 2026-11-16）。离线：脚本化模型、内存索引。

起因：“我在北京住一周最高报销多少”在只写了“一线城市每晚 650 元、伙食补贴每天 80 元”的办法上整题拒答——提示不许用
证据以外的知识（办法没写北京属于哪一类），也不许换算数值（7 晚 × 650），算出的合计在数值核对里也找不到。

- 宽松模式（默认）：按证据里的单价写出算式算合计，代码核对算术与单价的来历，结果标未验证、basis=calculation；
  前面结论算出的数后面可以接着用；算术不对的判不支持；按常识判断对象属于哪一类单列一条（不引证据，未验证，
  basis=common_knowledge，不作核心结论）；这两类结论不交给语义检查；答了一部分时写明还缺什么；
- 严格模式：同样的算式与常识推断判不支持（删除，核心结论不通过则拒答）；提示要求逐类给出规定；逐类回答照常通过；
- 租户默认与按次覆盖；回答、提问记录与会话里记下所用模式；两种模式的缓存不串用。
"""
from __future__ import annotations

from datetime import date

from eaer.orchestrator.fast import ClaimOut, Draft, Steps, claim_basis
from test_engine import SCOPE, Scripted, eng, ids_with  # noqa: F401
from test_ingestion import env  # noqa: F401
from test_query_api import H, api, ask  # noqa: F401
from test_retrieval import ix  # noqa: F401

TODAY = date(2026, 11, 11)
Q = "我在北京住一周最高报销多少"
ASSUME = {"text": "所引办法没有列出一线城市的范围，按常识，北京属于一线城市。", "evidence_ids": [], "core": False,
          "assumption": True}


def computed(p, total="4,550 + 560 = 5,110"):
    return {"text": f"按一线城市计、按 7 晚计：住宿 650 元/晚 × 7 晚 = 4,550 元，伙食补贴 80 元 × 7 天 = 560 元，"
                    f"合计 {total} 元。",
            "evidence_ids": ids_with(p, "650元")[:1] + ids_with(p, "80元")[:1], "core": True}


LENIENT_DRAFT = lambda p: {"abstain": False, "claims": [ASSUME, computed(p)]}  # noqa: E731


def gen(model) -> str:
    """最近一次生成回答用的提示（Verified 之后最后一条是语义检查的提示）。"""
    return next(x for x in reversed(model.prompts) if x.startswith("你是企业制度问答助手"))


def by_basis(r):
    return {claim_basis(c.checks): c for c in r.claims}


def test_lenient_is_the_default_and_answers_with_a_checked_formula(eng):
    model = Scripted(LENIENT_DRAFT)
    r = eng.make(model).ask(SCOPE, Q, today=TODAY)
    assert "可以按常识判断它属于哪一类" in gen(model) and "写出完整算式" in gen(model)
    assert r.strict is False and not r.abstained and "verified" in r.mode_path
    calc, guess = by_basis(r)["calculation"], by_basis(r)["common_knowledge"]
    assert calc.status == "unverified" and calc.core and "650 × 7 = 4550" in calc.checks["value_match"]["reason"]
    assert guess.status == "unverified" and not guess.core and guess.citations == []
    assert calc.semantic is None and guess.semantic is None              # 不交给语义检查：原文本来就没写合计
    assert "（常识推断，文件未写明）" in r.answer and "5,110 元。（未验证）" in r.answer
    body = r.response()
    assert body["strict"] is False and {c["basis"] for c in body["claims"]} == {"calculation", "common_knowledge"}


def test_wrong_arithmetic_is_removed(eng):
    r = eng.make(Scripted(lambda p: {"abstain": False, "claims": [ASSUME, computed(p, "4,550 + 560 = 5,200")]})).ask(
        SCOPE, Q, today=TODAY)
    assert r.abstained and "核心结论未能在证据中得到核实" in r.answer


def test_strict_mode_rejects_formulas_and_assumptions(eng):
    model = Scripted(LENIENT_DRAFT)                  # 模型不按严格模式的要求写
    r = eng.make(model).ask(SCOPE, Q, today=TODAY, strict=True)
    assert "不要替它归类" in gen(model) and "不要自己计算" in gen(model) and "按常识" not in gen(model)
    assert r.strict is True and r.abstained and "核心结论未能在证据中得到核实" in r.answer


def test_strict_mode_answers_each_category(eng):
    def per_category(p):
        ev = ids_with(p, "650元")[:1]
        return {"abstain": False, "claims": [
            {"text": "所引办法按一线城市规定住宿上限，没有写明北京属于哪一类城市。", "evidence_ids": ev, "core": False},
            {"text": "一线城市住宿上限为每晚650元。", "evidence_ids": ev, "core": True},
            {"text": "出差伙食补贴每天80元。", "evidence_ids": ids_with(p, "80元")[:1], "core": True}]}
    r = eng.make(Scripted(per_category)).ask(SCOPE, Q, today=TODAY, strict=True)
    assert not r.abstained and all(c.status == "supported" for c in r.claims)
    assert "650元" in r.answer and "80元" in r.answer and "（未验证）" not in r.answer


def test_tenant_default_and_per_question_override(eng):
    model = Scripted(LENIENT_DRAFT)
    strict_tenant = eng.make(model, strict_default=True)
    assert strict_tenant.ask(SCOPE, Q, today=TODAY).strict is True and "不要替它归类" in gen(model)
    assert strict_tenant.ask(SCOPE, Q, today=TODAY, strict=False).strict is False
    assert "可以按常识判断" in gen(model)
    st = strict_tenant.prepare(Q, today=TODAY)
    assert st["request"]["strict"] is True               # 记在状态里：转入后台、换进程接着执行也不变


def test_formula_results_carry_over_to_later_claims(eng):
    """合计分在另一条结论里写：前面结论算出的 4,550 与 560 可以接着用；严格模式下都判不支持。"""
    from eaer.adapters.authorizer import LocalAuthorizer

    hits = eng.retriever.search(SCOPE, "住宿上限 伙食补贴", today=TODAY, as_of_valid=None, as_of_system=None)
    from eaer.orchestrator.fast import target_of
    steps = Steps(eng.conn, eng.retriever, None, LocalAuthorizer(eng.conn, {}), store=eng.store)
    meta = steps.meta(hits, None)
    ev = {h.content: h.evidence_id for h in hits}
    e650 = next(i for c, i in ev.items() if "650元" in c)
    e80 = next(i for c, i in ev.items() if "80元" in c)
    claims = [{"text": "住宿 650 × 7 = 4,550 元。", "evidence_ids": [e650], "core": False},
              {"text": "伙食补贴 80 × 7 = 560 元。", "evidence_ids": [e80], "core": False},
              {"text": "合计 4,550 + 560 = 5,110 元。", "evidence_ids": [e650, e80], "core": True}]
    lenient = steps.check(SCOPE, claims, hits, meta, target_of(TODAY, None))
    assert [c.status for c in lenient] == ["unverified"] * 3
    assert [claim_basis(c.checks) for c in lenient] == ["calculation"] * 3
    strict = steps.check(SCOPE, claims, hits, meta, target_of(TODAY, None), strict=True)
    assert [c.status for c in strict] == ["unsupported"] * 3


def test_partial_answer_says_what_is_missing():
    claims = [ClaimOut("一线城市住宿上限为每晚650元。", "supported", True, [], {})]
    text, abstained = Steps.assemble(claims, Draft([], False, "没有找到交通费的标准。", [], "m", None))
    assert not abstained and text == "一线城市住宿上限为每晚650元。另外，没有找到：没有找到交通费的标准。"
    guess = [ClaimOut("按常识，北京属于一线城市。", "unverified", False, [], {"common_knowledge": {}})]
    assert Steps.assemble(guess, Draft([], False, None, [], "m", None))[1]   # 只剩常识推断时拒答


def test_api_records_the_mode_and_keeps_caches_apart(api):
    client, conn, state, *_ = api
    state["model"] = Scripted(lambda p: {"abstain": False, "claims": [
        {"text": "一线城市住宿上限为每晚650元。", "evidence_ids": ids_with(p, "650元")[:1], "core": True}]})
    assert client.get("/v1/config/answer-policy", headers=H("tok-staff")).json() == {"strict": False}
    a = ask(client).json()
    b = ask(client, strict=True).json()
    assert a["strict"] is False and b["strict"] is True
    assert a["claims"][0]["basis"] == "evidence"
    assert len(state["model"].prompts) >= 2 and "不要替它归类" in gen(state["model"])   # 没有命中宽松模式的缓存
    rows = conn.execute("SELECT strict FROM query_log ORDER BY asked_at, strict").fetchall()
    assert sorted(r["strict"] for r in rows) == [False, True]
    s = client.get(f"/v1/sessions/{b['session_id']}", headers=H("tok-staff")).json()
    assert s["items"][0]["strict"] is True and s["items"][0]["claims"][0]["basis"] == "evidence"
    conn.execute("INSERT INTO tenant_config (tenant_id, updated_at) VALUES ('t1', now()) ON CONFLICT DO NOTHING")
    conn.execute("UPDATE tenant_config SET strict_mode = true WHERE tenant_id = 't1'")
    assert ask(client).json()["strict"] is True                                   # 不传时取租户默认
