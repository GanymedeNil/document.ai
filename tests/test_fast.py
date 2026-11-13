"""Fast 模式（开发文档 8.3、8.6；AC-8-5 的 Fast 部分）。离线运行，用脚本化的模型替代线上生成服务。

覆盖：引用检查后的组装（supported 附引用、unverified 标“未验证”、unsupported 删除、核心结论不通过则拒答）；
最终回答由保留的结论拼成，unsupported 的内容不出现；引用本次检索以外的证据视为 unsupported；
没有检索结果时不调用模型直接拒答；强制验证判断；授权范围不进入模型上下文。
"""
from __future__ import annotations

import re
from datetime import date

import pytest

from eaer.adapters.authorizer import LocalAuthorizer
from eaer.adapters.model import Capabilities, Completion, Usage
from eaer.orchestrator.fast import ABSTAIN, FastPipeline
from eaer.retrieval.service import Scope
from test_ingestion import env, html  # noqa: F401
from test_retrieval import ix, publish  # noqa: F401

SCOPE = Scope("t1", frozenset({"R_STAFF"}))
TODAY = date(2026, 6, 1)


class Scripted:
    """按提示里出现的证据 id 生成结论：decide(ids, prompt) -> 结构化输出。"""

    def __init__(self, decide):
        self.decide, self.prompts = decide, []

    def capabilities(self):
        return Capabilities(True, 32000, False, 0)

    def generate(self, messages, schema):
        prompt = "\n".join(m.content for m in messages)
        self.prompts.append(prompt)
        ids = re.findall(r"\[(ev_[0-9a-f]+)\]", prompt)
        return Completion("{}", self.decide(ids, prompt), Usage(100, 20), "scripted")


def pipeline(ix, decide):
    model = Scripted(decide)
    return FastPipeline(ix.conn, ix.retriever, model, LocalAuthorizer(ix.conn, {}), store=ix.store), model


def evid(prompt: str, needle: str) -> str:
    """找出内容含 needle 的那条证据的 id（证据块：首行“[id] 版本信息”，次行是原文）。"""
    for m in re.finditer(r"\[(ev_[0-9a-f]+)\][^\n]*\n([^\n]*)", prompt):
        if needle in m.group(2):
            return m.group(1)
    raise AssertionError(f"提示中没有含“{needle}”的证据")


DOC = html("差旅报销管理办法", "第一条　为规范差旅报销，制定本办法。", "第二条　一线城市住宿上限为每晚650元。",
           "第三条　出差伙食补贴每天80元。", "第四条　本办法自2026年1月1日起施行。")


@pytest.fixture
def fast_env(ix):
    publish(ix, DOC, "p.html", source_ref="P1")
    ix.conn.execute("UPDATE document SET doc_type = 'normative'")
    ix.worker.run_until_idle()                      # 审核通过后的索引任务
    return ix


def test_supported_claim_with_citation(fast_env):
    p, model = pipeline(fast_env, lambda ids, pr: {"abstain": False, "claims": [
        {"text": "一线城市住宿上限为每晚650元。", "evidence_ids": [evid(pr, "650元")], "core": True}]})
    a = p.answer(SCOPE, "一线城市住宿上限是多少", today=TODAY)
    assert not a.abstained and a.answer == "一线城市住宿上限为每晚650元。"
    c = a.claims[0]
    assert c.status == "supported" and c.citations[0].version_no == 1 and "para" in c.citations[0].locator
    assert c.citations[0].valid_grade == "explicit" and a.mode_path == ["fast"]


def test_unsupported_core_claim_turns_the_answer_into_a_refusal(fast_env):
    p, _ = pipeline(fast_env, lambda ids, pr: {"abstain": False, "claims": [
        {"text": "一线城市住宿上限为每晚680元。", "evidence_ids": [evid(pr, "650元")], "core": True}]})
    a = p.answer(SCOPE, "一线城市住宿上限是多少", today=TODAY)
    assert a.abstained and "680" not in a.answer and a.answer.startswith(ABSTAIN)
    assert a.claims[0].status == "unsupported" and "680" in a.claims[0].checks["value_match"]["reason"]


def test_unsupported_side_claim_is_removed_from_the_answer(fast_env):
    p, _ = pipeline(fast_env, lambda ids, pr: {"abstain": False, "claims": [
        {"text": "一线城市住宿上限为每晚650元。", "evidence_ids": [evid(pr, "650元")], "core": True},
        {"text": "二线城市住宿上限为每晚500元。", "evidence_ids": [evid(pr, "650元")], "core": False}]})
    a = p.answer(SCOPE, "一线城市住宿上限是多少", today=TODAY)
    assert not a.abstained and a.answer == "一线城市住宿上限为每晚650元。" and "500" not in a.answer
    assert [c.status for c in a.claims] == ["supported", "unsupported"]


def test_citing_evidence_outside_the_retrieval_is_unsupported(fast_env):
    p, _ = pipeline(fast_env, lambda ids, pr: {"abstain": False, "claims": [
        {"text": "一线城市住宿上限为每晚650元。", "evidence_ids": ["ev_" + "0" * 16], "core": True}]})
    a = p.answer(SCOPE, "一线城市住宿上限是多少", today=TODAY)
    assert a.abstained and "检索结果以外" in a.claims[0].checks["locator_exists"]["reason"]


def test_unverified_claim_is_kept_and_labelled(ix):
    publish(ix, html("会议费管理规定", "第一条　会议费每人每天上限为500元。"), "m.html", source_ref="M1")   # 无生效日：inferred
    ix.conn.execute("UPDATE document SET doc_type = 'normative'")
    ix.worker.run_until_idle()
    p, _ = pipeline(ix, lambda ids, pr: {"abstain": False, "claims": [
        {"text": "会议费每人每天上限为500元。", "evidence_ids": [evid(pr, "500元")], "core": True}]})
    a = p.answer(SCOPE, "会议费上限是多少", today=TODAY)
    assert not a.abstained and a.answer == "会议费每人每天上限为500元。（未验证）"
    assert a.claims[0].status == "unverified" and a.claims[0].checks["time_valid"]["outcome"] == "unverified"


def test_model_abstains_with_reason(fast_env):
    p, _ = pipeline(fast_env, lambda ids, pr: {"abstain": True, "claims": [], "missing": "没有加班餐补的规定"})
    a = p.answer(SCOPE, "加班餐补是多少", today=TODAY)
    assert a.abstained and a.answer == ABSTAIN + "缺少：没有加班餐补的规定"


def test_no_retrieval_results_means_refusal_without_calling_the_model(fast_env):
    p, model = pipeline(fast_env, lambda ids, pr: pytest.fail("不应调用模型"))
    a = p.answer(Scope("t1", frozenset({"R_NOBODY"})), "一线城市住宿上限是多少", today=TODAY)
    assert a.abstained and a.answer == ABSTAIN and model.prompts == []


def test_forced_verify_is_reported_after_retrieval(fast_env):
    p, _ = pipeline(fast_env, lambda ids, pr: {"abstain": False, "claims": [
        {"text": "一线城市住宿上限为每晚650元。", "evidence_ids": [evid(pr, "650元")], "core": True}]})
    a = p.answer(SCOPE, "一线城市住宿上限是多少", today=TODAY)
    assert a.forced_verify == ["builtin-normative"]                       # 引用了规范类文档
    b = p.answer(SCOPE, "一线城市住宿上限是多少", today=TODAY, as_of_valid=date(2026, 3, 15))
    assert "builtin-time-version" in b.forced_verify and b.notices == ["按 2026 年 3 月 15 日生效的版本回答"]


def test_scope_does_not_enter_the_model_context(fast_env):
    p, model = pipeline(fast_env, lambda ids, pr: {"abstain": True, "claims": []})
    p.answer(SCOPE, "一线城市住宿上限是多少", today=TODAY)
    assert model.prompts and "R_STAFF" not in model.prompts[0] and "t1" not in model.prompts[0].split("问题")[0]
