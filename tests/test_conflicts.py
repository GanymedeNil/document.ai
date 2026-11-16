"""冲突处理（开发文档 7.4、8.6）：不同文件说法不一致时两份都列出，系统不代为选择——由代码保证。离线：脚本化模型。

事实表样例 Q10 的失败方式：Fast 把冲突写成一条结论（“通知写 500，与办法的 450 不一致”）却只引用了通知，数值核对
不通过，整条被删、转入 Deep；Deep 在 conflicts 里列了冲突，回答却只写 450。这里覆盖：
- 提示：Fast 与 Deep（多跳）都要求每种说法各写一条结论、各自引用自己的证据，不替用户选择；
- 组装：模型列了跨文件冲突、回答只写了一方时，按另一方证据的原文补一条非核心结论（经确定性检查）并加提示；
- 不补的情况：两方都已写出；同一文件内的“冲突”；回答没涉及这组冲突；另一方在目标日期不生效（不在检索结果里）。
"""
from __future__ import annotations

from datetime import date

from eaer.orchestrator.deep import GEN_RULE
from eaer.orchestrator.fast import SYSTEM
from test_engine import SCOPE, Scripted, claim, eng, ids_with  # noqa: F401
from test_ingestion import env, html  # noqa: F401
from test_retrieval import ix, publish  # noqa: F401

NOTICE = html("关于差旅住宿标准的补充通知", "第一条　为明确住宿标准，特作如下补充。", "第二条　一线城市住宿上限为每晚700元。",
              "第三条　本通知自2026年1月1日起施行。")
Q = "一线城市住宿上限是多少"


def with_notice(eng):
    publish(eng, NOTICE, "notice.html", source_ref="NOTICE")
    eng.conn.execute("UPDATE document SET doc_type = 'normative' WHERE source_ref = 'NOTICE'")
    eng.worker.run_until_idle()
    return eng


def one_side(p):
    """只写 650 一方，但在 conflicts 里列出 650 与 700 两条证据（Deep 常见的写法）。"""
    return {"abstain": False, "claims": [claim("一线城市住宿上限为每晚650元。", p, "650元")],
            "conflicts": [ids_with(p, "650元")[:1] + ids_with(p, "700元")[:1]]}


def test_prompts_require_one_claim_per_side():
    assert "为每种说法各写一条结论，写明出自哪份文件" in SYSTEM and "不要替用户选择" in SYSTEM
    assert "每个数值都必须能在它引用的证据中找到" in SYSTEM
    assert "为每种说法各写一条结论" in GEN_RULE["multihop"] and "不代为选择" in GEN_RULE["multihop"]


def test_the_missing_side_is_added_from_the_evidence(eng):
    with_notice(eng)
    r = eng.make(Scripted(one_side)).ask(SCOPE, Q, today=date(2026, 6, 1))
    texts = [c.text for c in r.claims]
    assert "一线城市住宿上限为每晚650元。" in texts
    added = [c for c in r.claims if c.text.startswith("另有《关于差旅住宿标准的补充通知》")]
    assert len(added) == 1 and "一线城市住宿上限为每晚700元。" in added[0].text and not added[0].core
    assert added[0].status in ("supported", "unverified") and added[0].citations[0].version_no == 1
    assert "650元" in r.answer and "700元" in r.answer
    assert any("两种说法都已列出" in n for n in r.notices)
    assert next(c for c in r.claims if "650元" in c.text and not c.text.startswith("另有")).core


def test_nothing_is_added_when_both_sides_are_stated(eng):
    with_notice(eng)

    def both(p):
        return {"abstain": False, "claims": [claim("一线城市住宿上限为每晚650元。", p, "650元"),
                                             claim("补充通知规定一线城市住宿上限为每晚700元。", p, "700元", core=False)],
                "conflicts": [ids_with(p, "650元")[:1] + ids_with(p, "700元")[:1]]}
    r = eng.make(Scripted(both)).ask(SCOPE, Q, today=date(2026, 6, 1))
    assert not any(c.text.startswith("另有") for c in r.claims)
    assert any("两种说法都已列出" in n for n in r.notices)            # 模型自己写全了，照样提示用户


def test_same_document_or_untouched_conflicts_are_left_alone(eng):
    with_notice(eng)

    def same_doc(p):                                   # 同一文件的两条证据被标成冲突：不补
        return {"abstain": False, "claims": [claim("一线城市住宿上限为每晚650元。", p, "650元")],
                "conflicts": [ids_with(p, "650元")[:1] + ids_with(p, "80元")[:1]]}
    r = eng.make(Scripted(same_doc)).ask(SCOPE, Q, today=date(2026, 6, 1))
    assert not any(c.text.startswith("另有") for c in r.claims) and not any("两种说法" in n for n in r.notices)

    def untouched(p):                                  # 回答完全没涉及这组冲突：不由代码替它回答
        return {"abstain": False, "claims": [claim("出差伙食补贴每天80元。", p, "80元")],
                "conflicts": [ids_with(p, "650元")[:1] + ids_with(p, "700元")[:1]]}
    r = eng.make(Scripted(untouched)).ask(SCOPE, "出差伙食补贴每天多少", today=date(2026, 6, 1))
    assert not any(c.text.startswith("另有") for c in r.claims)


def test_a_side_not_in_force_on_the_target_date_is_not_added(eng):
    later = html("关于差旅住宿标准的补充通知", "第一条　为明确住宿标准，特作如下补充。", "第二条　一线城市住宿上限为每晚700元。",
                 "第三条　本通知自2026年7月1日起施行。")
    publish(eng, later, "notice.html", source_ref="NOTICE")
    eng.conn.execute("UPDATE document SET doc_type = 'normative' WHERE source_ref = 'NOTICE'")
    eng.worker.run_until_idle()
    r = eng.make(Scripted(one_side)).ask(SCOPE, Q, today=date(2026, 6, 1))       # 通知 7 月起施行，6 月 1 日不生效
    assert "650元" in r.answer and not r.abstained                                  # 主制度照常回答
    assert not any(c.text.startswith("另有") for c in r.claims) and "700元" not in r.answer
