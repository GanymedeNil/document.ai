"""多轮记忆（2026-11-16，设计变更记录；eaer/orchestrator/memory.py，迁移 0016）。

- 新会话的第一个问题不改写；接着已有会话提问时，追问结合之前的提问改写成完整的问题，检索与生成都用改写后的，
  回答与会话记录给出 standalone_question，说明里写“结合上文，问题理解为……”，query_log.question 保留原话；
- 给模型的上文只有之前提问的原话与回答引用的、当前仍有权可搜索的文档标题：不含之前回答的正文，
  归档后的文档标题不再带入；失败的提问与别人的提问不算上文；
- 回答缓存按改写后的问题区分：同一句追问在不同会话里是不同的问题；
- 强制验证的关键词规则按原话与改写后的问题一起判断，改写不能让问题绕开规则；
- 模型不能结构化输出、或改写输出不合格式时按原问题回答并说明。
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

from eaer.adapters.model import Capabilities, Completion, Usage
from eaer.adapters.online import StructuredOutputError
from eaer.orchestrator import forced_verify as fv
from eaer.orchestrator.engine import Engine
from eaer.orchestrator.memory import Turn, condense
from test_engine import Scripted
from test_query_api import ANSWER, NOW, Boom, H, api  # noqa: F401  （api 是 fixture）

Q1 = "一线城市住宿上限是多少"


class Remembering(Scripted):
    """改写由 rewrite(当前问题, 提示) -> (改写结果, 是否依赖上文) 决定；记下每次改写的提示。"""

    def __init__(self, rewrite, *a, **kw):
        super().__init__(ANSWER, *a, **kw)
        self.rewrite, self.condense_prompts = rewrite, []

    def generate(self, messages, schema):
        if "standalone" in (schema or {}).get("properties", {}):
            prompt = messages[-1].content
            self.condense_prompts.append(prompt)
            self.calls.append("condense")
            s, dep = self.rewrite(prompt.rsplit("当前问题：", 1)[1], prompt)
            return Completion("{}", {"standalone": s, "depends_on_history": dep}, Usage(20, 5), "scripted")
        return super().generate(messages, schema)


def ask(client, question, sid=None, token="tok-staff"):
    body = {"question": question} if sid is None else {"question": question, "session_id": str(sid)}
    return client.post("/v1/query", headers=H(token), json=body)


def follow_up(q, prompt):
    return ("二线城市住宿上限是多少", True) if q.startswith("那") else (q, False)


def test_first_question_is_not_rewritten(api):  # noqa: F811
    client, _, state, _, _ = api
    state["model"] = m = Remembering(follow_up)
    r = ask(client, Q1).json()
    assert r["standalone_question"] is None and "condense" not in m.calls
    assert not any(n.startswith("结合上文") for n in r["notices"])


def test_follow_up_is_rewritten_and_recorded(api):  # noqa: F811
    client, conn, state, _, _ = api
    state["model"] = m = Remembering(follow_up)
    sid = uuid.uuid4()
    first = ask(client, Q1, sid).json()
    assert first["standalone_question"] is None and "condense" not in m.calls       # 会话还没有已完成的提问
    r = ask(client, "那二线城市呢？", sid).json()
    assert r["status"] == "completed" and r["standalone_question"] == "二线城市住宿上限是多少"
    assert r["notices"][0] == "结合上文，问题理解为：二线城市住宿上限是多少"
    prompt = m.condense_prompts[-1]
    assert f"1. 问：{Q1}" in prompt and "《差旅报销管理办法》" in prompt and prompt.endswith("当前问题：那二线城市呢？")
    assert "650" not in prompt                                     # 不给之前回答的正文
    assert "二线城市住宿上限是多少" in m.prompts[-1] or any("二线城市住宿上限是多少" in p for p in m.prompts[-3:])
    row = conn.execute("SELECT question, standalone_question FROM query_log WHERE session_id = %s ORDER BY asked_at DESC",
                       (sid,)).fetchone()
    assert row["question"] == "那二线城市呢？" and row["standalone_question"] == "二线城市住宿上限是多少"
    items = client.get(f"/v1/sessions/{sid}", headers=H("tok-staff")).json()["items"]
    assert [i["standalone_question"] for i in items] == [None, "二线城市住宿上限是多少"]


def test_complete_question_in_a_session_is_kept(api):  # noqa: F811
    client, _, state, _, _ = api
    state["model"] = m = Remembering(follow_up)
    sid = uuid.uuid4()
    ask(client, Q1, sid)
    r = ask(client, "出差伙食补贴是多少", sid).json()
    assert "condense" in m.calls and r["standalone_question"] is None


def test_archived_titles_and_failed_turns_are_not_history(api):  # noqa: F811
    client, _, state, _, doc = api
    sid = uuid.uuid4()
    state["model"] = Remembering(follow_up)
    ask(client, Q1, sid)
    state["model"] = Boom()
    assert ask(client, "会失败的问题", sid).status_code == 502                       # 失败的提问也记进了会话
    assert client.post(f"/v1/documents/{doc}/archive", headers=H("tok-ed")).status_code == 200
    state["model"] = m = Remembering(follow_up)
    ask(client, "那二线城市呢？", sid)
    prompt = m.condense_prompts[-1]
    assert f"1. 问：{Q1}" in prompt and "会失败的问题" not in prompt and "2. 问" not in prompt
    assert "《" not in prompt                                     # 文档已归档：标题不再带入


def test_cache_is_keyed_on_the_rewritten_question(api):  # noqa: F811
    client, _, state, _, _ = api
    rewrite = {"一线": "二线城市住宿上限是多少", "伙食": "二线城市出差伙食补贴是多少"}

    def by_topic(q, prompt):
        if not q.startswith("那"):
            return q, False
        topic = "伙食" if "伙食补贴" in prompt else "一线"
        return rewrite[topic], True

    state["model"] = m = Remembering(by_topic)
    a, b = uuid.uuid4(), uuid.uuid4()
    ask(client, Q1, a)
    ask(client, "出差伙食补贴是多少", b)
    ra = ask(client, "那二线城市呢？", a).json()
    drafts = m.calls.count("draft")
    rb = ask(client, "那二线城市呢？", b).json()                    # 同一句追问、不同上文：不能命中 a 的缓存
    assert ra["standalone_question"] != rb["standalone_question"] and m.calls.count("draft") == drafts + 1
    drafts = m.calls.count("draft")
    rc = ask(client, "二线城市住宿上限是多少").json()                # 新会话里直接问改写后的问题：命中 a 写的缓存
    assert m.calls.count("draft") == drafts and rc["answer"] == ra["answer"]


def test_forced_verify_keyword_still_matches_the_original_wording(api):  # noqa: F811
    client, conn, state, _, _ = api
    rule, _ = fv.add_rule(conn, "t1", "keyword", "特急", "admin", NOW, None)
    state["model"] = Remembering(lambda q, p: ("二线城市住宿上限是多少", True) if "特急" in q else (q, False))
    sid = uuid.uuid4()
    ask(client, Q1, sid)
    r = ask(client, "特急：那二线城市呢？", sid).json()
    assert r["standalone_question"] == "二线城市住宿上限是多少"
    cost = conn.execute("SELECT cost FROM query_log WHERE session_id = %s ORDER BY asked_at DESC", (sid,)).fetchone()["cost"]
    assert rule["id"] in cost["forced_verify"] and r["mode_path"][-1] == "verified"


def test_engine_answers_the_original_question_when_rewriting_is_not_possible():
    turns = [Turn(Q1, ())]

    class NoJson:
        def capabilities(self):
            return Capabilities(False, 32000, False, 0)

    class BadJson:
        def capabilities(self):
            return Capabilities(True, 32000, False, 0)

        def generate(self, messages, schema):
            raise StructuredOutputError("要求结构化输出，返回的不是 JSON")

    standalone, note = Engine._condense(SimpleNamespace(model=NoJson()), "那二线城市呢？", turns)
    assert standalone is None and "不支持结构化输出" in note
    standalone, note = Engine._condense(SimpleNamespace(model=BadJson()), "那二线城市呢？", turns)
    assert standalone is None and "按原问题回答" in note


def test_condense_ignores_rewrites_that_change_nothing():
    class Same:
        def generate(self, messages, schema):
            return Completion("{}", {"standalone": " 那二线城市呢？ ", "depends_on_history": True}, Usage(1, 1), "m")

    assert condense(Same(), "那二线城市呢？", [Turn(Q1, ())]) is None
