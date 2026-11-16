"""模型的思考过程（2026-11-16，设计变更记录；迁移 0014）。

- 线上客户端把 OpenAI 兼容响应里的 message.reasoning_content 交给编排器；没有时不报；
- 编排器按步骤收集（prepare、first、escalate、verify、assemble），随回答返回（提问接口、流式的 reasoning 事件）、
  写进 query_log.reasoning，会话接口原样返回；转入后台、换进程接着执行也不丢（在图的状态里，随检查点保存）；
- 思考文字不进 trace（13.3：原文不进 trace）。
"""
from __future__ import annotations

import json

from eaer.adapters.model import Message, collect_reasoning, report_reasoning
from eaer.adapters.online import Endpoint, OnlineModelClient
from test_async import Gated, aq, run_in_background, task_row, wait_for  # noqa: F401  （aq 是 fixture）
from test_engine import Scripted
from test_query_api import ANSWER, H, api  # noqa: F401  （api 是 fixture）


class Thinking(Scripted):
    """每次生成都报一段思考文字，标出是哪种调用。"""

    def generate(self, messages, schema):
        props = (schema or {}).get("properties", {})
        kind = "judge" if "verdict" in props else "time" if "as_of_valid" in props else "draft"
        report_reasoning(f"思考：{kind}")
        return super().generate(messages, schema)


def test_online_client_reports_reasoning_content():
    def transport(url, headers, body, timeout):
        assert body["thinking"] == {"type": "enabled"}
        return 200, {"choices": [{"message": {"content": "{\"x\": 1}", "reasoning_content": "先算一下。"}}],
                     "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    m = OnlineModelClient(Endpoint("http://gen", "k", "g", extra={"thinking": {"type": "enabled"}}), None, None,
                          transport=transport)
    with collect_reasoning() as got:
        c = m.generate([Message("user", "q")], {"type": "object"})
    assert got == ["先算一下。"] and c.reasoning == "先算一下。"
    with collect_reasoning() as got:                                  # 没有思考文字时不报
        m.transport = lambda *a: (200, {"choices": [{"message": {"content": "答"}}]})
        m.generate([Message("user", "q")], None)
    assert got == []


def test_reasoning_is_returned_logged_and_kept_in_session(api):  # noqa: F811
    client, conn, state, _, _ = api
    state["model"] = Thinking(ANSWER)
    r = client.post("/v1/query", headers=H("tok-staff"), json={"question": "一线城市住宿上限是多少"}).json()
    assert r["status"] == "completed" and r["reasoning"], r
    steps = {x["step"] for x in r["reasoning"]}
    assert "first" in steps and all(x["text"].startswith("思考：") for x in r["reasoning"])
    row = conn.execute("SELECT reasoning FROM query_log WHERE session_id = %s", (r["session_id"],)).fetchone()
    assert row["reasoning"] == r["reasoning"]
    s = client.get(f"/v1/sessions/{r['session_id']}", headers=H("tok-staff")).json()
    assert s["items"][0]["reasoning"] == r["reasoning"]


def test_stream_emits_reasoning_before_the_answer(api):  # noqa: F811
    client, _, state, _, _ = api
    state["model"] = Thinking(ANSWER)
    text = client.post("/v1/query", headers=H("tok-staff"), json={"question": "一线城市住宿上限是多少", "stream": True}).text
    events = [line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")]
    assert "reasoning" in events and events.index("reasoning") < events.index("delta") < events.index("done")
    done = json.loads(text.split("event: done\ndata: ")[1].split("\n")[0])
    assert done["reasoning"]


def test_no_reasoning_when_the_model_gives_none(api):  # noqa: F811
    client, _, _, _, _ = api
    r = client.post("/v1/query", headers=H("tok-staff"), json={"question": "一线城市住宿上限是多少"}).json()
    assert r["reasoning"] == []


class ThinkingGated(Gated):
    def generate(self, messages, schema):
        props = (schema or {}).get("properties", {})
        report_reasoning("思考：判断" if "verdict" in props else "思考：起草" if "as_of_valid" not in props else "思考：时间")
        return super().generate(messages, schema)


def test_reasoning_survives_the_handoff_to_the_background_worker(aq):  # noqa: F811
    """接口进程做完“检索与生成”“升级判断”，卡在“验证”时超时转后台；后台任务进程从检查点接着做完。
    两段执行里的思考都在：接口进程那段在图的状态里，随检查点交给后台进程。"""
    aq.st["model"] = ThinkingGated(ANSWER)
    r, release = run_in_background(aq)
    sid = r.json()["session_id"]
    running = aq.client.get(f"/v1/sessions/{sid}", headers=H("tok-staff")).json()["items"][0]
    assert running["status"] == "running" and running["reasoning"] == []          # 提问结束时写入
    release.set()
    wait_for(lambda: task_row(aq.conn)["status"] == "queued")
    assert aq.bg().run_pending() == 1
    item = aq.client.get(f"/v1/sessions/{sid}", headers=H("tok-staff")).json()["items"][0]
    assert item["status"] == "completed" and item["background"]
    by_step = {(x["step"], x["text"]) for x in item["reasoning"]}
    assert ("first", "思考：起草") in by_step and ("verify", "思考：判断") in by_step
