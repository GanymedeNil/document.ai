"""线上模型通用客户端（开发文档 11.3、12.2）。离线运行：用脚本化的假传输层与本地 HTTP 服务注入错误和超时。

AC-12-5：注入超时和错误时按策略重试（可重试错误重试 2 次、带退避），最终失败返回明确错误，不返回空答案。
另覆盖：结构化输出的 JSON 模式与本地校验、Embedding 分批与维度、重排解析、计量记录、错误信息不含密钥、
未配置的能力抛 CapabilityNotSupported、Fast 缺能力时能指出缺什么。
"""
from __future__ import annotations

import http.server
import json
import threading
import time

import pytest

from eaer.adapters.model import CapabilityNotSupported, Message
from eaer.adapters.online import (Endpoint, ModelCallError, OnlineModelClient, StructuredOutputError,
                                  urllib_transport, validate)

KEY = "sk-test-secret-123"
GEN = Endpoint("https://gen.example", KEY, "gen-model", timeout=30, retries=2, extra={"thinking": {"type": "disabled"}})
EMB = Endpoint("https://emb.example", KEY, "emb-model")
RR = Endpoint("https://rr.example", KEY, "rr-model")


def chat(content, prompt=12, completion=3):
    return 200, {"model": "gen-model", "choices": [{"message": {"content": content}}],
                 "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


class Script:
    """按顺序返回预设结果；元素为 (状态码, 响应) 或要抛出的异常。"""

    def __init__(self, *steps):
        self.steps, self.calls = list(steps), []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, body, timeout))
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def client(script, **kw):
    sleeps, records = [], []
    c = OnlineModelClient(GEN, EMB, RR, embedding_dims=3, transport=script, sleep=sleeps.append,
                          meter=records.append, **kw)
    return c, sleeps, records


MSG = [Message("user", "一加一等于几？")]


def test_generate_sends_openai_compatible_request_and_records_usage():
    s = Script(chat("2"))
    c, _, rec = client(s)
    out = c.generate(MSG, None)
    url, headers, body, timeout = s.calls[0]
    assert url == "https://gen.example/chat/completions" and headers["Authorization"] == f"Bearer {KEY}"
    assert body["model"] == "gen-model" and body["thinking"] == {"type": "disabled"} and timeout == 30
    assert out.text == "2" and out.parsed is None and (out.usage.input_tokens, out.usage.output_tokens) == (12, 3)
    assert rec[0].kind == "generate" and rec[0].attempts == 1 and rec[0].ok


def test_images_are_sent_as_data_uri_blocks_only_to_vision_models():
    """12.1 图像输入：有图片的消息发成文字块加 image_url 块（base64 data URI）；没声明 vision 的模型直接拒绝，
    不把图片静默丢掉；没有图片的消息仍是字符串。"""
    import base64

    from eaer.adapters.model import CapabilityNotSupported, Image
    png = b"\x89PNG\r\n\x1a\nfake"
    msgs = [Message("system", "读图"), Message("user", "图中数值是多少？", (Image("image/png", png),))]
    s = Script(chat('{"value": "128"}'))
    c, _, _ = client(s, vision=True)
    c.generate(msgs, {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]})
    body = s.calls[0][2]
    assert isinstance(body["messages"][1]["content"], str)                      # 系统提示（含 schema）不变
    user = body["messages"][-1]["content"]
    assert user[0] == {"type": "text", "text": "图中数值是多少？"}
    assert user[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}
    plain, _, _ = client(Script(), vision=False)
    with pytest.raises(CapabilityNotSupported, match="EAER_GEN_VISION"):
        plain.generate(msgs, None)
    assert plain.capabilities().vision is False and c.capabilities().vision is True


@pytest.mark.parametrize("failures", [
    [(429, {"error": {"message": "rate limited"}})],
    [(503, None), TimeoutError("timed out")],
    [ConnectionError("reset"), (502, None)],
])
def test_retryable_errors_are_retried_with_backoff(failures):
    s = Script(*failures, chat("2"))
    c, sleeps, rec = client(s)
    assert c.generate(MSG, None).text == "2"
    assert len(s.calls) == len(failures) + 1 and sleeps == [0.5, 1.0][:len(failures)]
    assert rec[-1].attempts == len(failures) + 1 and rec[-1].ok


def test_final_failure_is_an_explicit_error_after_three_attempts():
    s = Script((500, None), TimeoutError("t"), (503, {"error": {"message": "overloaded"}}))
    c, sleeps, rec = client(s)
    with pytest.raises(ModelCallError, match="尝试 3 次.*HTTP 503.*overloaded"):
        c.generate(MSG, None)
    assert sleeps == [0.5, 1.0] and rec[-1].ok is False and rec[-1].attempts == 3


def test_non_retryable_error_fails_immediately_without_leaking_the_key():
    s = Script((401, {"error": {"message": f"invalid key {KEY}"}}))
    c, sleeps, _ = client(s)
    with pytest.raises(ModelCallError) as e:
        c.generate(MSG, None)
    assert len(s.calls) == 1 and sleeps == [] and KEY not in str(e.value) and "***" in str(e.value)


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_answer_is_an_error_not_an_empty_answer(content):
    c, _, _ = client(Script(chat(content)))
    with pytest.raises(ModelCallError, match="空内容"):
        c.generate(MSG, None)


SCHEMA = {"type": "object", "required": ["answer", "claims"], "properties": {
    "answer": {"type": "string"},
    "claims": {"type": "array", "items": {"type": "object", "required": ["text", "evidence_ids"], "properties": {
        "text": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}}}}}}}


def test_structured_output_uses_json_mode_and_validates_locally():
    good = {"answer": "650 元", "claims": [{"text": "上限 650 元", "evidence_ids": ["ev_1"]}]}
    s = Script(chat(json.dumps(good, ensure_ascii=False)))
    c, _, _ = client(s)
    out = c.generate(MSG, SCHEMA)
    body = s.calls[0][2]
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["role"] == "system" and '"evidence_ids"' in body["messages"][0]["content"]
    assert out.parsed == good


@pytest.mark.parametrize("content, problem", [
    ("这不是 JSON", "不是 JSON"),
    ('{"answer": "x"}', "缺少字段 claims"),
    ('{"answer": 1, "claims": []}', "$.answer 应为 string"),
    ('{"answer": "x", "claims": [{"text": "t", "evidence_ids": "ev_1"}]}', "evidence_ids 应为 array"),
])
def test_invalid_structured_output_is_rejected(content, problem):
    c, _, _ = client(Script(chat(content)))
    with pytest.raises(StructuredOutputError, match=problem.replace("$", r"\$").replace("[", r"\[")):
        c.generate(MSG, SCHEMA)


def test_truncated_structured_output_says_so():
    status, body = chat('{"answer": "x", "claims": [{"text": "t", "evid')
    body["choices"][0]["finish_reason"] = "length"
    c, _, _ = client(Script((status, body)))
    with pytest.raises(StructuredOutputError, match="超过长度上限被截断"):
        c.generate(MSG, SCHEMA)


def test_embed_batches_orders_by_index_and_checks_dims():
    def emb(n, offset=0):
        return 200, {"data": [{"index": i, "embedding": [float(offset + i), 0.0, 1.0]} for i in reversed(range(n))],
                     "usage": {"total_tokens": n}}
    s = Script(emb(64), emb(6, 64))
    c, _, rec = client(s)
    vecs = c.embed([f"t{i}" for i in range(70)])
    assert len(vecs) == 70 and [v[0] for v in vecs] == [float(i) for i in range(70)]
    assert [len(call[2]["input"]) for call in s.calls] == [64, 6] and s.calls[0][0] == "https://emb.example/embeddings"
    assert c.embed([]) == [] and rec[0].input_tokens == 64
    bad, _, _ = client(Script((200, {"data": [{"index": 0, "embedding": [1.0, 2.0]}]})))
    with pytest.raises(ModelCallError, match="维度"):
        bad.embed(["x"])


def test_rerank_returns_scores_in_passage_order():
    s = Script((200, {"results": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1},
                                  {"index": 1, "relevance_score": 0.5}]}))
    c, _, _ = client(s)
    assert c.rerank("住宿上限", ["a", "b", "c"]) == [0.1, 0.5, 0.9]
    assert s.calls[0][0] == "https://rr.example/rerank" and s.calls[0][2]["documents"] == ["a", "b", "c"]
    assert c.rerank("x", []) == []


def test_unconfigured_capabilities_and_fast_requirements():
    only_gen = OnlineModelClient(GEN, transport=Script())
    caps = only_gen.capabilities()
    assert caps.structured_output and caps.embedding_dims == 0
    assert only_gen.missing_for_fast() == ["Embedding", "重排"]
    with pytest.raises(CapabilityNotSupported):
        only_gen.embed(["x"])
    with pytest.raises(CapabilityNotSupported):
        only_gen.rerank("q", ["p"])
    with pytest.raises(CapabilityNotSupported):
        OnlineModelClient(None).generate(MSG, None)
    with pytest.raises(ValueError, match="维度"):
        OnlineModelClient(GEN, EMB)


def test_from_env():
    env = {"EAER_GEN_BASE_URL": "https://api.deepseek.com/", "EAER_GEN_API_KEY": KEY, "EAER_GEN_MODEL": "deepseek-flash",
           "EAER_GEN_THINKING": "disabled", "EAER_GEN_CONTEXT_TOKENS": "1048576", "EAER_GEN_VISION": "1",
           "EAER_MODEL_TIMEOUT": "20", "EAER_GEN_TEMPERATURE": "0"}
    c = OnlineModelClient.from_env(env)
    assert c.gen.base_url == "https://api.deepseek.com" and c.gen.timeout == 20
    assert c.gen.extra == {"thinking": {"type": "disabled"}, "temperature": 0.0}
    assert c.capabilities().context_tokens == 1048576 and c.capabilities().vision and c.emb is None
    with pytest.raises(ValueError, match="EAER_EMBED_MODEL"):
        OnlineModelClient.from_env({**env, "EAER_EMBED_BASE_URL": "https://x", "EAER_EMBED_API_KEY": "k"})


def test_validate_subset():
    s = {"type": "object", "properties": {"n": {"type": "integer"}, "k": {"enum": ["a", "b"]}}}
    assert validate({"n": 1, "k": "a"}, s) is None
    assert validate({"n": True}, s) == "$.n 应为 integer"
    assert validate({"k": "c"}, s).startswith("$.k 取值应在")


# ---------- 真实 HTTP：默认传输层的超时与错误 ----------

class _Handler(http.server.BaseHTTPRequestHandler):
    plan: list = []

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        action = _Handler.plan.pop(0)
        if action == "slow":
            time.sleep(1.0)
            return
        code, payload = action
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_real_http_timeout_and_server_error_then_success(server):
    _Handler.plan = ["slow", (503, {"error": {"message": "busy"}}), chat("2")]
    sleeps = []
    c = OnlineModelClient(Endpoint(server, KEY, "m", timeout=0.3, retries=2), transport=urllib_transport,
                          sleep=sleeps.append)
    assert c.generate(MSG, None).text == "2" and sleeps == [0.5, 1.0]


def test_real_http_all_attempts_time_out(server):
    _Handler.plan = ["slow", "slow", "slow"]
    c = OnlineModelClient(Endpoint(server, KEY, "m", timeout=0.2, retries=2), transport=urllib_transport,
                          sleep=lambda s: None)
    with pytest.raises(ModelCallError, match="超时"):
        c.generate(MSG, None)
