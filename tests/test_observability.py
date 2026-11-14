"""AC-13-5 可观测（开发文档 13.3、10.2）：每次查询一条 trace，每个节点都有记录；Trace 原文访问受权限控制。

开发文档把它列为人工检查（抽查查询的 trace），这里把能自动化的部分做成用例，人工抽查时看的就是同样的内容
（EAER_TRACE_EXPORTER=console 打印，或 otlp 导出到 Phoenix）：
- 经接口提问（接口进程内用 TestClient，模型服务用 OpenAI 兼容的替身，走线上客户端的代码）：根 span eaer.query，
  prepare 与 first、escalate、verify、assemble 每个节点各一个 span，都在同一条 trace 里，记了耗时、检索候选数、
  模型调用次数与 token 数；检索 span 记各阶段候选数；模型调用 span 记模型名与 token 数；
- 转入后台或换执行者时，从查询状态里的 traceparent 续上同一条 trace；
- 后台任务（解析、写入索引）各有 span；
- 原文不进 trace：所有 span 的属性都是 ASCII 标识符、数字或布尔值，问题、证据、回答的原文都不出现；
  原文只能经有权限控制的接口取得（会话接口只给提问人，见 tests/test_query_api.py）。
"""
from __future__ import annotations

from datetime import datetime, timezone

import psycopg
import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from psycopg.rows import dict_row

from eaer.adapters.authorizer import LocalAuthorizer
from eaer.adapters.online import OnlineModelClient
from eaer.api.app import TokenAuthenticator, create_app
from eaer.domain.identity import Identity
from eaer.ingestion.indexer import Indexer
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.observability import REDACTED, configure, safe, span, traceparent
from eaer.orchestrator.engine import NODES, Engine
from eaer.retrieval.index import MemoryIndex
from eaer.retrieval.service import Retriever
from eaer.storage import LocalFileStore
from stubmodel import StubModelServer

TOKENS = {"tok-ed": Identity("ed", "t1", frozenset({"editor"})), "tok-rv": Identity("rv", "t1", frozenset({"reviewer"})),
          "tok-staff": Identity("staff", "t1", frozenset()), "tok-other": Identity("other", "t1", frozenset())}
ACCOUNTS = {"staff": {"tenant": "t1", "principals": ["R_STAFF"]}, "other": {"tenant": "t1", "principals": ["R_STAFF"]}}
NOW = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)
QUESTION = "一线城市住宿上限是多少"
BODY = ("<h1>差旅报销管理办法</h1><p>第一条　为规范差旅报销，制定本办法。</p><p>第二条　一线城市住宿上限为每晚650元。</p>"
        "<p>第三条　本办法自2026年1月1日起施行。</p>")


class Recorder(SpanExporter):
    """只在用例运行期间记录（全局 TracerProvider 一个进程只能设一次，其余用例的 span 不记）。"""

    def __init__(self):
        self.spans, self.on = [], False

    def export(self, spans):
        if self.on:
            self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


@pytest.fixture(scope="session")
def recorder():
    rec = Recorder()
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(SimpleSpanProcessor(rec))
    else:
        assert configure({}, "eaer-test", exporter=rec)
    return rec


@pytest.fixture
def spans(recorder):
    recorder.spans.clear()
    recorder.on = True
    yield recorder.spans
    recorder.on = False


@pytest.fixture
def api(db_url, tmp_path):
    with StubModelServer() as stub:
        model = OnlineModelClient.from_env(stub.env())
        embedder, store, index = model.embedder(), LocalFileStore(tmp_path / "files"), MemoryIndex()
        conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)

        def engine(c, tenant):
            return Engine(c, Retriever(c, index, embedder, model), model, LocalAuthorizer(c, ACCOUNTS), store=store)

        app = create_app(db_url, store, TokenAuthenticator(TOKENS), lambda: NOW,
                         authorizer=lambda c: LocalAuthorizer(c, ACCOUNTS), engine=engine)
        client = TestClient(app)
        worker = Worker(Ingestion(conn, store, lambda: NOW), Indexer(conn, index, embedder, "b1"))
        yield client, conn, worker, engine
        conn.close()


def H(t):
    return {"Authorization": f"Bearer {t}"}


def publish(client, worker):
    client.post("/v1/documents", headers=H("tok-ed"), data={"principals": "R_STAFF"},
                files={"file": ("p.html", BODY.encode(), "text/html")})
    worker.run_until_idle()
    task = client.get("/v1/review/tasks", headers=H("tok-rv")).json()["items"][0]["task_id"]
    assert client.post("/v1/review/decisions", headers=H("tok-rv"), json={"decisions": [
        {"task_id": task, "decision": "approve", "corrections": {"doc_type": "normative"}}]}).json()["results"][0]["ok"]
    worker.run_until_idle()


def by_name(spans, name):
    return [s for s in spans if s.name == name]


def assert_no_text(spans, *texts):
    for s in spans:
        for k, v in s.attributes.items():
            for x in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(x, str):
                    assert x.isascii(), (s.name, k, x)
                    assert all(t not in x for t in texts), (s.name, k, x)
        assert REDACTED not in s.attributes, (s.name, "调用方传了原文，被丢弃")


def test_safe_attribute_values():
    assert safe("deepseek-flash") == "deepseek-flash" and safe("Qwen3-Embedding-0.6B") == "Qwen3-Embedding-0.6B"
    assert safe("ev_0123456789abcdef") and safe("3f1c2a9e-0b7d-4c55-9a51-6e0f3f6e8a10")
    assert safe(["fast", "verified"]) == ["fast", "verified"] and safe(3) == 3 and safe(True) is True
    for text in ["一线城市住宿上限是多少", "some free text with spaces", "x" * 200, ["fast", "原文"], {"a": 1}]:
        assert safe(text) is None


def test_redacted_values_are_counted(spans):
    with span("eaer.test", **{"eaer.question": "一线城市住宿上限是多少", "eaer.n": 3}):
        pass
    (s,) = spans
    assert "eaer.question" not in s.attributes and s.attributes["eaer.n"] == 3 and s.attributes[REDACTED] == 1


def test_every_node_of_a_query_is_traced_without_text(api, spans):
    client, conn, worker, _ = api
    publish(client, worker)
    ingest = {s.name for s in spans if s.name.startswith("eaer.ingest.")}
    assert {"eaer.ingest.parse_text", "eaer.ingest.classify_extract", "eaer.ingest.publish_index"} <= ingest
    assert all(s.attributes["eaer.job_status"] == "done" for s in spans if s.name.startswith("eaer.ingest."))
    spans.clear()

    r = client.post("/v1/query", headers=H("tok-staff"), json={"question": QUESTION})
    b = r.json()
    assert r.status_code == 200 and b["status"] == "completed" and "650元" in b["answer"], b
    (root,) = by_name(spans, "eaer.query")
    (execute,) = by_name(spans, "eaer.query.execute")
    tid = root.context.trace_id
    assert all(s.context.trace_id == tid for s in spans)                        # 一次查询一条 trace
    assert execute.parent.span_id == root.context.span_id
    assert root.attributes["eaer.mode_path"] == tuple(b["mode_path"]) and root.attributes["eaer.claims"] >= 1
    assert root.attributes["eaer.session_id"] == b["session_id"]

    nodes = {s.attributes["eaer.node"]: s for s in spans if s.name.startswith("eaer.node.")}
    assert set(nodes) == {"prepare", *NODES}                                   # 每个节点都有记录
    assert nodes["prepare"].parent.span_id == root.context.span_id
    for name in NODES:
        s = nodes[name]
        assert s.parent.span_id == execute.context.span_id and s.end_time > s.start_time
        assert {"eaer.mode_path", "eaer.model_calls", "gen_ai.usage.input_tokens"} <= set(s.attributes)
    assert nodes["first"].attributes["eaer.candidates"] >= 1
    assert nodes["first"].attributes["eaer.model_calls"] >= 1 and nodes["first"].attributes["gen_ai.usage.input_tokens"] > 0

    (ret, *_) = by_name(spans, "eaer.retrieve")
    assert ret.attributes["eaer.candidates.fused"] >= ret.attributes["eaer.candidates.rechecked"] >= 1
    assert ret.parent.span_id == nodes["first"].context.span_id
    gen = [s for s in spans if s.name == "eaer.model.generate"]
    assert gen and all(s.attributes["gen_ai.request.model"] == "stub-gen" and s.attributes["eaer.ok"] for s in gen)
    assert sum(s.attributes["gen_ai.usage.input_tokens"] for s in gen) > 0
    assert by_name(spans, "eaer.model.embed") and by_name(spans, "eaer.model.rerank")

    evidence = conn.execute("SELECT content FROM evidence").fetchall()
    assert_no_text(spans, QUESTION, b["answer"], *(e["content"] for e in evidence))
    # 原文只能经有权限控制的接口取得：会话接口只给提问人
    assert client.get(f"/v1/sessions/{b['session_id']}", headers=H("tok-other")).status_code == 404
    assert client.get(f"/v1/sessions/{b['session_id']}", headers=H("tok-staff")).status_code == 200


def test_a_resumed_execution_continues_the_same_trace(api, spans):
    """转入后台：提问时的 traceparent 存在查询状态里，另一个执行者（没有当前 span）从它续上同一条 trace。"""
    client, conn, worker, engine = api
    publish(client, worker)
    spans.clear()
    from eaer.domain.identity import Scope

    eng = engine(conn, "t1")
    eng.scope = Scope("t1", frozenset({"R_STAFF"}))
    with span("eaer.query") as root:
        state = eng.prepare(QUESTION, today=NOW.date())
    assert state["traceparent"] and traceparent() is None                     # 根 span 已结束，当前没有 span
    with span("eaer.query.execute", parent=state["traceparent"]):
        eng.result(eng.graph().invoke(state), "s1")
    (execute,) = by_name(spans, "eaer.query.execute")
    assert execute.context.trace_id == root.context.trace_id and execute.parent.span_id == root.context.span_id
    assert {s.attributes.get("eaer.node") for s in spans if s.name.startswith("eaer.node.")} == {"prepare", *NODES}


def test_exporter_configuration():
    from eaer.observability import TraceConfigError
    assert configure({}, "x") is False and configure({"EAER_TRACE_EXPORTER": "none"}, "x") is False
    with pytest.raises(TraceConfigError, match="none、console、otlp"):
        configure({"EAER_TRACE_EXPORTER": "jaeger"}, "x")


# ---------- 监控指标（13.3） ----------

def log_row(conn, mode_path, *, seconds, abstained=False, claims=(), status="completed", at=NOW):
    import json
    import uuid
    conn.execute(
        "INSERT INTO query_log (id, tenant_id, session_id, user_id, asked_at, question, mode, mode_path, status, claims, "
        "abstained, authz_decision, model_id, index_build_id, cost) VALUES (%s, 't1', %s, 'u', %s, '问题', %s, %s, %s, "
        "%s, %s, '{}', 'm', 'b1', %s)",
        (uuid.uuid4(), uuid.uuid4(), at, mode_path[-1], list(mode_path), status,
         json.dumps([{"status": s} for s in claims]), abstained, json.dumps({"seconds": seconds})))


def test_query_metrics(conn):
    from datetime import timedelta

    from eaer.metrics import collect
    for k in range(10):
        log_row(conn, ["fast"], seconds=1 + k / 10, claims=["supported"])
    log_row(conn, ["fast", "verified"], seconds=5, claims=["supported", "unverified"])
    log_row(conn, ["fast", "deep", "verified"], seconds=9, abstained=True)
    log_row(conn, ["fast"], seconds=0, status="failed")
    log_row(conn, ["fast"], seconds=1, at=NOW - timedelta(days=1))                  # 统计窗口之外
    m = collect(conn, NOW, NOW - timedelta(hours=1))
    q = m["queries"]
    assert q["completed"] == 12 and q["failed"] == 1
    assert q["by_mode"]["fast"]["n"] == 10 and q["by_mode"]["verified"]["n"] == 2
    assert q["by_mode"]["fast"]["p95_seconds"] == 1.855 and q["escalation_rate"] == round(2 / 12, 4)
    assert q["abstain_rate"] == round(1 / 12, 4)
    assert q["claims"] == {"supported": 11, "unverified": 1}
    assert {"queued", "running", "failed"} <= set(m["ingest_queue"]) and m["index_latency"]["target_p95"] == 60.0
    assert "问题" not in str(m)


def test_metrics_command(db_url, monkeypatch, capsys):
    import json

    from eaer.metrics import main
    monkeypatch.setenv("EAER_DATABASE_URL", db_url)
    assert main(["--since-minutes", "5"]) == 0
    assert json.loads(capsys.readouterr().out)["queries"]["completed"] == 0
