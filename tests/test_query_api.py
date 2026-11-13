"""提问与会话接口（开发文档 11.2）。离线：脚本化模型、内存索引，文档经真实的上传、审核、入库与索引流程。

- 响应格式与 query_log：每次提问（含失败的）都写一条，必填字段完整（AC-10-9 硬性）；
- 流式（AC-11-3）：事件顺序 mode 先于 delta、done 最后；claim 中的证据 id 都能经证据接口取得；
- 授权服务不可用时提问被拒（AC-10-5 硬性中提问接口的部分）；未配置提问服务时 503；模型失败时 502、不返回空答案；
- 会话只属于发起它的用户，别人的会话与不存在的会话一律 404；
- 归档与历史回答（AC-6-6 中“历史回答的引用做展示检查”）：归档后再问查不到，历史会话仍保留引用，
  点开引用得到“该资料已归档”，取消归档后恢复。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from eaer.adapters.authorizer import AuthorizerUnavailable, LocalAuthorizer
from eaer.adapters.online import ModelCallError
from eaer.api.app import TokenAuthenticator, create_app
from eaer.domain.identity import Identity
from eaer.ingestion.indexer import Indexer
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.orchestrator.engine import Engine
from eaer.retrieval.embed import HashingEmbedder
from eaer.retrieval.index import MemoryIndex
from eaer.retrieval.service import Retriever
from eaer.storage import LocalFileStore
from test_engine import Scripted, claim

TOKENS = {"tok-ed": Identity("ed", "t1", frozenset({"editor"})),
          "tok-rv": Identity("rv", "t1", frozenset({"reviewer"})),
          "tok-staff": Identity("staff", "t1", frozenset()),
          "tok-staff2": Identity("staff2", "t1", frozenset())}
ACCOUNTS = {"staff": {"tenant": "t1", "principals": ["R_STAFF"]}, "staff2": {"tenant": "t1", "principals": ["R_STAFF"]}}
NOW = datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc)
ANSWER = lambda p: {"abstain": False, "claims": [claim("一线城市住宿上限为每晚650元。", p, "650元")]}  # noqa: E731


def H(t):
    return {"Authorization": f"Bearer {t}"}


class Boom:
    def capabilities(self):
        from eaer.adapters.model import Capabilities
        return Capabilities(True, 32000, False, 0)

    def generate(self, messages, schema):
        raise ModelCallError("generate 调用失败（gen，尝试 3 次）：超时")


@pytest.fixture
def api(db_url, tmp_path):
    store, index, emb = LocalFileStore(tmp_path / "files"), MemoryIndex(), HashingEmbedder()
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    state = {"model": Scripted(ANSWER), "authz": lambda c: LocalAuthorizer(c, ACCOUNTS)}

    def engine(c, tenant):
        return Engine(c, Retriever(c, index, emb), state["model"], LocalAuthorizer(c, ACCOUNTS), store=store)

    app = create_app(db_url, store, TokenAuthenticator(TOKENS), lambda: NOW,
                     authorizer=lambda c: state["authz"](c), engine=engine)
    client = TestClient(app)
    worker = Worker(Ingestion(conn, store, lambda: NOW), Indexer(conn, index, emb, "b1"))
    body = ("<h1>差旅报销管理办法</h1><p>第一条　为规范差旅报销，制定本办法。</p><p>第二条　一线城市住宿上限为每晚650元。</p>"
            "<p>第三条　本办法自2026年1月1日起施行。</p>")
    up = client.post("/v1/documents", headers=H("tok-ed"), data={"principals": "R_STAFF"},
                     files={"file": ("p.html", body.encode(), "text/html")}).json()
    worker.run_until_idle()
    task = client.get("/v1/review/tasks", headers=H("tok-rv")).json()["items"][0]["task_id"]
    assert client.post("/v1/review/decisions", headers=H("tok-rv"), json={"decisions": [
        {"task_id": task, "decision": "approve", "corrections": {"doc_type": "normative"}}]}).json()["results"][0]["ok"]
    worker.run_until_idle()
    yield client, conn, state, worker, up["document_id"]
    conn.close()


def ask(client, token="tok-staff", **body):
    return client.post("/v1/query", headers=H(token), json={"question": "一线城市住宿上限是多少", **body})


REQUIRED = ("id", "tenant_id", "session_id", "user_id", "asked_at", "question", "mode", "mode_path", "status",
            "claims", "evidence_ids", "abstained", "authz_decision", "model_id", "index_build_id", "cost")


def assert_log_complete(conn):
    rows = conn.execute("SELECT * FROM query_log").fetchall()
    assert rows
    for r in rows:
        assert all(r[k] is not None for k in REQUIRED), {k: r[k] for k in REQUIRED if r[k] is None}
        assert r["authz_decision"]["principals"] == ["R_STAFF"]
        assert r["status"] == "failed" or r["answer"]
    return rows


def test_query_response_and_audit_log(api):
    client, conn, *_ = api
    r = ask(client)
    b = r.json()
    assert r.status_code == 200 and b["mode_path"] == ["fast", "verified"] and not b["abstained"]
    assert b["answer"] == "一线城市住宿上限为每晚650元。" and b["claims"][0]["citations"][0]["version_no"] == 1
    rows = assert_log_complete(conn)
    assert str(rows[0]["session_id"]) == b["session_id"] and rows[0]["model_id"] == "scripted"


def test_stream_event_order_and_cited_evidence_is_retrievable(api):
    client, conn, *_ = api
    r = ask(client, stream=True, question="2026 年 3 月 15 日一线城市住宿上限是多少")
    events = [(m.group(1), json.loads(m.group(2))) for m in re.finditer(r"event: (\w+)\ndata: (.*)\n\n", r.text)]
    names = [e for e, _ in events]
    assert names[0] == "mode" and names[-1] == "done"
    assert names.index("mode") < names.index("delta") < names.index("claim")
    assert "".join(d["text"] for e, d in events if e == "delta") == events[-1][1]["answer"]
    for _, c in [e for e in events if e[0] == "claim"]:
        for cite in c["citations"]:
            assert client.get(f"/v1/evidence/{cite['evidence_id']}", headers=H("tok-staff")).status_code == 200
    assert_log_complete(conn)


def test_authorizer_unavailable_rejects_queries(api):
    client, conn, state, *_ = api

    class Down:
        def resolve(self, i):
            raise AuthorizerUnavailable("down")

    state["authz"] = lambda c: Down()
    r = ask(client)
    assert r.status_code == 503 and r.json()["code"] == "authz_unavailable"
    assert conn.execute("SELECT count(*) AS n FROM query_log").fetchone()["n"] == 0


def test_query_service_not_configured(db_url, tmp_path):
    app = create_app(db_url, LocalFileStore(tmp_path), TokenAuthenticator(TOKENS), lambda: NOW,
                     authorizer=lambda c: LocalAuthorizer(c, ACCOUNTS))
    r = TestClient(app).post("/v1/query", headers=H("tok-staff"), json={"question": "x"})
    assert r.status_code == 503 and r.json()["code"] == "query_unavailable"


def test_model_failure_is_an_explicit_error_and_is_logged(api):
    client, conn, state, *_ = api
    state["model"] = Boom()
    r = ask(client)
    assert r.status_code == 502 and r.json()["code"] == "model_error" and "超时" in r.json()["message"]
    rows = assert_log_complete(conn)
    assert rows[0]["status"] == "failed" and rows[0]["answer"] is None


def test_sessions_belong_to_their_user(api):
    client, *_ = api
    sid = ask(client).json()["session_id"]
    assert ask(client, session_id=sid).json()["session_id"] == sid                      # 同一会话继续提问
    hist = client.get(f"/v1/sessions/{sid}", headers=H("tok-staff")).json()
    assert len(hist["items"]) == 2 and hist["items"][0]["claims"][0]["citations"]
    for path, token in ((f"/v1/sessions/{sid}", "tok-staff2"), ("/v1/sessions/" + "0" * 32, "tok-staff"),
                        ("/v1/sessions/nope", "tok-staff")):
        r = client.get(path, headers=H(token))
        assert r.status_code == 404 and r.json()["code"] == "not_found"
    assert ask(client, token="tok-staff2", session_id=sid).status_code == 404            # 不能续写别人的会话


def test_history_keeps_citations_after_archive(api):
    client, conn, state, worker, doc = api
    sid = ask(client).json()["session_id"]
    assert client.post(f"/v1/documents/{doc}/archive", headers=H("tok-ed")).status_code == 200
    worker.run_until_idle()
    again = ask(client).json()
    assert again["abstained"] and "650" not in again["answer"]                           # 归档后查不到
    cite = client.get(f"/v1/sessions/{sid}", headers=H("tok-staff")).json()["items"][0]["claims"][0]["citations"][0]
    ev = client.get(f"/v1/evidence/{cite['evidence_id']}", headers=H("tok-staff")).json()
    assert ev["status"] == "archived" and ev["notice"] == "该资料已归档" and "content" not in ev
    client.post(f"/v1/documents/{doc}/unarchive", headers=H("tok-ed"))
    worker.run_until_idle()
    assert "650" in client.get(f"/v1/evidence/{cite['evidence_id']}", headers=H("tok-staff")).json()["content"]
