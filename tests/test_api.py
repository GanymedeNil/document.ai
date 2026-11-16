"""对外接口（开发文档 11.1、11.2；AC-11-1 的部分用例、AC-11-2）。"""
from __future__ import annotations

import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from dbutil import Clock
from eaer.api.app import Identity, TokenAuthenticator, create_app
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.storage import LocalFileStore

TOKENS = {
    "tok-editor": Identity("ed", "t1", frozenset({"editor"})),
    "tok-reviewer": Identity("rv", "t1", frozenset({"reviewer"})),
    "tok-both": Identity("boss", "t1", frozenset({"editor", "reviewer"})),
    "tok-other": Identity("eve", "t2", frozenset({"editor", "reviewer"})),
}


def H(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api(db_url, tmp_path):
    store = LocalFileStore(tmp_path / "files")
    clock = Clock()
    client = TestClient(create_app(db_url, store, TokenAuthenticator(TOKENS), clock))
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    worker = Worker(Ingestion(conn, store, clock))
    yield client, conn, worker
    conn.close()


def upload(client, name="r.html", body="<h1>报销规定</h1><p>本规定自2026年1月1日起施行。</p>", token="tok-editor", **form):
    return client.post("/v1/documents", headers=H(token), files={"file": (name, body.encode(), "text/html")},
                       data={"principals": "R_STAFF; R_FIN", **form})


def assert_error(resp, status, code):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["code"] == code and body["message"] and body["request_id"] == resp.headers["X-Request-Id"]


def test_authentication_and_roles(api):
    client, _, _ = api
    assert_error(client.get("/v1/review/tasks"), 401, "unauthenticated")
    assert_error(client.get("/v1/review/tasks", headers=H("nope")), 401, "unauthenticated")
    assert_error(client.get("/v1/review/tasks", headers=H("tok-editor")), 403, "forbidden")
    assert_error(upload(client, token="tok-reviewer"), 403, "forbidden")


def test_upload_and_duplicate(api):
    client, conn, _ = api
    r = upload(client)
    assert r.status_code == 201 and r.json()["status"] == "pending_review" and r.json()["duplicate"] is False
    again = upload(client, name="copy.html")
    assert again.status_code == 200 and again.json()["duplicate"] is True
    assert again.json()["version_id"] == r.json()["version_id"]
    acl = conn.execute("SELECT p.principals FROM document_version v JOIN acl_policy p ON p.id = v.acl_policy_id").fetchone()
    assert acl["principals"] == ["R_FIN", "R_STAFF"]


def test_bad_requests(api):
    client, _, _ = api
    assert_error(upload(client, name="plan.dwg"), 400, "invalid_request")     # 不支持的格式
    assert_error(client.post("/v1/documents", headers=H("tok-editor"), files={"file": ("a.html", b"<p>x</p>")}),
                 400, "invalid_request")       # 缺少 principals
    assert_error(client.post("/v1/review/decisions", headers=H("tok-reviewer"), json={"decisions": []}),
                 400, "invalid_request")
    assert_error(client.post("/v1/review/decisions", headers=H("tok-reviewer"),
                             json={"decisions": [{"task_id": "x", "decision": "maybe"}]}), 400, "invalid_request")


def test_review_and_archive_flow(api):
    client, _, worker = api
    doc = upload(client).json()["document_id"]
    # 待审核的文档没有可归档的版本
    assert_error(client.post(f"/v1/documents/{doc}/archive", headers=H("tok-editor")), 409, "invalid_state")
    worker.run_until_idle()
    tasks = client.get("/v1/review/tasks", headers=H("tok-reviewer")).json()
    assert len(tasks["items"]) == 1 and tasks["items"][0]["suggestions"]["doc_type"] == "normative"
    res = client.post("/v1/review/decisions", headers=H("tok-reviewer"),
                      json={"decisions": [{"task_id": tasks["items"][0]["task_id"], "decision": "approve"},
                                          {"task_id": str(uuid.uuid4()), "decision": "approve"}]}).json()
    assert [r["ok"] for r in res["results"]] == [True, False]
    a = client.post(f"/v1/documents/{doc}/archive", headers=H("tok-editor"))
    assert a.status_code == 200 and a.json()["status"] == "archived"
    assert_error(client.post(f"/v1/documents/{doc}/archive", headers=H("tok-editor")), 409, "invalid_state")
    u = client.post(f"/v1/documents/{doc}/unarchive", headers=H("tok-editor"))
    assert u.status_code == 200 and u.json()["status"] == "searchable"


def test_not_found_does_not_reveal_other_tenants(api):
    client, _, _ = api
    doc = upload(client).json()["document_id"]
    for path in (f"/v1/documents/{doc}/archive", f"/v1/documents/{uuid.uuid4()}/archive", "/v1/documents/abc/archive"):
        assert_error(client.post(path, headers=H("tok-other") if doc in path else H("tok-editor")), 404, "not_found")
    assert client.get("/v1/review/tasks", headers=H("tok-other")).json()["items"] == []


def test_openapi_lists_the_operations(api):
    client, _, _ = api
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/v1/documents", "/v1/review/tasks", "/v1/review/decisions",
            "/v1/documents/{document_id}/archive", "/v1/documents/{document_id}/unarchive"} <= set(paths)
