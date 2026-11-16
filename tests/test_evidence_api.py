"""证据接口 GET /v1/evidence/{id}（开发文档 11.2、10.2）。

AC-10-6（硬性）：无权与不存在的 id 返回相同结果，均为 404——响应体除 request_id 外逐字相同；
待审核、已删除、其他租户的证据同样如此。已归档且有权时只给归档提示、不给原文（6.6），取消归档后恢复。
AC-10-5（硬性）中证据接口的部分：授权接口不可用或未配置时请求被拒绝（503），不退化为放行。
"""
from __future__ import annotations

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from dbutil import Clock
from eaer.adapters.authorizer import AuthorizerUnavailable, LocalAuthorizer
from eaer.api.app import TokenAuthenticator, create_app
from eaer.domain.identity import Identity
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.storage import LocalFileStore

TOKENS = {
    "tok-ed": Identity("ed", "t1", frozenset({"editor"})),
    "tok-rv": Identity("rv", "t1", frozenset({"reviewer"})),
    "tok-staff": Identity("staff", "t1", frozenset()),
    "tok-fin": Identity("fin", "t1", frozenset()),
    "tok-other": Identity("eve", "t2", frozenset({"editor", "reviewer"})),
    "tok-nobody": Identity("ghost", "t1", frozenset()),
}
ACCOUNTS = {"staff": {"tenant": "t1", "principals": ["R_STAFF"]},
            "fin": {"tenant": "t1", "principals": ["R_FIN"]},
            "eve": {"tenant": "t2", "principals": ["R_STAFF", "R_FIN"]}}


def H(t):
    return {"Authorization": f"Bearer {t}"}


class Down:
    def resolve(self, identity):
        raise AuthorizerUnavailable("目录服务超时")

    def check(self, scope, ids):
        raise AuthorizerUnavailable("目录服务超时")


@pytest.fixture
def env(db_url, tmp_path):
    store, clock = LocalFileStore(tmp_path / "files"), Clock()
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    mk = lambda authz: TestClient(create_app(db_url, store, TokenAuthenticator(TOKENS), clock, authz))  # noqa: E731
    client = mk(lambda c: LocalAuthorizer(c, ACCOUNTS))
    worker = Worker(Ingestion(conn, store, clock))

    def doc(body: str, principals: str, token="tok-ed", approve=True) -> tuple[str, str]:
        r = client.post("/v1/documents", headers=H(token), data={"principals": principals},
                        files={"file": ("p.html", f"<h1>规定</h1><p>{body}</p><p>本规定自2026年1月1日起施行。</p>".encode(),
                                        "text/html")})
        assert r.status_code == 201, r.text
        worker.run_until_idle()
        vid = r.json()["version_id"]
        if approve:
            rv = "tok-other" if token == "tok-other" else "tok-rv"
            task = next(t for t in client.get("/v1/review/tasks", headers=H(rv)).json()["items"]
                        if t["version_id"] == vid)
            res = client.post("/v1/review/decisions", headers=H(rv), json={"decisions": [
                {"task_id": task["task_id"], "decision": "approve", "corrections": {"doc_type": "normative"}}]})
            assert res.json()["results"][0]["ok"], res.text
            worker.run_until_idle()
        ev = conn.execute("SELECT id FROM evidence WHERE version_id = %s AND content LIKE %s",
                          (vid, f"%{body}%")).fetchone()["id"]
        return r.json()["document_id"], ev

    yield client, conn, doc, mk
    conn.close()


def body_without_request_id(resp) -> dict:
    b = resp.json()
    b.pop("request_id")
    return b


def test_permitted_searchable_evidence_returns_text_and_locator(env):
    client, _, doc, _ = env
    _, ev = doc("一线城市住宿上限为每晚650元。", "R_STAFF")
    r = client.get(f"/v1/evidence/{ev}", headers=H("tok-staff"))
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["status"] == "searchable" and "650元" in b["content"] and "para" in b["locator"]
    assert b["validity"]["valid_from"] == "2026-01-01" and b["version"]["version_no"] == 1
    assert "doc_date" in b["version"] and b["version"]["publisher"] is None      # 上传接口不带出处


def test_evidence_carries_document_date_and_publisher(env):
    """证据详情给出所在版本的文档日期与出处（数据源随文档给出，迁移 0012），与证据标注一致。"""
    client, conn, doc, _ = env
    _, ev = doc("差旅住宿标准见附表。", "R_STAFF")
    vid = conn.execute("SELECT version_id FROM evidence WHERE id = %s", (ev,)).fetchone()["version_id"]
    conn.execute("UPDATE document_version SET doc_date = '2025-12-20', publisher = '财务部' WHERE id = %s", (vid,))
    b = client.get(f"/v1/evidence/{ev}", headers=H("tok-staff")).json()
    assert b["version"]["doc_date"] == "2025-12-20" and b["version"]["publisher"] == "财务部"


def test_unauthorized_and_nonexistent_are_indistinguishable(env):
    client, conn, doc, _ = env
    _, fin_only = doc("部门经理单笔采购审批上限10万元。", "R_FIN")
    _, pending = doc("待审核文件中的条款。", "R_STAFF", approve=False)
    _, other_tenant = doc("另一租户的条款。", "R_STAFF", token="tok-other")
    _, deleted = doc("将被删除的条款。", "R_STAFF", approve=False)
    task = client.get("/v1/review/tasks", headers=H("tok-rv")).json()["items"]
    tid = next(t["task_id"] for t in task if conn.execute(
        "SELECT 1 FROM evidence WHERE id = %s AND version_id = %s", (deleted, t["version_id"])).fetchone())
    client.post("/v1/review/decisions", headers=H("tok-rv"), json={"decisions": [{"task_id": tid, "decision": "delete"}]})

    nonexistent = client.get("/v1/evidence/ev_" + "0" * 64, headers=H("tok-staff"))
    assert nonexistent.status_code == 404
    expected = body_without_request_id(nonexistent)
    assert expected == {"code": "not_found", "message": "证据不存在"}
    for ev in (fin_only, pending, other_tenant, deleted, "not-an-id", "ev_' OR '1'='1"):
        r = client.get(f"/v1/evidence/{ev}", headers=H("tok-staff"))
        assert r.status_code == 404 and body_without_request_id(r) == expected, (ev, r.text)
    assert client.get(f"/v1/evidence/{fin_only}", headers=H("tok-fin")).status_code == 200     # 对照：有权即可取到


def test_archived_shows_notice_without_text_and_comes_back_after_unarchive(env):
    client, _, doc, _ = env
    did, ev = doc("出差伙食补贴每天80元。", "R_STAFF")
    assert client.post(f"/v1/documents/{did}/archive", headers=H("tok-ed")).status_code == 200
    r = client.get(f"/v1/evidence/{ev}", headers=H("tok-staff"))
    assert r.status_code == 200 and r.json()["status"] == "archived" and r.json()["notice"] == "该资料已归档"
    assert "content" not in r.json() and "locator" not in r.json()
    assert client.get(f"/v1/evidence/{ev}", headers=H("tok-fin")).status_code == 404       # 无权：仍是 404
    client.post(f"/v1/documents/{did}/unarchive", headers=H("tok-ed"))
    assert "80元" in client.get(f"/v1/evidence/{ev}", headers=H("tok-staff")).json()["content"]


def test_revocation_takes_effect_immediately(env):
    client, conn, doc, _ = env
    did, ev = doc("一线城市住宿上限为每晚650元。", "R_STAFF")
    assert client.get(f"/v1/evidence/{ev}", headers=H("tok-staff")).status_code == 200
    from eaer.lifecycle.service import Lifecycle
    import uuid
    Lifecycle(conn).change_acl(uuid.UUID(did), ["R_FIN"], "admin")      # 不等索引更新，数据库核对立即生效
    assert client.get(f"/v1/evidence/{ev}", headers=H("tok-staff")).status_code == 404
    assert client.get(f"/v1/evidence/{ev}", headers=H("tok-fin")).status_code == 200


@pytest.mark.parametrize("authz", [None, lambda c: Down()], ids=["未配置", "不可用"])
def test_authorizer_unavailable_rejects(env, authz):
    _, _, doc, mk = env
    _, ev = doc("一线城市住宿上限为每晚650元。", "R_STAFF")
    r = mk(authz).get(f"/v1/evidence/{ev}", headers=H("tok-staff"))
    assert r.status_code == 503 and r.json()["code"] == "authz_unavailable"


def test_identity_without_authorization_scope_and_missing_token(env):
    client, _, doc, _ = env
    _, ev = doc("一线城市住宿上限为每晚650元。", "R_STAFF")
    r = client.get(f"/v1/evidence/{ev}", headers=H("tok-nobody"))
    assert r.status_code == 403 and r.json()["code"] == "forbidden"
    assert client.get(f"/v1/evidence/{ev}").status_code == 401


def test_framework_errors_use_the_uniform_format(env):
    client, _, _, _ = env
    for r in (client.get("/v1/evidence/../etc/passwd", headers=H("tok-staff")), client.get("/v1/nope")):
        assert r.status_code == 404 and r.json()["code"] == "not_found"
        assert r.json()["request_id"] == r.headers["X-Request-Id"]
    r = client.delete("/v1/review/decisions", headers=H("tok-rv"))
    assert r.status_code == 405 and set(r.json()) == {"code", "message", "request_id"}
