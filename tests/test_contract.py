"""契约测试（开发文档 11；AC-11-1）：实现与 OpenAPI 文件 eaer/api/openapi.yaml 一致。

- 文件本身：OpenAPI 3.1，$ref 全部能解析，schema 都是合法的 JSON Schema；操作与开发文档 11.2 的 11 个一致；
  每个操作至少声明一个成功、一个错误响应；每个响应声明 X-Request-Id，错误响应都是 {code, message, request_id}；
  接口进程的 /openapi.json 就是这份文件。
- 路由与参数：实现的路由与文件中的操作一一对应；路径参数、查询参数与请求体字段（名称、是否必填，逐层）
  与实现声明的一致。
- 行为：每个操作把文件中为它声明的每个状态码（及媒体类型）都走一遍——正常与错误路径都在内，多于 AC-11-1
  要求的“各至少一个”；所有响应经 tests/contract.py 按文件检查，成功的请求本身也须符合文件。
- 文件不允许的请求（缺必填、类型不对、取值超范围、格式不对）实现一律返回 400；文件允许的边界请求实现不报格式错误。

离线：脚本化模型、内存索引；文档经真实的上传、审核、入库与索引流程，数据库是真实的 PostgreSQL。
"""
from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone

import psycopg
import pytest
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from psycopg.rows import dict_row

from contract import Checked, Contract, pointer
from eaer.adapters.authorizer import AuthorizerUnavailable, LocalAuthorizer
from eaer.api.app import TokenAuthenticator, create_app, openapi_spec
from eaer.domain.identity import Identity
from eaer.ingestion.indexer import Indexer
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.orchestrator.engine import Engine
from eaer.retrieval.embed import HashingEmbedder
from eaer.retrieval.index import MemoryIndex
from eaer.retrieval.service import Retriever
from eaer.storage import LocalFileStore
from test_async import Gated, wait_for
from test_engine import claim
from test_query_api import Boom

# 开发文档 11.2 的表：对外的 11 个 HTTP 操作
DEV_DOC_OPERATIONS = {
    ("POST", "/v1/query"), ("GET", "/v1/sessions/{session_id}"), ("POST", "/v1/sessions/{session_id}/cancel"),
    ("GET", "/v1/evidence/{evidence_id}"), ("POST", "/v1/documents"), ("GET", "/v1/review/tasks"),
    ("POST", "/v1/review/decisions"), ("POST", "/v1/documents/{document_id}/archive"),
    ("POST", "/v1/documents/{document_id}/unarchive"), ("GET", "/v1/config/forced-verify"),
    ("POST", "/v1/config/forced-verify"),
}

TOKENS = {
    "tok-ed": Identity("ed", "t1", frozenset({"editor"})),
    "tok-rv": Identity("rv", "t1", frozenset({"reviewer"})),
    "tok-admin": Identity("admin", "t1", frozenset({"admin"})),
    "tok-staff": Identity("staff", "t1", frozenset()),
    "tok-staff2": Identity("staff2", "t1", frozenset()),
    "tok-fin": Identity("fin", "t1", frozenset()),
    "tok-nobody": Identity("nobody", "t1", frozenset()),            # 授权服务里没有这个人：无法确定授权范围
    "tok-eve": Identity("eve", "t2", frozenset({"editor", "reviewer"})),
}
ACCOUNTS = {"staff": {"tenant": "t1", "principals": ["R_STAFF"]}, "staff2": {"tenant": "t1", "principals": ["R_STAFF"]},
            "fin": {"tenant": "t1", "principals": ["R_FIN"]}, "admin": {"tenant": "t1", "principals": ["R_STAFF"]},
            "eve": {"tenant": "t2", "principals": ["R_STAFF"]}}
NOW = datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc)
Q = "一线城市住宿上限是多少"
NORM = ("<h1>差旅报销管理办法</h1><p>第一条　为规范差旅报销，制定本办法。</p><p>第二条　一线城市住宿上限为每晚650元。</p>"
        "<p>第三条　本办法自2026年1月1日起施行。</p>")
ANSWER = lambda p: {"abstain": False, "claims": [claim("一线城市住宿上限为每晚650元。", p, "650元")]}  # noqa: E731


def H(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def html(title: str, text: str) -> str:
    return f"<h1>{title}</h1><p>第一条　{text}</p><p>本规定自2026年1月1日起施行。</p>"


class Down:
    def resolve(self, identity):
        raise AuthorizerUnavailable("目录服务超时")

    def check(self, scope, ids):
        raise AuthorizerUnavailable("目录服务超时")


class Env:
    def __init__(self, db_url: str, tmp_path):
        self.db_url, self.contract = db_url, Contract()
        self.store, self.index, self.emb = LocalFileStore(tmp_path / "files"), MemoryIndex(), HashingEmbedder()
        self.conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
        self.st = {"model": Gated(ANSWER), "sync": 5.0, "authz": lambda c: LocalAuthorizer(c, ACCOUNTS)}
        self.worker = Worker(Ingestion(self.conn, self.store, lambda: NOW),
                             Indexer(self.conn, self.index, self.emb, "b1"))
        self.client = self.app()
        self.doc, version = self.publish(NORM, "p.html")
        self.ev = self.conn.execute("SELECT id FROM evidence WHERE version_id = %s AND content LIKE '%%650元%%'",
                                    (version,)).fetchone()["id"]

    def engine(self, c, tenant):
        return Engine(c, Retriever(c, self.index, self.emb), self.st["model"], LocalAuthorizer(c, ACCOUNTS),
                      store=self.store, budgets={"sync_seconds": self.st["sync"]})

    def app(self, *, engine: bool = True, authorizer=None) -> Checked:
        app = create_app(self.db_url, self.store, TokenAuthenticator(TOKENS), lambda: NOW,
                         authorizer=authorizer or (lambda c: self.st["authz"](c)),
                         engine=self.engine if engine else None, owner="contract")
        return Checked(TestClient(app), self.contract)

    def upload(self, body: str, name: str, token: str = "tok-ed", principals: str = "R_STAFF", **form):
        return self.client.post("/v1/documents", headers=H(token), data={"principals": principals, **form},
                                files={"file": (name, body.encode(), "text/html")})

    def pending(self, body: str, name: str) -> str:
        """上传并完成解析与抽取，返回审核任务 id。"""
        version = self.upload(body, name).json()["version_id"]
        self.worker.run_until_idle()
        page = self.client.get("/v1/review/tasks", headers=H("tok-rv"), params={"limit": 200}).json()
        return next(t["task_id"] for t in page["items"] if t["version_id"] == version)

    def publish(self, body: str, name: str) -> tuple[str, str]:
        up = self.upload(body, name).json()
        self.worker.run_until_idle()
        page = self.client.get("/v1/review/tasks", headers=H("tok-rv"), params={"limit": 200}).json()
        task = next(t["task_id"] for t in page["items"] if t["version_id"] == up["version_id"])
        res = self.client.post("/v1/review/decisions", headers=H("tok-rv"), json={"decisions": [
            {"task_id": task, "decision": "approve", "corrections": {"doc_type": "normative"}}]}).json()
        assert res["results"][0]["ok"], res
        self.worker.run_until_idle()
        return up["document_id"], up["version_id"]

    def ask(self, token: str = "tok-staff", client: Checked | None = None, **body):
        return (client or self.client).post("/v1/query", headers=H(token), json={"question": Q, **body})


@pytest.fixture
def env(db_url, tmp_path):
    e = Env(db_url, tmp_path)
    yield e
    e.conn.close()


def expect(resp, status: int, code: str | None = None):
    assert resp.status_code == status, resp.text
    if code is not None:
        assert resp.json()["code"] == code, resp.text
    return resp


# ---------- 文件本身 ----------

def _walk(node, path=()):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, path + (str(i),))


def _is_error_schema(c: Contract, schema: dict) -> bool:
    if "$ref" in schema:
        if schema["$ref"] == "#/components/schemas/Error":
            return True
        return _is_error_schema(c, c.at(schema["$ref"].removeprefix("#")))
    return any(_is_error_schema(c, s) for s in schema.get("allOf", []))


def test_spec_is_well_formed():
    spec = openapi_spec()
    c = Contract(spec)
    assert spec["openapi"].startswith("3.1.")
    json.dumps(spec)                                          # 没有被 YAML 读成日期等非 JSON 值的取值
    for path, node in _walk(spec):
        if "$ref" in node and isinstance(node["$ref"], str):
            c.at(node["$ref"].removeprefix("#"))              # 每个 $ref 都能解析
    for name, schema in spec["components"]["schemas"].items():
        Draft202012Validator.check_schema(schema)
    assert spec["security"] == [{"bearer": []}] and spec["components"]["securitySchemes"]["bearer"]["scheme"] == "bearer"
    ids = [o.id for o in c.operations]
    assert len(ids) == len(set(ids))
    for op in c.operations:
        assert op.spec.get("summary") and op.spec.get("tags"), op.id
        statuses = set(op.spec["responses"])
        assert all(isinstance(s, str) for s in statuses), op.id
        assert any(s.startswith("2") for s in statuses) and any(s[0] in "45" for s in statuses), op.id
        for status, r in op.spec["responses"].items():
            r, _ = c.deref(r, "")
            assert "X-Request-Id" in r.get("headers", {}), (op.id, status)
            assert r.get("content"), (op.id, status)
            for media, m in r["content"].items():
                Draft202012Validator.check_schema(m["schema"])
                if int(status) >= 400:
                    assert media == "application/json" and _is_error_schema(c, m["schema"]), (op.id, status)
            if "text/event-stream" in r["content"]:
                assert r["content"]["text/event-stream"].get("x-events"), op.id


def test_operations_are_the_eleven_in_the_dev_doc():
    assert {(o.method, o.template) for o in Contract().operations} == DEV_DOC_OPERATIONS


def test_served_openapi_is_the_file(env):
    raw = TestClient(create_app(env.db_url, env.store, TokenAuthenticator(TOKENS), lambda: NOW))
    assert raw.get("/openapi.json").json() == json.loads(json.dumps(openapi_spec()))
    assert raw.get("/docs").status_code == 200


# ---------- 路由与参数 ----------

def _app_without_services(tmp_path):
    return create_app("postgresql://unused", LocalFileStore(tmp_path), TokenAuthenticator({}))


def test_routes_match_the_operations(tmp_path):
    app = _app_without_services(tmp_path)
    routes = {(m, r.path) for r in app.routes if isinstance(r, APIRoute) for m in r.methods}
    ops = {(o.method, o.template) for o in Contract().operations}
    assert routes - ops == set(), "实现中有文件里没有的操作"
    assert ops - routes == set(), "文件中有实现里没有的操作"


def _deref(doc: dict, schema: dict) -> dict:
    while "$ref" in schema:
        node = doc
        for part in schema["$ref"].removeprefix("#/").split("/"):
            node = node[part.replace("~1", "/").replace("~0", "~")]
        schema = node
    return schema


def shape(doc: dict, schema: dict):
    """把 schema 归约成“对象的字段名与必填 / 数组的元素 / 其他”，用来比较实现与文件声明的输入结构。"""
    schema = _deref(doc, schema)
    branches = schema.get("anyOf") or schema.get("oneOf")
    if branches:
        real = [b for b in branches if _deref(doc, b).get("type") != "null"]
        if len(real) == 1:
            return shape(doc, real[0])
    if "properties" in schema:
        return {"fields": {k: shape(doc, v) for k, v in schema["properties"].items()},
                "required": sorted(schema.get("required", []))}
    if schema.get("type") == "array" or "items" in schema:
        return {"items": shape(doc, schema.get("items", {}))}
    return "·"


def test_parameters_and_request_bodies_match_the_implementation(tmp_path):
    app = _app_without_services(tmp_path)
    generated = get_openapi(title="impl", version="0", routes=app.routes)
    c = Contract()
    for op in c.operations:
        impl = generated["paths"][op.template][op.method.lower()]
        mine = {(p["in"], p["name"]): bool(p.get("required")) for p, _ in c.parameters(op)}
        theirs = {(p["in"], p["name"]): bool(p.get("required")) for p in impl.get("parameters", [])
                  if p["in"] != "header"}                     # 身份令牌在 security 中声明
        assert mine == theirs, op.id
        rb_mine, rb_theirs = op.spec.get("requestBody"), impl.get("requestBody")
        assert (rb_mine is None) == (rb_theirs is None), op.id
        if rb_mine:
            assert set(rb_mine["content"]) == set(rb_theirs["content"]), op.id
            for media in rb_mine["content"]:
                assert shape(c.spec, rb_mine["content"][media]["schema"]) == \
                    shape(generated, rb_theirs["content"][media]["schema"]), (op.id, media)


# ---------- 行为：每个操作的每个声明状态 ----------

def s_query(e: Env):
    ok = expect(e.ask(), 200).json()
    assert ok["status"] == "completed" and ok["claims"]
    stream = expect(e.ask(question="一线城市的住宿上限", stream=True), 200)
    assert stream.headers["content-type"].startswith("text/event-stream")
    expect(e.ask(question=""), 400, "invalid_request")
    expect(e.client.post("/v1/query", headers=H("tok-staff"), json={}), 400, "invalid_request")
    expect(e.client.post("/v1/query", json={"question": Q}), 401, "unauthenticated")
    expect(e.ask("tok-nobody"), 403, "forbidden")
    other = expect(e.ask("tok-staff2"), 200).json()["session_id"]
    expect(e.ask(session_id=other), 404, "not_found")             # 别人的会话
    expect(e.ask(session_id=str(uuid.uuid4())), 200)              # 客户端生成的新会话 id
    e.st["model"] = Boom()
    expect(e.ask(question="出差伙食补贴是多少"), 502, "model_error")
    e.st["model"] = Gated(ANSWER)
    e.st["authz"] = lambda c: Down()
    expect(e.ask(), 503, "authz_unavailable")
    e.st["authz"] = lambda c: LocalAuthorizer(c, ACCOUNTS)
    expect(e.ask(client=e.app(engine=False)), 503, "query_unavailable")


def s_get_session(e: Env):
    sid = e.ask().json()["session_id"]
    e.st["model"] = Boom()
    expect(e.ask(question="出差伙食补贴是多少", session_id=sid), 502)   # 失败的提问也在会话里
    items = expect(e.client.get(f"/v1/sessions/{sid}", headers=H("tok-staff")), 200).json()["items"]
    assert [i["status"] for i in items] == ["completed", "failed"] and items[0]["claims"]
    expect(e.client.get(f"/v1/sessions/{sid}"), 401, "unauthenticated")
    for path in (f"/v1/sessions/{uuid.uuid4()}", "/v1/sessions/abc"):
        expect(e.client.get(path, headers=H("tok-staff")), 404, "not_found")
    expect(e.client.get(f"/v1/sessions/{sid}", headers=H("tok-staff2")), 404, "not_found")


def s_cancel_session(e: Env):
    e.st["sync"] = 1.0
    entered, release = e.st["model"].gate("judge")             # 卡在“验证”步骤，接口超时返回 running
    running = expect(e.ask(), 200).json()
    assert running["status"] == "running" and entered.wait(5)
    sid = running["session_id"]
    item = expect(e.client.get(f"/v1/sessions/{sid}", headers=H("tok-staff")), 200).json()["items"][0]
    assert item["status"] == "running" and item["progress"]["done"]
    c = expect(e.client.post(f"/v1/sessions/{sid}/cancel", headers=H("tok-staff")), 200).json()
    assert c["items"][0]["status"] == "cancelling"
    release.set()
    wait_for(lambda: e.conn.execute("SELECT status FROM query_task").fetchone()["status"] == "cancelled")
    expect(e.client.post(f"/v1/sessions/{sid}/cancel", headers=H("tok-staff")), 409, "not_running")
    expect(e.client.post(f"/v1/sessions/{sid}/cancel"), 401, "unauthenticated")
    expect(e.client.post(f"/v1/sessions/{uuid.uuid4()}/cancel", headers=H("tok-staff")), 404, "not_found")
    expect(e.client.post(f"/v1/sessions/{sid}/cancel", headers=H("tok-staff2")), 404, "not_found")


def s_get_evidence(e: Env):
    url = f"/v1/evidence/{e.ev}"
    assert expect(e.client.get(url, headers=H("tok-staff")), 200).json()["status"] == "searchable"
    expect(e.client.post(f"/v1/documents/{e.doc}/archive", headers=H("tok-ed")), 200)
    archived = expect(e.client.get(url, headers=H("tok-staff")), 200).json()
    assert archived["status"] == "archived" and "content" not in archived
    expect(e.client.post(f"/v1/documents/{e.doc}/unarchive", headers=H("tok-ed")), 200)
    expect(e.client.get(url), 401, "unauthenticated")
    expired = e.app(authorizer=lambda c: LocalAuthorizer(c, ACCOUNTS, ttl_seconds=0))
    expect(expired.get(url, headers=H("tok-staff")), 401, "unauthenticated")
    expect(e.client.get(url, headers=H("tok-nobody")), 403, "forbidden")
    a = expect(e.client.get(url, headers=H("tok-fin")), 404, "not_found").json()
    b = expect(e.client.get("/v1/evidence/ev_0000000000000000", headers=H("tok-staff")), 404, "not_found").json()
    assert {**a, "request_id": ""} == {**b, "request_id": ""}  # 无权与不存在不区分（AC-10-6）
    expect(e.app(authorizer=lambda c: Down()).get(url, headers=H("tok-staff")), 503, "authz_unavailable")


def s_upload(e: Env):
    body = html("会议费管理办法", "会议费每人每天上限为500元。")
    expect(e.upload(body, "m.html"), 201)
    assert expect(e.upload(body, "copy.html"), 200).json()["duplicate"] is True
    expect(e.upload(html("培训费管理办法", "培训费每人每天上限为400元。"), "t.html", source_ref="T-001", title="培训费",
                    import_batch_id="2026-batch-1", batch_id=str(uuid.uuid4())), 201)
    expect(e.upload(body, "plan.dwg"), 400, "invalid_request")                  # 不支持的格式
    expect(e.upload(html("x", "y"), "x.html", principals=" ; "), 400, "invalid_request")
    expect(e.client.post("/v1/documents", headers=H("tok-ed"), files={"file": ("a.html", b"<p>x</p>")}),
           400, "invalid_request")                                              # 缺 principals
    expect(e.client.post("/v1/documents", headers=H("tok-ed"), data={"principals": "R_STAFF"}),
           400, "invalid_request")                                              # 缺文件
    expect(e.client.post("/v1/documents", data={"principals": "R_STAFF"},
                         files={"file": ("a.html", b"<p>x</p>")}), 401, "unauthenticated")
    expect(e.upload(body, "r.html", token="tok-rv"), 403, "forbidden")


def s_list_review_tasks(e: Env):
    for i in range(3):
        e.upload(html(f"制度{i}", f"第{i}项费用上限为{100 + i}元。"), f"d{i}.html")
    e.worker.run_until_idle()
    first = expect(e.client.get("/v1/review/tasks", headers=H("tok-rv"), params={"limit": 2}), 200).json()
    assert len(first["items"]) == 2 and first["next_cursor"]
    rest = expect(e.client.get("/v1/review/tasks", headers=H("tok-rv"),
                               params={"limit": 2, "cursor": first["next_cursor"]}), 200).json()
    assert len(rest["items"]) == 1 and rest["next_cursor"] is None
    assert expect(e.client.get("/v1/review/tasks", headers=H("tok-rv"), params={"status": "approved"}), 200).json()["items"]
    expect(e.client.get("/v1/review/tasks", headers=H("tok-rv"), params={"batch_id": str(uuid.uuid4())}), 200)
    for bad in ({"limit": 0}, {"limit": 201}, {"limit": "x"}, {"status": "bogus"}, {"batch_id": "xyz"}):
        expect(e.client.get("/v1/review/tasks", headers=H("tok-rv"), params=bad), 400, "invalid_request")
    expect(e.client.get("/v1/review/tasks"), 401, "unauthenticated")
    expect(e.client.get("/v1/review/tasks", headers=H("tok-ed")), 403, "forbidden")


def s_review_decisions(e: Env):
    a = e.pending(html("车辆管理办法", "公务用车每公里补贴1.2元。"), "car.html")
    b = e.pending(html("草稿", "这是一份要删除的草稿。"), "draft.html")
    res = expect(e.client.post("/v1/review/decisions", headers=H("tok-rv"), json={"decisions": [
        {"task_id": a, "decision": "approve", "corrections": {
            "doc_type": "normative", "valid_from": {"value": "2026-02-01"}, "valid_to": None,
            "relations": [{"relation": "references", "to_document_id": e.doc, "to_version_no": 1}]}},
        {"task_id": b, "decision": "delete"},
        {"task_id": str(uuid.uuid4()), "decision": "approve"}]}), 200).json()["results"]
    assert [r["ok"] for r in res] == [True, True, False], res
    for bad in ({}, {"decisions": []}, {"decisions": [{"task_id": a, "decision": "maybe"}]},
                {"decisions": [{"task_id": a, "decision": "approve", "corrections": {"doc_type": "bogus"}}]}):
        expect(e.client.post("/v1/review/decisions", headers=H("tok-rv"), json=bad), 400, "invalid_request")
    expect(e.client.post("/v1/review/decisions", json={"decisions": [{"task_id": a, "decision": "delete"}]}),
           401, "unauthenticated")
    expect(e.client.post("/v1/review/decisions", headers=H("tok-ed"),
                         json={"decisions": [{"task_id": a, "decision": "delete"}]}), 403, "forbidden")


def s_archive(e: Env):
    url = f"/v1/documents/{e.doc}/archive"
    r = expect(e.client.post(url, headers=H("tok-ed")), 200).json()
    assert r["status"] == "archived" and r["versions"]
    expect(e.client.post(url, headers=H("tok-ed")), 409, "invalid_state")                 # 已经归档
    for path, token in ((f"/v1/documents/{uuid.uuid4()}/archive", "tok-ed"), ("/v1/documents/abc/archive", "tok-ed"),
                        (url, "tok-eve")):                                                # 其他租户的文档
        expect(e.client.post(path, headers=H(token)), 404, "not_found")
    expect(e.client.post(url), 401, "unauthenticated")
    expect(e.client.post(url, headers=H("tok-rv")), 403, "forbidden")


def s_unarchive(e: Env):
    url = f"/v1/documents/{e.doc}/unarchive"
    expect(e.client.post(url, headers=H("tok-ed")), 409, "invalid_state")                 # 没有已归档的版本
    expect(e.client.post(f"/v1/documents/{e.doc}/archive", headers=H("tok-ed")), 200)
    assert expect(e.client.post(url, headers=H("tok-ed")), 200).json()["status"] == "searchable"
    for path, token in ((f"/v1/documents/{uuid.uuid4()}/unarchive", "tok-ed"), (url, "tok-eve")):
        expect(e.client.post(path, headers=H(token)), 404, "not_found")
    expect(e.client.post(url), 401, "unauthenticated")
    expect(e.client.post(url, headers=H("tok-rv")), 403, "forbidden")


def s_get_forced_verify(e: Env):
    empty = expect(e.client.get("/v1/config/forced-verify", headers=H("tok-staff")), 200).json()
    assert empty["builtin"] and empty["rules"] == []
    expect(e.client.post("/v1/config/forced-verify", headers=H("tok-admin"), json={"kind": "keyword", "value": "安全"}), 201)
    assert expect(e.client.get("/v1/config/forced-verify", headers=H("tok-staff")), 200).json()["rules"]
    expect(e.client.get("/v1/config/forced-verify"), 401, "unauthenticated")


def s_add_forced_verify(e: Env):
    url, admin = "/v1/config/forced-verify", H("tok-admin")
    expect(e.client.post(url, headers=admin, json={"kind": "keyword", "value": "安全生产", "description": "安全"}), 201)
    expect(e.client.post(url, headers=admin, json={"kind": "keyword", "value": "安全生产"}), 200)   # 已存在
    expect(e.client.post(url, headers=admin, json={"kind": "document", "value": e.doc}), 201)
    expect(e.client.post(url, headers=admin, json={"kind": "doc_type", "value": "record"}), 201)
    expect(e.client.post(url, headers=admin, json={"kind": "all", "value": None, "description": None}), 201)
    for bad in ({"kind": "bogus"}, {"kind": "keyword"}, {"kind": "doc_type", "value": "x"},
                {"kind": "document", "value": "abc"},
                {"kind": "document", "value": str(uuid.uuid4())}):                # 文件允许、业务上不存在的文档
        expect(e.client.post(url, headers=admin, json=bad), 400, "invalid_request")
    expect(e.client.post(url, json={"kind": "all"}), 401, "unauthenticated")
    expect(e.client.post(url, headers=H("tok-staff"), json={"kind": "all"}), 403, "forbidden")


SCENARIOS = {
    "query": s_query, "getSession": s_get_session, "cancelSession": s_cancel_session, "getEvidence": s_get_evidence,
    "uploadDocument": s_upload, "listReviewTasks": s_list_review_tasks, "submitReviewDecisions": s_review_decisions,
    "archiveDocument": s_archive, "unarchiveDocument": s_unarchive, "getForcedVerify": s_get_forced_verify,
    "addForcedVerifyRule": s_add_forced_verify,
}


def test_the_checker_catches_mismatches(env):
    """检查器本身：文件与实现有出入时确实报错（否则上面的场景全部通过也说明不了什么）。"""
    def broken(edit) -> Checked:
        spec = copy.deepcopy(openapi_spec())
        edit(spec)
        return Checked(env.client.client, Contract(spec))

    def fails(client: Checked, call, needle: str):
        with pytest.raises(AssertionError) as err:
            call(client)
        assert needle in str(err.value), str(err.value)

    def drop_field(spec):                                     # 响应多出文件里没有的字段
        s = spec["components"]["schemas"]["UploadResult"]
        del s["properties"]["duplicate"]
        s["required"].remove("duplicate")
    fails(broken(drop_field), lambda c: c.post("/v1/documents", headers=H("tok-ed"), data={"principals": "R_STAFF"},
                                               files={"file": ("n.html", html("新", "新内容").encode())}), "响应体不符合文件")

    def drop_409(spec):                                       # 返回了文件里没有声明的状态码
        del spec["paths"]["/v1/documents/{document_id}/unarchive"]["post"]["responses"]["409"]
    fails(broken(drop_409), lambda c: c.post(f"/v1/documents/{env.doc}/unarchive", headers=H("tok-ed")), "状态码未在文件中声明")

    def tighter_limit(spec):                                  # 实现接受了文件不允许的请求
        for p in spec["paths"]["/v1/review/tasks"]["get"]["parameters"]:
            if p["name"] == "limit":
                p["schema"]["maximum"] = 100
    fails(broken(tighter_limit), lambda c: c.get("/v1/review/tasks", headers=H("tok-rv"), params={"limit": 150}),
          "实现接受了文件不允许的请求")

    def no_delta(spec):                                       # 流式事件未声明
        del spec["paths"]["/v1/query"]["post"]["responses"]["200"]["content"]["text/event-stream"]["x-events"]["delta"]
    fails(broken(no_delta), lambda c: c.post("/v1/query", headers=H("tok-staff"), json={"question": Q, "stream": True}),
          "事件 delta 未在 x-events 中声明")

    def citation_needs_page(spec):                            # 嵌套对象缺必填字段
        spec["components"]["schemas"]["Citation"]["required"].append("page")
    fails(broken(citation_needs_page), lambda c: c.post("/v1/query", headers=H("tok-staff"), json={"question": Q}),
          "响应体不符合文件")


def test_every_operation_has_a_scenario():
    assert set(SCENARIOS) == {o.id for o in Contract().operations}


@pytest.mark.parametrize("op_id", sorted(SCENARIOS))
def test_operation_conforms_to_the_file(env, op_id):
    SCENARIOS[op_id](env)
    op = env.contract.by_id(op_id)
    missing = env.contract.declared(op) - env.contract.seen[op_id]
    assert not missing, f"{op_id}：文件中声明的这些响应没有被测到：{sorted(missing)}"


# ---------- 文件不允许的请求实现一律 400 ----------

BASES = {
    "query": {"question": Q},
    "submitReviewDecisions": {"decisions": [{"task_id": "t-1", "decision": "delete"}]},
    "addForcedVerifyRule": {"kind": "keyword", "value": "安全生产"},
}


def _wrong_type(schema: dict):
    types = schema.get("type", [])
    return 12345 if "object" in (types if isinstance(types, list) else [types]) else {"k": 1}


def generated_invalid(c: Contract, op_id: str) -> list[tuple[str, dict]]:
    """按请求体 schema 的顶层字段自动生成：缺必填、类型不对、枚举外取值、超长、过短、数组越界、格式不对。"""
    op = c.by_id(op_id)
    schema = _deref(c.spec, op.spec["requestBody"]["content"]["application/json"]["schema"])
    base, out = BASES[op_id], []
    for name in schema.get("required", []):
        out.append((f"缺 {name}", {k: v for k, v in base.items() if k != name}))
    for name, prop in schema["properties"].items():
        prop = _deref(c.spec, prop)
        out.append((f"{name} 类型不对", {**base, name: _wrong_type(prop)}))
        if "enum" in prop:
            out.append((f"{name} 枚举外取值", {**base, name: "__bogus__"}))
        if "maxLength" in prop:
            out.append((f"{name} 超长", {**base, name: "x" * (prop["maxLength"] + 1)}))
        if prop.get("minLength", 0) > 0:
            out.append((f"{name} 过短", {**base, name: ""}))
        if prop.get("type") == "array" and "minItems" in prop:
            out.append((f"{name} 元素太少", {**base, name: []}))
        if prop.get("type") == "array" and "maxItems" in prop:
            out.append((f"{name} 元素太多", {**base, name: base[name] * (prop["maxItems"] + 1)}))
        if prop.get("format") == "uuid":
            out.append((f"{name} 不是 uuid", {**base, name: "not-a-uuid"}))
        if prop.get("format") == "date":
            out.append((f"{name} 不是日期", {**base, name: "2026-02-30"}))
    return [(f"{op_id}：{d}", b) for d, b in out]


def _decision(**kw):
    return {"decisions": [{"task_id": "t-1", "decision": "approve", **kw}]}


NESTED_INVALID = [
    ("submitReviewDecisions", "task_id 不是字符串", {"decisions": [{"task_id": 1, "decision": "delete"}]}),
    ("submitReviewDecisions", "corrections 不是对象", _decision(corrections="x")),
    ("submitReviewDecisions", "doc_type 为 null", _decision(corrections={"doc_type": None})),
    ("submitReviewDecisions", "doc_type 枚举外", _decision(corrections={"doc_type": "manual"})),
    ("submitReviewDecisions", "valid_from 缺 value", _decision(corrections={"valid_from": {"grade": "explicit"}})),
    ("submitReviewDecisions", "valid_from 不是日期", _decision(corrections={"valid_from": {"value": "2026-13-01"}})),
    ("submitReviewDecisions", "grade 枚举外",
     _decision(corrections={"valid_to": {"value": "2026-01-01", "grade": "guess"}})),
    ("submitReviewDecisions", "relations 不是数组", _decision(corrections={"relations": {"x": 1}})),
    ("submitReviewDecisions", "关系类型枚举外", _decision(corrections={"relations": [
        {"relation": "likes", "to_document_id": str(uuid.uuid4()), "to_version_no": 1}]})),
    ("submitReviewDecisions", "关系缺版本号", _decision(corrections={"relations": [{"to_document_id": str(uuid.uuid4())}]})),
    ("submitReviewDecisions", "关系版本号为 0", _decision(corrections={"relations": [
        {"to_document_id": str(uuid.uuid4()), "to_version_no": 0}]})),
    ("submitReviewDecisions", "关系文档 id 不是 uuid", _decision(corrections={"relations": [
        {"to_document_id": "P002", "to_version_no": 1}]})),
    ("addForcedVerifyRule", "keyword 缺 value", {"kind": "keyword"}),
    ("addForcedVerifyRule", "keyword 的 value 为空", {"kind": "keyword", "value": ""}),
    ("addForcedVerifyRule", "value 超过 100 字", {"kind": "keyword", "value": "x" * 101}),
    ("addForcedVerifyRule", "doc_type 的 value 枚举外", {"kind": "doc_type", "value": "manual"}),
    ("addForcedVerifyRule", "document 的 value 不是 uuid", {"kind": "document", "value": "P002"}),
]

# 文件允许的边界请求：实现不应报格式错误（业务上可以失败）
VALID_EDGES = [
    ("query", {"question": Q, "session_id": None, "as_of_valid": None, "as_of_system": None, "mode_hint": None,
               "stream": False}),
    ("query", {"question": Q, "as_of_valid": "2026-03-15", "as_of_system": "2026-11-01", "mode_hint": "deep"}),
    ("query", {"question": "问" * 2000}),
    ("submitReviewDecisions", {"decisions": [{"task_id": "t-1", "decision": "approve", "corrections": None}]}),
    ("submitReviewDecisions", _decision(corrections={})),
    ("submitReviewDecisions", _decision(corrections={"valid_from": None, "valid_to": None, "relations": None})),
    ("submitReviewDecisions", _decision(corrections={"valid_from": {"value": "2026-01-01", "grade": "inferred"},
                                                     "relations": [{"to_document_id": str(uuid.uuid4()),
                                                                    "to_version_no": 1}]})),
    ("submitReviewDecisions", {"decisions": [{"task_id": f"t-{i}", "decision": "delete"} for i in range(1000)]}),
    ("addForcedVerifyRule", {"kind": "all"}),
    ("addForcedVerifyRule", {"kind": "all", "value": "忽略", "description": "x" * 200}),
    ("addForcedVerifyRule", {"kind": "keyword", "value": "x" * 100, "description": None}),
]

PATHS = {"query": "/v1/query", "submitReviewDecisions": "/v1/review/decisions",
         "addForcedVerifyRule": "/v1/config/forced-verify"}
WHO = {"query": "tok-staff", "submitReviewDecisions": "tok-rv", "addForcedVerifyRule": "tok-admin"}


def _body_errors(c: Contract, op_id: str, body) -> list[str]:
    op = c.by_id(op_id)
    return c.errors(f"{op.ptr}/requestBody/content/{pointer('application/json')[1:]}/schema", body)


def test_requests_the_file_rejects_get_400(env):
    c = env.contract
    cases = [x for op_id in BASES for x in generated_invalid(c, op_id)] + \
            [(f"{op_id}：{d}", b) for op_id, d, b in NESTED_INVALID]
    assert len(cases) >= 40
    for desc, body in cases:
        op_id = desc.split("：")[0]
        assert _body_errors(c, op_id, body), f"{desc}：测试用例本身应当不符合文件"
        r = env.client.post(PATHS[op_id], headers=H(WHO[op_id]), json=body)
        assert r.status_code == 400 and r.json()["code"] == "invalid_request", f"{desc}：{r.status_code} {r.text[:200]}"


def test_requests_the_file_allows_are_not_format_errors(env):
    c = env.contract
    for op_id, body in [(k, v) for k, v in BASES.items()] + VALID_EDGES:
        assert not _body_errors(c, op_id, body), (op_id, body)
        r = env.client.post(PATHS[op_id], headers=H(WHO[op_id]), json=copy.deepcopy(body))
        assert r.status_code < 400, f"{op_id} {json.dumps(body, ensure_ascii=False)[:120]}：{r.status_code} {r.text[:200]}"
