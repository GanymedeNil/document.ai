"""强制验证清单（开发文档 8.1、11.2）。

AC-11-5（硬性）：不存在删除或缩小的路径；增加后立即生效。
- 接口只有读取与追加：路由表里没有别的方法，DELETE、PUT、PATCH 返回 405；
- 代码里写 forced_verify_rules 的 SQL 只有追加一种（静态检查）；
- 数据库触发器拒绝缩小（见 tests/test_db_invariants.py）；
- 追加后下一次判断立即命中，不经缓存。
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from dbutil import Clock
from eaer.api.app import TokenAuthenticator, create_app
from eaer.domain.identity import Identity
from eaer.orchestrator.forced_verify import BUILTIN, Cited, add_rule, mentions_time_or_version, must_verify
from eaer.storage import LocalFileStore

TOKENS = {"tok-admin": Identity("ad", "t1", frozenset({"admin"})),
          "tok-ed": Identity("ed", "t1", frozenset({"editor"})),
          "tok-other": Identity("eve", "t2", frozenset({"admin", "editor"}))}
NOW = datetime(2026, 11, 12, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def H(t):
    return {"Authorization": f"Bearer {t}"}


@pytest.fixture
def api(db_url, tmp_path):
    app = create_app(db_url, LocalFileStore(tmp_path / "f"), TokenAuthenticator(TOKENS), Clock())
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    yield TestClient(app), app, conn
    conn.close()


def test_read_and_append_through_the_api(api):
    client, _, _ = api
    r = client.get("/v1/config/forced-verify", headers=H("tok-ed"))
    assert r.status_code == 200 and r.json() == {"builtin": BUILTIN, "rules": []}
    assert client.post("/v1/config/forced-verify", headers=H("tok-ed"),
                       json={"kind": "keyword", "value": "报销"}).status_code == 403          # 只有管理员能追加
    r = client.post("/v1/config/forced-verify", headers=H("tok-admin"),
                    json={"kind": "keyword", "value": "报销", "description": "报销类问题一律验证"})
    assert r.status_code == 201 and r.json()["created_by"] == "ad"
    again = client.post("/v1/config/forced-verify", headers=H("tok-admin"), json={"kind": "keyword", "value": "报销"})
    assert again.status_code == 200 and again.json()["id"] == r.json()["id"]                # 重复追加不产生新规则
    rules = client.get("/v1/config/forced-verify", headers=H("tok-ed")).json()["rules"]
    assert [x["id"] for x in rules] == [r.json()["id"]]
    assert client.get("/v1/config/forced-verify", headers=H("tok-other")).json()["rules"] == []   # 租户隔离


@pytest.mark.parametrize("body", [
    {"kind": "regex", "value": ".*"}, {"kind": "keyword", "value": "  "}, {"kind": "keyword"},
    {"kind": "doc_type", "value": "memo"}, {"kind": "document", "value": "not-a-uuid"},
    {"kind": "document", "value": str(uuid.uuid4())},
])
def test_invalid_rules_are_rejected(api, body):
    client, _, _ = api
    r = client.post("/v1/config/forced-verify", headers=H("tok-admin"), json=body)
    assert r.status_code == 400 and r.json()["code"] == "invalid_request"


def test_no_route_can_delete_or_shrink(api):
    client, app, _ = api
    methods = {m for route in app.routes if "forced-verify" in getattr(route, "path", "")
               for m in route.methods}
    assert methods <= {"GET", "POST", "HEAD"}, methods
    for m in ("delete", "put", "patch"):
        assert getattr(client, m)("/v1/config/forced-verify", headers=H("tok-admin")).status_code == 405
    assert client.delete("/v1/config/forced-verify/fv_1", headers=H("tok-admin")).status_code == 404


def test_code_only_ever_appends_to_the_rules():
    """静态检查：eaer/ 中写 forced_verify_rules 的 SQL 只有“forced_verify_rules = forced_verify_rules || …”。"""
    writes = []
    for p in (ROOT / "eaer").rglob("*.py"):
        if "migrations" in p.parts:
            continue
        for m in re.finditer(r"forced_verify_rules\s*=\s*([^,\n]+)", p.read_text(encoding="utf-8")):
            writes.append((p.name, m.group(1).strip()))
    assert writes and all(w.startswith("forced_verify_rules ||") for _, w in writes), writes


def test_appended_rule_takes_effect_on_the_next_question(api):
    client, _, conn = api
    q = "差旅报销的标准是多少"
    assert must_verify(conn, "t1", question=q) == []
    rid = client.post("/v1/config/forced-verify", headers=H("tok-admin"),
                      json={"kind": "keyword", "value": "报销"}).json()["id"]
    assert must_verify(conn, "t1", question=q) == [rid]                     # 同一连接，下一次判断即命中
    assert must_verify(conn, "t2", question=q) == []


def test_builtin_rules_and_tenant_rule_kinds(conn):
    conn.execute("INSERT INTO tenant_config (tenant_id, updated_at) VALUES ('t1', now())")
    conn.execute("INSERT INTO acl_policy (id, tenant_id, principals, source, updated_at) "
                 "VALUES (gen_random_uuid(), 't1', '{R}', 'reference', now())")
    did = conn.execute("INSERT INTO document (id, tenant_id, source_ref, title, created_at) "
                       "VALUES (gen_random_uuid(), 't1', 'X', '采购规定', now()) RETURNING id").fetchone()["id"]
    plain = "伙食补贴是多少"
    assert must_verify(conn, "t1", question=plain) == []
    assert must_verify(conn, "t1", question=plain, time_params=True) == ["builtin-time-version"]
    assert must_verify(conn, "t1", question=plain, cited=[Cited("d", "normative")]) == ["builtin-normative"]
    rec = add_rule(conn, "t1", "doc_type", "record", "ad", NOW)[0]["id"]
    doc = add_rule(conn, "t1", "document", str(did), "ad", NOW)[0]["id"]
    assert must_verify(conn, "t1", question=plain, cited=[Cited("d", "record")]) == [rec]
    assert must_verify(conn, "t1", question=plain, cited=[Cited(str(did), None)]) == [doc]
    every = add_rule(conn, "t1", "all", "忽略的值", "ad", NOW)[0]
    assert every["value"] is None and must_verify(conn, "t1", question=plain) == [every["id"]]


@pytest.mark.parametrize("q, hit", [
    ("截至2026年3月1日住宿上限是多少", True), ("2025-12-01 当时执行的标准", True), ("修订前的规定怎么写", True),
    ("旧版和新版有什么区别", True), ("第2版的采购上限", True), ("第三版是否还有效", True), ("3月1日起的标准", True),
    ("一线城市住宿上限是多少", False), ("伙食补贴标准", False),
    ("2026 年第二季度的差旅总费用是多少？", False), ("2026 年 7 月修订前后有什么变化", True),
])
def test_time_and_version_wording(q, hit):
    assert mentions_time_or_version(q) is hit
