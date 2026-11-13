"""AC-10-4 缓存隔离（硬性：无串用）与 AC-10-10 Trace 清理（开发文档 10.1 第 7 步、10.2）。离线：脚本化模型、内存索引，
文档经真实的上传、审核、入库与索引流程，经提问接口提问。

- 权限不同的两个用户连续问同一问题，各自得到按自己权限算出的回答，谁先问都一样；权限相同的用户才复用；
- 权限变更、归档、取消归档、有效期更正、租户配置变化都由数据库清除该租户的缓存；
- 即使缓存没被清除（触发器被停用、或键出错把别人的回答放到自己名下），命中后按当前授权范围核对所引证据，
  不通过即作废重算，不会交出无权内容；
- 键随租户、授权范围、索引构建、提问日、时间参数、模式提示、模型与 Deep 开关变化，principals 顺序与问题空白不影响；
- Trace 清理：已结束任务的检查点与任务行超过租户保留天数即删除；执行中的不动；query_log 保留；过期缓存删除。
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from eaer.adapters.authorizer import LocalAuthorizer
from eaer.api.app import TokenAuthenticator, create_app
from eaer.domain.identity import Identity, Scope
from eaer.ingestion.indexer import Indexer
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.lifecycle.service import Lifecycle
from eaer.orchestrator import cache
from eaer.orchestrator.engine import Engine
from eaer.orchestrator.tasks import purge
from eaer.retrieval.embed import HashingEmbedder
from eaer.retrieval.index import MemoryIndex
from eaer.retrieval.service import Retriever
from eaer.storage import LocalFileStore
from test_engine import Scripted, ids_with

NOW = datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc)
TOKENS = {"tok-ed": Identity("ed", "t1", frozenset({"editor"})),
          "tok-rv": Identity("rv", "t1", frozenset({"reviewer"})),
          "tok-fin": Identity("fin", "t1", frozenset()),
          "tok-staff": Identity("staff", "t1", frozenset()),
          "tok-staff2": Identity("staff2", "t1", frozenset())}
ACCOUNTS = {"fin": {"tenant": "t1", "principals": ["R_STAFF", "R_FIN"]},
            "staff": {"tenant": "t1", "principals": ["R_STAFF"]},
            "staff2": {"tenant": "t1", "principals": ["R_STAFF"]}}
SECRET_Q, PUBLIC_Q = "高管年薪上限是多少", "一线城市住宿上限是多少"


def answer(p):
    """按证据作答：问高管薪酬就找 300 万元那条，问住宿就找 650 元那条；提示里没有（无权或已归档）就拒答。"""
    question = p.split("问题：")[1].split("\n")[0]
    needle, text = (("300万元", "高管年薪上限为300万元。") if "高管" in question
                    else ("650元", "一线城市住宿上限为每晚650元。"))
    ids = ids_with(p, needle)
    if ids:
        return {"abstain": False, "claims": [{"text": text, "evidence_ids": ids[:1], "core": True}]}
    return {"abstain": True, "missing": "没有相关规定", "claims": []}


def H(t):
    return {"Authorization": f"Bearer {t}"}


@pytest.fixture
def cq(db_url, tmp_path):
    store, index, emb = LocalFileStore(tmp_path / "files"), MemoryIndex(), HashingEmbedder()
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    st = {"model": Scripted(answer), "budgets": {}}

    def engine(c, tenant):
        return Engine(c, Retriever(c, index, emb), st["model"], LocalAuthorizer(c, ACCOUNTS), store=store,
                      budgets=st["budgets"])

    client = TestClient(create_app(db_url, store, TokenAuthenticator(TOKENS), lambda: NOW,
                                   authorizer=lambda c: LocalAuthorizer(c, ACCOUNTS), engine=engine, owner="test"))
    ingest = Worker(Ingestion(conn, store, lambda: NOW), Indexer(conn, index, emb, "b1"))
    docs = {}
    for name, principals, body in (
            ("pub", "R_STAFF", "<h1>差旅报销管理办法</h1><p>第一条　一线城市住宿上限为每晚650元。</p>"
                               "<p>第二条　本办法自2026年1月1日起施行。</p>"),
            ("fin", "R_FIN", "<h1>高管薪酬管理办法</h1><p>第一条　高管年薪上限为300万元。</p>"
                             "<p>第二条　本办法自2026年1月1日起施行。</p>")):
        up = client.post("/v1/documents", headers=H("tok-ed"), data={"principals": principals},
                         files={"file": (f"{name}.html", body.encode(), "text/html")}).json()
        docs[name] = up["document_id"]
        ingest.run_until_idle()
        task = [t for t in client.get("/v1/review/tasks", headers=H("tok-rv")).json()["items"]
                if t["status"] == "open"][0]["task_id"]
        client.post("/v1/review/decisions", headers=H("tok-rv"), json={"decisions": [
            {"task_id": task, "decision": "approve", "corrections": {"doc_type": "normative"}}]})
        ingest.run_until_idle()
    ns = type("NS", (), {})()
    ns.__dict__.update(client=client, conn=conn, st=st, docs=docs, ingest=ingest,
                       lc=Lifecycle(conn, lambda: NOW), engine=engine)
    yield ns
    conn.close()


def ask(cq, token, q):
    r = cq.client.post("/v1/query", headers=H(token), json={"question": q})
    assert r.status_code == 200, r.text
    return r.json()


def drafts(cq):
    return cq.st["model"].calls.count("draft")


def last_log(cq, user):
    return cq.conn.execute("SELECT * FROM query_log WHERE user_id = %s ORDER BY asked_at DESC, ctid DESC LIMIT 1",
                           (user,)).fetchone()


# ---------- AC-10-4（硬性）：权限不同的用户不串用 ----------

@pytest.mark.parametrize("first", ["fin", "staff"])
def test_users_with_different_permissions_never_share_answers(cq, first):
    order = ["tok-fin", "tok-staff"] if first == "fin" else ["tok-staff", "tok-fin"]
    got = {t: ask(cq, t, SECRET_Q) for t in order}
    assert "300万元" in got["tok-fin"]["answer"] and not got["tok-fin"]["abstained"]
    assert "300万" not in got["tok-staff"]["answer"] and got["tok-staff"]["abstained"]   # 无串用
    assert drafts(cq) == 2                                         # 两人各自算了一次，没有互相命中
    assert cq.conn.execute("SELECT count(DISTINCT scope_hash) AS n FROM query_cache").fetchone()["n"] == 2
    for t in order:                                                # 各自再问：命中自己的缓存
        again = ask(cq, t, SECRET_Q)
        assert again["answer"] == got[t]["answer"] and again["session_id"] != got[t]["session_id"]
    assert drafts(cq) == 2


def test_same_permissions_reuse_and_the_hit_is_logged(cq):
    a = ask(cq, "tok-staff", PUBLIC_Q)
    b = ask(cq, "tok-staff2", PUBLIC_Q)
    assert a["answer"] == b["answer"] and "650元" in b["answer"] and drafts(cq) == 1
    log = last_log(cq, "staff2")
    assert log["status"] == "completed" and log["cost"]["cache"]["source_query"] and log["cost"]["retrieval_rounds"] == 0
    assert log["evidence_ids"] and log["authz_decision"]["principals"] == ["R_STAFF"]


def test_permission_change_archive_and_unarchive_clear_the_cache(cq):
    assert "300万元" in ask(cq, "tok-fin", SECRET_Q)["answer"]
    cq.lc.change_acl(uuid.UUID(cq.docs["fin"]), ["R_HR"], "admin")           # 撤销 R_FIN
    cq.ingest.run_until_idle()
    assert cq.conn.execute("SELECT count(*) AS n FROM query_cache").fetchone()["n"] == 0
    assert "300万" not in ask(cq, "tok-fin", SECRET_Q)["answer"]

    assert "650元" in ask(cq, "tok-staff", PUBLIC_Q)["answer"]
    assert cq.client.post(f"/v1/documents/{cq.docs['pub']}/archive", headers=H("tok-ed")).status_code == 200
    cq.ingest.run_until_idle()
    assert ask(cq, "tok-staff", PUBLIC_Q)["abstained"]                        # 归档后不再用缓存里的回答
    cq.client.post(f"/v1/documents/{cq.docs['pub']}/unarchive", headers=H("tok-ed"))
    cq.ingest.run_until_idle()
    assert "650元" in ask(cq, "tok-staff", PUBLIC_Q)["answer"]                 # 取消归档后也不用旧的拒答


def test_validity_correction_and_tenant_config_clear_the_cache(cq):
    ask(cq, "tok-staff", PUBLIC_Q)
    vid = cq.conn.execute("SELECT id FROM document_version WHERE document_id = %s", (cq.docs["pub"],)).fetchone()["id"]
    cq.lc.correct_validity(vid, "admin", valid_from=date(2026, 2, 1))
    assert cq.conn.execute("SELECT count(*) AS n FROM query_cache").fetchone()["n"] == 0
    ask(cq, "tok-staff", PUBLIC_Q)
    assert cq.conn.execute("SELECT count(*) AS n FROM query_cache").fetchone()["n"] == 1
    cq.conn.execute("INSERT INTO tenant_config (tenant_id, updated_at) VALUES ('t1', now()) "
                    "ON CONFLICT (tenant_id) DO UPDATE SET thresholds = '{\"rerank_min_top\": 0.5}'")
    assert cq.conn.execute("SELECT count(*) AS n FROM query_cache").fetchone()["n"] == 0


def test_stale_or_misfiled_entries_are_rechecked_before_return(cq):
    """失效漏掉（触发器被停用）或键出错（别人的回答落在自己名下）时，命中后的核对仍挡住无权内容。"""
    assert "300万元" in ask(cq, "tok-fin", SECRET_Q)["answer"]
    staff_scope = LocalAuthorizer(cq.conn, ACCOUNTS).resolve(TOKENS["tok-staff"])
    eng = cq.engine(cq.conn, "t1")
    k = cache.key("t1", staff_scope, eng._build_id(), SECRET_Q, today=NOW.date(), as_of_valid=None, as_of_system=None,
                  mode_hint=None, model="unknown", deep_enabled=True)
    cq.conn.execute("UPDATE query_cache SET key = %s, scope_hash = %s", (k, cache.scope_hash(staff_scope)))
    b = ask(cq, "tok-staff", SECRET_Q)
    assert "300万" not in b["answer"] and b["abstained"] and drafts(cq) == 2      # 作废重算

    assert "300万元" in ask(cq, "tok-fin", SECRET_Q)["answer"]
    cq.conn.execute("ALTER TABLE document_version DISABLE TRIGGER cache_on_version")
    cq.lc.change_acl(uuid.UUID(cq.docs["fin"]), ["R_HR"], "admin")
    cq.conn.execute("ALTER TABLE document_version ENABLE TRIGGER cache_on_version")
    assert cq.conn.execute("SELECT count(*) AS n FROM query_cache").fetchone()["n"] >= 1   # 缓存没被清掉
    assert "300万" not in ask(cq, "tok-fin", SECRET_Q)["answer"]


def test_cache_can_be_turned_off(cq):
    cq.st["budgets"] = {"cache_ttl_seconds": 0}
    ask(cq, "tok-staff", PUBLIC_Q)
    ask(cq, "tok-staff", PUBLIC_Q)
    assert drafts(cq) == 2 and cq.conn.execute("SELECT count(*) AS n FROM query_cache").fetchone()["n"] == 0


def test_cache_key_parts():
    s1, s2 = Scope("t1", frozenset({"A", "B"})), Scope("t1", frozenset({"B", "A"}))
    base = dict(today=date(2026, 11, 11), as_of_valid=None, as_of_system=None, mode_hint=None, model="m",
                deep_enabled=True)
    k = cache.key("t1", s1, "eaer-b1", "住宿 上限", **base)
    assert k == cache.key("t1", s2, "eaer-b1", " 住宿   上限 ", **base)          # principals 顺序、空白不影响
    variants = [cache.key("t2", s1, "eaer-b1", "住宿 上限", **base),
                cache.key("t1", Scope("t1", frozenset({"A"})), "eaer-b1", "住宿 上限", **base),
                cache.key("t1", s1, "eaer-b2", "住宿 上限", **base),
                cache.key("t1", s1, "eaer-b1", "住宿上限", **base)]
    for field, value in (("today", date(2026, 11, 12)), ("as_of_valid", date(2026, 3, 1)),
                         ("as_of_system", date(2026, 3, 1)), ("mode_hint", "verified"), ("model", "m2"),
                         ("deep_enabled", False)):
        variants.append(cache.key("t1", s1, "eaer-b1", "住宿 上限", **{**base, field: value}))
    assert k not in variants and len(set(variants)) == len(variants)


# ---------- AC-10-10：Trace 清理 ----------

def test_purge_deletes_traces_older_than_the_retention(cq):
    for token in ("tok-staff", "tok-fin"):
        ask(cq, token, PUBLIC_Q)
    old, recent = [r["id"] for r in cq.conn.execute("SELECT id FROM query_task ORDER BY created_at")]
    # 已结束的任务不可修改（不变量），直接调整时间用于测试：先停用守卫
    cq.conn.execute("ALTER TABLE query_task DISABLE TRIGGER query_task_guard")
    cq.conn.execute("UPDATE query_task SET updated_at = %s WHERE id = %s", (NOW - timedelta(days=31), old))
    cq.conn.execute("UPDATE query_task SET updated_at = %s WHERE id = %s", (NOW - timedelta(days=5), recent))
    cq.conn.execute("ALTER TABLE query_task ENABLE TRIGGER query_task_guard")
    running = uuid.uuid4()                                         # 执行中的任务再旧也不动
    cq.conn.execute("INSERT INTO query_log (id, tenant_id, session_id, user_id, asked_at, question, mode, mode_path, "
                    "status, authz_decision, model_id, index_build_id) VALUES (%s, 't1', %s, 'staff', %s, 'q', 'fast', "
                    "'{fast}', 'running', '{}', 'm', 'b')", (running, uuid.uuid4(), NOW - timedelta(days=60)))
    cq.conn.execute("INSERT INTO query_task (id, tenant_id, session_id, user_id, identity, state, status, locked_by, "
                    "locked_until, created_at, updated_at) VALUES (%s, 't1', %s, 'staff', '{}', '{}', 'running', 'w', %s, "
                    "%s, %s)", (running, uuid.uuid4(), NOW, NOW - timedelta(days=60), NOW - timedelta(days=60)))
    cp = lambda qid: cq.conn.execute("SELECT count(*) AS n FROM checkpoints WHERE thread_id = %s",  # noqa: E731
                                     (str(qid),)).fetchone()["n"]
    assert cp(old) > 0 and cp(recent) > 0
    out = purge(cq.conn, NOW + timedelta(hours=1))
    assert out["tasks"] == 1 and out["checkpoints"] > 0 and out["cache"] == 2   # 缓存 600 秒，已过期
    assert cp(old) == 0 and cp(recent) > 0
    assert {r["id"] for r in cq.conn.execute("SELECT id FROM query_task")} == {recent, running}
    for table in ("checkpoint_blobs", "checkpoint_writes"):
        assert cq.conn.execute(f"SELECT count(*) AS n FROM {table} WHERE thread_id = %s", (str(old),)).fetchone()["n"] == 0
    assert cq.conn.execute("SELECT count(*) AS n FROM query_log").fetchone()["n"] == 3     # 审计记录保留
    cq.conn.execute("INSERT INTO tenant_config (tenant_id, trace_retention_days, updated_at) VALUES ('t1', 3, now()) "
                    "ON CONFLICT (tenant_id) DO UPDATE SET trace_retention_days = 3")
    assert purge(cq.conn, NOW)["tasks"] == 1 and cp(recent) == 0                # 租户保留期 3 天
    assert purge(cq.conn, NOW)["tasks"] == 0


def test_purge_command(cq, monkeypatch, capsys, db_url):
    from eaer.ingestion.__main__ import main

    monkeypatch.setenv("EAER_DATABASE_URL", db_url)
    assert main(["purge"]) == 0 and "已删除" in capsys.readouterr().out
