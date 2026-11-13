"""AC-4-2：每条不变量的违规操作都被数据库拒绝。AC-4-4（数据库部分）：删除待审核版本后无残留。

这些用例绕过生命周期服务直接写 SQL，模拟应用代码出错时数据库这道防线是否有效。
"""
from __future__ import annotations

import uuid
from datetime import date

import psycopg
import pytest
from psycopg import errors

from dbutil import seed
from eaer.lifecycle.service import InvalidState


@pytest.fixture
def s(conn):
    lc, clock, ids = seed(conn)
    return conn, lc, clock, ids


def current_validity_id(conn, vid):
    return conn.execute("SELECT id FROM version_validity WHERE version_id = %s AND replaced_at IS NULL "
                        "AND recorded_at IS NOT NULL", (vid,)).fetchone()["id"]


VIOLATIONS = [
    ("删除可搜索版本", "DELETE FROM document_version WHERE id = %(v1)s",
     errors.RaiseException, "只有待审核的版本可以删除"),
    ("改写 approved_at", "UPDATE document_version SET approved_at = now() WHERE id = %(v1)s",
     errors.RaiseException, "approved_at 只能写一次"),
    ("可搜索退回待审核", "UPDATE document_version SET status = 'pending_review', approved_at = NULL WHERE id = %(v1)s",
     errors.RaiseException, "approved_at 只能写一次"),
    ("可搜索直接退回待审核（不动 approved_at）", "UPDATE document_version SET status = 'pending_review' WHERE id = %(v1)s",
     errors.RaiseException, "不允许的状态转换"),
    ("待审核直接归档", "UPDATE document_version SET status = 'archived', archived_at = now() WHERE id = %(v2)s",
     errors.RaiseException, "不允许的状态转换"),
    ("归档但不写归档时间", "UPDATE document_version SET status = 'archived' WHERE id = %(v1)s",
     errors.CheckViolation, "archived_iff_archived_at"),
    ("修改已发布版本的内容哈希", "UPDATE document_version SET content_hash = 'sha256:x' WHERE id = %(v1)s",
     errors.RaiseException, "不可修改"),
    ("修改已记录的有效期", "UPDATE version_validity SET valid_from = '2024-01-01' WHERE id = %(cur)s",
     errors.RaiseException, "只能写入 replaced_at"),
    ("删除已记录的有效期", "DELETE FROM version_validity WHERE id = %(cur)s",
     errors.RaiseException, "不可删除"),
    ("同一版本写入第二行当前有效期",
     "INSERT INTO version_validity (id, version_id, valid_from_grade, valid_to_grade, recorded_at) "
     "VALUES (gen_random_uuid(), %(v1)s, 'unknown', 'unknown', now())",
     errors.UniqueViolation, "version_validity_one_current"),
    ("修改生命周期事件", "UPDATE lifecycle_event SET actor = 'mallory'",
     errors.RaiseException, "只能追加"),
    ("删除生命周期事件", "DELETE FROM lifecycle_event",
     errors.RaiseException, "只能追加"),
]


@pytest.mark.parametrize("name, sql, exc, message", VIOLATIONS, ids=[v[0] for v in VIOLATIONS])
def test_violation_is_rejected(s, name, sql, exc, message):
    conn, _, _, ids = s
    params = {**ids, "cur": current_validity_id(conn, ids["v1"])}
    with pytest.raises(exc) as e:
        conn.execute(sql, params)
    assert message in str(e.value)


def test_recorded_relation_cannot_change(s):
    conn, lc, clock, ids = s
    clock.set(date(2025, 12, 18))
    lc.publish(ids["v2"], "alice")
    with pytest.raises(errors.RaiseException, match="只能写入 retracted_at"):
        conn.execute("UPDATE version_relation SET grade = 'inferred'")
    # 撤销关系是允许的
    conn.execute("UPDATE version_relation SET retracted_at = now()")


def test_replacing_current_validity_is_allowed(s):
    conn, _, _, ids = s
    conn.execute("UPDATE version_validity SET replaced_at = now() WHERE id = %s",
                 (current_validity_id(conn, ids["v1"]),))


def test_service_refuses_before_database(s):
    _, lc, _, ids = s
    with pytest.raises(InvalidState):
        lc.delete_pending(ids["v1"], "alice")
    with pytest.raises(InvalidState):
        lc.publish(ids["v1"], "alice")


# ---------- AC-4-4（数据库部分） ----------

def _count(conn, table, column, vid):
    return conn.execute(f"SELECT count(*) AS n FROM {table} WHERE {column} = %s", (vid,)).fetchone()["n"]


def test_delete_pending_leaves_no_residue(s):
    conn, lc, _, ids = s
    v2 = ids["v2"]
    assert _count(conn, "evidence", "version_id", v2) == 1
    uri = lc.delete_pending(v2, "bob")
    assert uri == "file://v2"   # 调用方据此删除文件存储中的原件
    for table, column in [("document_version", "id"), ("evidence", "version_id"),
                          ("version_validity", "version_id"), ("version_relation", "from_version_id"),
                          ("ingest_job", "version_id"), ("review_task", "version_id")]:
        assert _count(conn, table, column, v2) == 0, table
    # 审核记录保留谁在何时删除，但不指向版本、不保留内容
    task = conn.execute("SELECT * FROM review_task WHERE status = 'deleted'").fetchone()
    assert task["reviewer"] == "bob" and task["version_id"] is None and task["corrections"] is None
    # 文档还有已发布的 v1，所以文档保留
    assert _count(conn, "document", "id", ids["doc"]) == 1


def test_deleting_only_version_removes_document(conn):
    from dbutil import Clock
    from eaer.lifecycle.service import Lifecycle
    lc = Lifecycle(conn, Clock())
    lc.ensure_tenant("t1")
    acl = lc.create_acl_policy("t1", ["R"])
    doc = lc.create_document("t1", "X", "临时文件")
    v = lc.add_version(doc, content_hash="h", raw_object_uri="file://x", acl_policy_id=acl, parser_version="t")
    lc.delete_pending(v, "bob")
    assert _count(conn, "document", "id", doc) == 0


# ---------- AC-11-5（数据库部分）：强制验证清单只增不减 ----------

FV_VIOLATIONS = [
    ("清空规则", "UPDATE tenant_config SET forced_verify_rules = '[]' WHERE tenant_id = 't1'"),
    ("删掉一条规则", "UPDATE tenant_config SET forced_verify_rules = forced_verify_rules - 0 WHERE tenant_id = 't1'"),
    ("修改已有规则", "UPDATE tenant_config SET forced_verify_rules = jsonb_set(forced_verify_rules, '{0,value}', "
                   "'\"差旅\"') WHERE tenant_id = 't1'"),
    ("调换规则顺序", "UPDATE tenant_config SET forced_verify_rules = jsonb_build_array(forced_verify_rules -> 1, "
                   "forced_verify_rules -> 0) WHERE tenant_id = 't1'"),
    ("改成非数组", "UPDATE tenant_config SET forced_verify_rules = '{}' WHERE tenant_id = 't1'"),
    ("删除带规则的租户配置", "DELETE FROM tenant_config WHERE tenant_id = 't1'"),
]


@pytest.fixture
def fv_rules(s):
    from datetime import datetime, timezone

    from eaer.orchestrator.forced_verify import add_rule
    conn = s[0]
    now = datetime(2026, 11, 12, tzinfo=timezone.utc)
    add_rule(conn, "t1", "keyword", "报销", "admin", now)
    add_rule(conn, "t1", "doc_type", "record", "admin", now)
    return conn


@pytest.mark.parametrize("name, sql", FV_VIOLATIONS, ids=[v[0] for v in FV_VIOLATIONS])
def test_forced_verify_shrink_is_rejected(fv_rules, name, sql):
    before = fv_rules.execute("SELECT forced_verify_rules FROM tenant_config WHERE tenant_id = 't1'").fetchone()
    with pytest.raises(errors.RaiseException, match="invariant: 强制验证清单"):
        fv_rules.execute(sql)
    after = fv_rules.execute("SELECT forced_verify_rules FROM tenant_config WHERE tenant_id = 't1'").fetchone()
    assert after == before


def test_forced_verify_append_and_empty_tenant_delete_are_allowed(fv_rules):
    fv_rules.execute("UPDATE tenant_config SET forced_verify_rules = forced_verify_rules || "
                     "'[{\"id\": \"fv_x\", \"kind\": \"all\", \"value\": null}]' WHERE tenant_id = 't1'")
    assert len(fv_rules.execute("SELECT forced_verify_rules FROM tenant_config WHERE tenant_id = 't1'")
               .fetchone()["forced_verify_rules"]) == 3
    fv_rules.execute("INSERT INTO tenant_config (tenant_id, updated_at) VALUES ('t-empty', now())")
    fv_rules.execute("DELETE FROM tenant_config WHERE tenant_id = 't-empty'")     # 没有规则的配置可以删除


# ---------- AC-8-6（数据库部分）：提问记录与查询任务 ----------

@pytest.fixture
def qt(conn):
    """一条运行中的提问记录与任务、一条已完成的提问记录与任务。"""
    ids = {"run": uuid.uuid4(), "done": uuid.uuid4(), "sid": uuid.uuid4()}
    for key, status in (("run", "running"), ("done", "completed")):
        conn.execute(
            "INSERT INTO query_log (id, tenant_id, session_id, user_id, asked_at, question, mode, mode_path, status, "
            "authz_decision, model_id, index_build_id) VALUES (%s, 't1', %s, 'alice', now(), '问题', 'fast', '{fast}', "
            "%s, '{}', 'm', 'b1')", (ids[key], ids["sid"], status))
        lease = "now() + interval '1 minute'" if status == "running" else "NULL"
        owner = "'api-1'" if status == "running" else "NULL"
        conn.execute(
            "INSERT INTO query_task (id, tenant_id, session_id, user_id, identity, state, status, locked_by, "
            f"locked_until, created_at, updated_at) VALUES (%s, 't1', %s, 'alice', '{{}}', '{{}}', %s, {owner}, {lease}, "
            "now(), now())", (ids[key], ids["sid"], status))
    return conn, ids


QT_VIOLATIONS = [
    ("改写已完成的提问记录", "UPDATE query_log SET answer = '改过' WHERE id = %(done)s", "已结束的提问记录不可修改"),
    ("已完成的提问记录退回运行中", "UPDATE query_log SET status = 'running' WHERE id = %(done)s", "已结束的提问记录不可修改"),
    ("改写提问人", "UPDATE query_log SET user_id = 'bob' WHERE id = %(run)s", "提问人、会话、问题与时间不可修改"),
    ("改写问题", "UPDATE query_log SET question = '别的问题' WHERE id = %(run)s", "提问人、会话、问题与时间不可修改"),
    ("已完成的任务重新排队", "UPDATE query_task SET status = 'queued' WHERE id = %(done)s", "已结束的查询任务不可修改"),
    ("改写任务的提问人", "UPDATE query_task SET user_id = 'bob' WHERE id = %(run)s", "查询任务的提问人与会话不可修改"),
    ("改写任务的身份", "UPDATE query_task SET identity = '{\"roles\": [\"admin\"]}' WHERE id = %(run)s",
     "查询任务的提问人与会话不可修改"),
]


@pytest.mark.parametrize("name, sql, message", QT_VIOLATIONS, ids=[v[0] for v in QT_VIOLATIONS])
def test_query_log_and_task_violations_are_rejected(qt, name, sql, message):
    conn, ids = qt
    with pytest.raises(errors.RaiseException, match=f"invariant: .*{message}"):
        conn.execute(sql, ids)


def test_query_task_lease_must_match_status(qt):
    conn, ids = qt
    with pytest.raises(errors.CheckViolation):            # 运行中没有租约
        conn.execute("UPDATE query_task SET locked_until = NULL, locked_by = NULL WHERE id = %(run)s", ids)
    with pytest.raises(errors.CheckViolation):            # 排队中还留着租约
        conn.execute("UPDATE query_task SET status = 'queued' WHERE id = %(run)s", ids)
    conn.execute("UPDATE query_task SET status = 'queued', locked_until = NULL, locked_by = NULL WHERE id = %(run)s", ids)
    conn.execute("UPDATE query_log SET status = 'cancelled' WHERE id = %(run)s", ids)       # 运行中 → 已取消：允许
