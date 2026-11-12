"""审核（开发文档 6.3、AC-6-5）。"""
from __future__ import annotations

import uuid
from datetime import date

import pytest

from test_ingestion import draft, env, html  # noqa: F401  复用夹具


def task_of(conn, vid):
    return str(conn.execute("SELECT id FROM review_task WHERE version_id = %s", (vid,)).fetchone()["id"])


def status_of(conn, vid):
    row = conn.execute("SELECT status FROM document_version WHERE id = %s", (vid,)).fetchone()
    return row["status"] if row else None


def current(conn, vid):
    return conn.execute("SELECT * FROM version_validity WHERE version_id = %s AND recorded_at IS NOT NULL "
                        "AND replaced_at IS NULL", (vid,)).fetchone()


def test_batch_of_500_with_5_already_deleted(env):
    batch = uuid.uuid4()
    vids = []
    for i in range(500):
        r = env.ing.receive("t1", html(f"报销规定{i}", f"第一条　第{i}号规定。", "本规定自2026年1月1日起施行。"),
                            f"doc{i}.html", acl_policy_id=env.acl, batch_id=batch)
        vids.append(r.version_id)
    env.worker.run_until_idle()
    tasks = [task_of(env.conn, v) for v in vids]
    gone = tasks[100:105]
    assert all(r["ok"] for r in env.review.decide("t1", [{"task_id": t, "decision": "delete"} for t in gone], "bob"))

    results = env.review.decide("t1", [{"task_id": t, "decision": "approve"} for t in tasks], "alice")
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    assert len(ok) == 495 and len(bad) == 5
    assert {r["task_id"] for r in bad} == set(gone) and all("已处理" in r["error"] for r in bad)
    assert sum(status_of(env.conn, v) == "searchable" for v in vids) == 495
    items, nxt = env.review.list_tasks("t1", status="open", batch_id=batch)
    assert items == [] and nxt is None


def test_list_tasks_shows_suggestions_and_pages(env):
    for i in range(3):
        env.ing.receive("t1", html(f"报销规定{i}", "本规定自2026年1月1日起施行。"), f"d{i}.html", acl_policy_id=env.acl)
    env.worker.run_until_idle()
    first, nxt = env.review.list_tasks("t1", limit=2)
    rest, end = env.review.list_tasks("t1", limit=2, cursor=nxt)
    assert len(first) == 2 and len(rest) == 1 and end is None
    item = first[0]
    assert item["extraction_status"] == "done" and item["suggestions"]["doc_type"] == "normative"
    assert item["draft_validity"]["valid_from"] == "2026-01-01"
    assert env.review.list_tasks("other-tenant")[0] == []


def test_reviewer_supplied_date_is_explicit_and_attributed(env):
    r = env.ing.receive("t1", html("采购审批权限规定", "部门经理单笔采购审批上限为50,000元。"), "p.html", acl_policy_id=env.acl)
    env.worker.run_until_idle()
    assert draft(env.conn, r.version_id)["valid_from_grade"] == "inferred"
    res = env.review.decide("t1", [{"task_id": task_of(env.conn, r.version_id), "decision": "approve",
                                    "corrections": {"valid_from": {"value": "2025-03-01"}}}], "carol")
    assert res[0]["ok"], res
    cur = current(env.conn, r.version_id)
    assert (cur["valid_from"], cur["valid_from_grade"], cur["locator"]) == (date(2025, 3, 1), "explicit", {"corrected_by": "carol"})
    saved = env.conn.execute("SELECT corrections, reviewer FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()
    assert saved["corrections"] == {"valid_from": {"value": "2025-03-01"}} and saved["reviewer"] == "carol"


def test_cannot_approve_before_extraction(env):
    r = env.ing.receive("t1", html("报销规定", "正文。"), "r.html", acl_policy_id=env.acl)
    res = env.review.decide("t1", [{"task_id": task_of(env.conn, r.version_id), "decision": "approve"}], "alice")
    assert not res[0]["ok"] and "抽取尚未完成" in res[0]["error"]


def test_document_type_must_be_confirmed(env):
    r = env.ing.receive("t1", html("新员工手册", "欢迎加入。"), "h.html", acl_policy_id=env.acl)
    env.worker.run_until_idle()
    t = task_of(env.conn, r.version_id)
    res = env.review.decide("t1", [{"task_id": t, "decision": "approve"}], "alice")
    assert not res[0]["ok"] and "文档类型未确认" in res[0]["error"]
    res = env.review.decide("t1", [{"task_id": t, "decision": "approve", "corrections": {"doc_type": "record"}}], "alice")
    assert res[0]["ok"] and status_of(env.conn, r.version_id) == "searchable"


def test_bad_correction_rolls_back_the_item(env):
    r = env.ing.receive("t1", html("报销规定", "本规定自2026年1月1日起施行。"), "r.html", acl_policy_id=env.acl)
    env.worker.run_until_idle()
    res = env.review.decide("t1", [{"task_id": task_of(env.conn, r.version_id), "decision": "approve", "corrections": {
        "valid_from": {"value": "2025-01-01"},
        "relations": [{"to_document_id": str(uuid.uuid4()), "to_version_no": 1}]}}], "alice")
    assert not res[0]["ok"] and "不存在" in res[0]["error"]
    # 整份回滚：仍待审核，抽取值未被改动，更正未保存
    assert status_of(env.conn, r.version_id) == "pending_review"
    assert draft(env.conn, r.version_id)["valid_from"] == date(2026, 1, 1)
    assert env.conn.execute("SELECT corrections FROM review_task WHERE version_id = %s",
                            (r.version_id,)).fetchone()["corrections"] is None


def test_delete_decision_removes_original_file(env):
    r = env.ing.receive("t1", html("报销规定", "正文。"), "r.html", acl_policy_id=env.acl)
    uri = env.conn.execute("SELECT raw_object_uri FROM document_version WHERE id = %s", (r.version_id,)).fetchone()["raw_object_uri"]
    assert env.store.exists(uri)
    res = env.review.decide("t1", [{"task_id": task_of(env.conn, r.version_id), "decision": "delete"}], "bob")
    assert res[0]["ok"] and not env.store.exists(uri) and status_of(env.conn, r.version_id) is None


def test_task_of_other_tenant_is_not_found(env):
    r = env.ing.receive("t1", html("报销规定", "正文。"), "r.html", acl_policy_id=env.acl)
    res = env.review.decide("t2", [{"task_id": task_of(env.conn, r.version_id), "decision": "delete"}], "mallory")
    assert not res[0]["ok"] and "不存在" in res[0]["error"]
    assert status_of(env.conn, r.version_id) == "pending_review"
