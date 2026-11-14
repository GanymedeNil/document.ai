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


def test_deleted_tasks_are_listed_with_who_and_when(env):
    """删除待审核版本后，审核记录仍按租户列出：保留删除人和时间，不保留内容（6.3）；其他租户既看不到也不能处理。"""
    batch = uuid.uuid4()
    r = env.ing.receive("t1", html("临时草稿", "第一条　草稿内容。"), "draft.html", acl_policy_id=env.acl, batch_id=batch)
    env.worker.run_until_idle()
    task = task_of(env.conn, r.version_id)
    assert env.review.decide("t1", [{"task_id": task, "decision": "delete"}], "bob")[0]["ok"]

    items, nxt = env.review.list_tasks("t1", status="deleted")
    assert nxt is None and [i["task_id"] for i in items] == [uuid.UUID(task)]
    item = items[0]
    assert item["reviewer"] == "bob" and item["decided_at"] is not None and item["batch_id"] == batch
    assert all(item[k] is None for k in ("version_id", "document_id", "title", "suggestions", "corrections",
                                         "extraction_status", "draft_validity"))
    assert env.review.list_tasks("t1", status="deleted", batch_id=batch)[0] == items
    assert env.review.list_tasks("other-tenant", status="deleted")[0] == []
    # 其他租户处理这个任务：与不存在的任务一样，不暴露“已处理”
    assert env.review.decide("other-tenant", [{"task_id": task, "decision": "delete"}], "eve")[0]["error"] == "审核任务不存在"
    assert "已处理" in env.review.decide("t1", [{"task_id": task, "decision": "delete"}], "bob")[0]["error"]


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


def test_batch_item_failing_date_check_does_not_affect_others(env):
    """同一文档的新版本与旧版本同日生效：该份返回可读错误，同批其余份照常发布（6.3、7.3）。"""
    def up(title, ref, body="第一条　总则。"):
        r = env.ing.receive("t1", html(title, body, "本规定自2026年1月1日起施行。"), f"{ref}.html",
                            acl_policy_id=env.acl, source_ref=ref)
        env.worker.run_until_idle()
        return r.version_id
    approve = lambda v: {"task_id": task_of(env.conn, v), "decision": "approve",   # noqa: E731
                         "corrections": {"doc_type": "normative"}}
    v1 = up("住宿规定", "INS")
    assert env.review.decide("t1", [approve(v1)], "alice")[0]["ok"]
    a, bad, b = up("采购规定", "A"), up("住宿规定", "INS", "第一条　修订后的总则。"), up("会议规定", "B")
    assert bad != v1
    res = env.review.decide("t1", [approve(a), approve(bad), approve(b)], "alice")
    assert [r["ok"] for r in res] == [True, False, True]
    assert "不晚于被取代版本的生效日 2026-01-01" in res[1]["error"]
    assert [status_of(env.conn, v) for v in (a, bad, b)] == ["searchable", "pending_review", "searchable"]


def test_batch_item_rejected_by_database_invariant_is_isolated(env, monkeypatch):
    """业务检查漏掉、由数据库不变量拒绝的情形：该份回滚并报错，不中断整批。"""
    vs = []
    for i in range(3):
        r = env.ing.receive("t1", html(f"规定{i}", "本规定自2026年1月1日起施行。"), f"r{i}.html", acl_policy_id=env.acl)
        vs.append(r.version_id)
    env.worker.run_until_idle()
    real = env.lc.publish

    def publish(vid, actor):
        if vid == vs[1]:   # 模拟应用代码出错：写入一个空区间，由检查约束 interval_ordered 拒绝
            env.conn.execute("UPDATE version_validity SET valid_to = valid_from WHERE version_id = %s", (vid,))
        return real(vid, actor)

    monkeypatch.setattr(env.lc, "publish", publish)
    res = env.review.decide("t1", [{"task_id": task_of(env.conn, v), "decision": "approve",
                                    "corrections": {"doc_type": "normative"}} for v in vs], "alice")
    assert [r["ok"] for r in res] == [True, False, True]
    assert res[1]["error"].startswith("违反数据不变量：") and "interval_ordered" in res[1]["error"]
    assert status_of(env.conn, vs[1]) == "pending_review" and draft(env.conn, vs[1])["valid_to"] is None
