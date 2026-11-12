"""生命周期服务：发布、作废日推导、更正、归档、权限变更（开发文档 6.4–6.6、7.3）。"""
from __future__ import annotations

from datetime import date

import pytest

from dbutil import seed
from eaer.lifecycle.asof import end_of_day, validity_at
from eaer.lifecycle.service import InvalidState


def current(conn, vid):
    return conn.execute("SELECT * FROM version_validity WHERE version_id = %s AND recorded_at IS NOT NULL "
                        "AND replaced_at IS NULL", (vid,)).fetchone()


def test_publish_closes_predecessor_with_explicit_date(conn):
    lc, clock, ids = seed(conn)
    clock.set(date(2025, 12, 18))
    lc.publish(ids["v2"], "alice")
    v1 = current(conn, ids["v1"])
    assert (v1["valid_to"], v1["valid_to_grade"], v1["valid_to_reason"]) == (date(2026, 1, 1), "explicit", "new_version")
    # 发布前一刻的认知：v1 仍未作废
    before = validity_at(conn, ids["v1"], end_of_day(date(2025, 12, 17)))
    assert before["valid_to"] is None
    jobs = conn.execute("SELECT stage, count(*) AS n FROM ingest_job WHERE status = 'queued' GROUP BY stage").fetchall()
    assert {r["stage"]: r["n"] for r in jobs} == {"publish_index": 2, "parse_deep": 2, "index_update": 1}


def test_inferred_successor_closes_with_approval_date_then_correction_updates(conn):
    lc, clock, ids = seed(conn)
    lc.set_draft_validity(ids["v2"], valid_from=None, valid_from_grade="inferred")
    clock.set(date(2025, 12, 18))
    lc.publish(ids["v2"], "alice")
    v1 = current(conn, ids["v1"])
    assert (v1["valid_to"], v1["valid_to_grade"]) == (date(2025, 12, 18), "inferred")
    # 审核人补录新版本生效日后，旧版本的作废日随之更正
    clock.set(date(2026, 2, 1))
    lc.correct_validity(ids["v2"], "carol", valid_from=date(2026, 1, 1))
    v1 = current(conn, ids["v1"])
    assert (v1["valid_to"], v1["valid_to_grade"]) == (date(2026, 1, 1), "explicit")
    v2 = current(conn, ids["v2"])
    assert (v2["valid_from"], v2["valid_from_grade"], v2["locator"]) == (date(2026, 1, 1), "explicit", {"corrected_by": "carol"})
    # 更正之前的回溯仍看到旧值
    old = validity_at(conn, ids["v1"], end_of_day(date(2026, 1, 31)))
    assert (old["valid_to"], old["valid_to_grade"]) == (date(2025, 12, 18), "inferred")


def test_archive_and_unarchive(conn):
    lc, clock, ids = seed(conn)
    clock.set(date(2026, 3, 1))
    assert lc.archive_document(ids["doc"], "alice") == [ids["v1"]]   # 待审核的 v2 不受影响
    row = conn.execute("SELECT status, archived_at FROM document_version WHERE id = %s", (ids["v1"],)).fetchone()
    assert row["status"] == "archived" and row["archived_at"] is not None
    with pytest.raises(InvalidState):
        lc.archive_document(ids["doc"], "alice")
    clock.set(date(2026, 4, 1))
    lc.unarchive_document(ids["doc"], "alice")
    row = conn.execute("SELECT status, archived_at, approved_at FROM document_version WHERE id = %s", (ids["v1"],)).fetchone()
    assert row["status"] == "searchable" and row["archived_at"] is None
    assert row["approved_at"].date() == date(2025, 1, 3)   # 取消归档不重新审核，审核时间不变
    events = [r["event"] for r in conn.execute(
        "SELECT event FROM lifecycle_event WHERE version_id = %s ORDER BY occurred_at", (ids["v1"],))]
    assert events == ["published", "archived", "unarchived"]


def test_archived_version_cannot_be_deleted(conn):
    lc, clock, ids = seed(conn)
    clock.set(date(2026, 3, 1))
    lc.archive_document(ids["doc"], "alice")
    with pytest.raises(InvalidState):
        lc.delete_pending(ids["v1"], "alice")


def test_change_acl_keeps_old_policy(conn):
    lc, clock, ids = seed(conn)
    clock.set(date(2026, 5, 1))
    new = lc.change_acl(ids["doc"], ["R_FIN"], "alice")
    pols = {r["id"]: r["principals"] for r in conn.execute("SELECT id, principals FROM acl_policy")}
    assert pols[new] == ["R_FIN"] and pols[ids["acl"]] == ["R_STAFF"]
    rows = conn.execute("SELECT DISTINCT acl_policy_id FROM document_version WHERE document_id = %s", (ids["doc"],)).fetchall()
    assert [r["acl_policy_id"] for r in rows] == [new]
    ev = conn.execute("SELECT detail FROM lifecycle_event WHERE event = 'acl_changed'").fetchall()
    assert [e["detail"] for e in ev] == [{"old": ["R_STAFF"], "new": ["R_FIN"]}]   # 只记已发布的 v1


def test_draft_can_change_until_publish(conn):
    lc, clock, ids = seed(conn)
    lc.set_draft_validity(ids["v2"], valid_from=date(2026, 2, 1), valid_from_grade="explicit")
    clock.set(date(2025, 12, 18))
    lc.publish(ids["v2"], "alice")
    assert current(conn, ids["v2"])["valid_from"] == date(2026, 2, 1)
    with pytest.raises(InvalidState):
        lc.set_draft_validity(ids["v2"], valid_from=date(2026, 3, 1), valid_from_grade="explicit")
