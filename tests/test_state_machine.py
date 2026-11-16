"""AC-6-3：状态机（开发文档 6.5；硬性）。状态转换表每一行一个用例，每个非法转换一个用例。

合法转换全部成功，并检查表中“动作”一栏；非法转换全部被拒（InvalidState，接口层对应 409），
且数据库没有任何变化。

状态：待审核 pending_review、可搜索 searchable、已归档 archived、已删除（物理删除，版本不存在）。
事件：审核人更正、审核通过、删除、归档、取消归档、权限变更、有效期更正。

说明：
- “清除相关缓存”：查询缓存尚未实现（第 10 节 AC-10-4 时一并加），此处不检查。
- “上传或同步”：同步需要 DataSource（11.3），尚未实现，此处只测上传。
- 权限变更在待审核与已归档版本上同样执行（不改状态），理由见 Lifecycle.change_acl 与设计变更记录。
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from dbutil import Clock
from eaer.ingestion.indexer import Indexer
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.lifecycle.service import InvalidState
from eaer.retrieval.embed import HashingEmbedder
from eaer.retrieval.index import MemoryIndex
from eaer.review.service import Review
from eaer.storage import LocalFileStore

DOC = ("<html><body><h1>差旅报销管理办法</h1><p>第一条　为规范差旅报销，制定本办法。</p>"
       "<p>第二条　一线城市住宿上限为每晚650元。</p><p>第三条　本办法自2026年1月1日起施行。</p></body></html>")


@pytest.fixture
def sm(conn, tmp_path):
    clock = Clock()
    store = LocalFileStore(tmp_path / "files")
    ing = Ingestion(conn, store, clock)
    ing.lc.ensure_tenant("t1")
    acl = ing.lc.create_acl_policy("t1", ["R_STAFF"])
    index = MemoryIndex()
    worker = Worker(ing, Indexer(conn, index, HashingEmbedder(), "b1"))
    return SimpleNamespace(conn=conn, clock=clock, store=store, ing=ing, lc=ing.lc, acl=acl, index=index,
                           worker=worker, review=Review(conn, ing.lc, store))


def upload(sm, n: int = 0):
    r = sm.ing.receive("t1", DOC.replace("650", str(650 + n)).encode(), f"d{n}.html", acl_policy_id=sm.acl)
    sm.worker.run_until_idle()
    return r.document_id, r.version_id


def make(sm, state: str):
    """走真实流程把一个版本推到指定状态，返回 (文档 id, 版本 id)。"""
    doc, vid = upload(sm)
    if state in ("searchable", "archived"):
        sm.conn.execute("UPDATE document SET doc_type = 'normative' WHERE id = %s", (doc,))
        sm.lc.publish(vid, "alice")
    if state == "archived":
        sm.lc.archive_document(doc, "carol")
    if state == "deleted":
        task = sm.conn.execute("SELECT id FROM review_task WHERE version_id = %s", (vid,)).fetchone()["id"]
        assert sm.review.decide("t1", [{"task_id": str(task), "decision": "delete"}], "bob")[0]["ok"]
    sm.worker.run_until_idle()
    return doc, vid


def row(sm, vid):
    return sm.conn.execute("SELECT * FROM document_version WHERE id = %s", (vid,)).fetchone()


def current(sm, vid):
    return sm.conn.execute("SELECT * FROM version_validity WHERE version_id = %s AND recorded_at IS NOT NULL "
                           "AND replaced_at IS NULL", (vid,)).fetchone()


def events(sm, vid):
    return [r["event"] for r in sm.conn.execute(
        "SELECT event FROM lifecycle_event WHERE version_id = %s ORDER BY occurred_at", (vid,))]


def indexed(sm, vid):
    return [d for d in sm.index.docs.values() if d.version_id == str(vid)]


# ---------- 合法转换：6.5 表逐行 ----------

def test_row1_none_upload_to_pending_runs_stages_1_to_3(sm):
    r = sm.ing.receive("t1", DOC.encode(), "d.html", acl_policy_id=sm.acl)
    v = row(sm, r.version_id)
    assert v["status"] == "pending_review" and v["approved_at"] is None
    assert sm.store.exists(v["raw_object_uri"])
    task = sm.conn.execute("SELECT * FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()
    assert task["status"] == "open"
    sm.worker.run_until_idle()
    stages = {j["stage"]: j["status"] for j in sm.conn.execute(
        "SELECT stage, status FROM ingest_job WHERE version_id = %s", (r.version_id,))}
    assert stages == {"parse_text": "done", "classify_extract": "done"}
    assert sm.conn.execute("SELECT count(*) AS n FROM evidence WHERE version_id = %s", (r.version_id,)).fetchone()["n"] > 0
    draft = sm.conn.execute("SELECT * FROM version_validity WHERE version_id = %s AND recorded_at IS NULL",
                            (r.version_id,)).fetchone()
    assert draft["valid_from"] == date(2026, 1, 1)
    assert indexed(sm, r.version_id) == []                  # 待审核不进索引


def test_row2_pending_reviewer_correction_stays_pending(sm):
    doc, vid = make(sm, "pending_review")
    sm.lc.set_draft_validity(vid, valid_from=date(2026, 3, 1), valid_from_grade="explicit",
                             locator={"corrected_by": "bob"})
    assert row(sm, vid)["status"] == "pending_review"
    draft = sm.conn.execute("SELECT * FROM version_validity WHERE version_id = %s AND recorded_at IS NULL",
                            (vid,)).fetchone()
    assert (draft["valid_from"], draft["valid_from_grade"]) == (date(2026, 3, 1), "explicit")
    assert draft["locator"] == {"corrected_by": "bob"}


def test_row3_pending_approve_to_searchable(sm):
    doc, vid = make(sm, "pending_review")
    sm.conn.execute("UPDATE document SET doc_type = 'normative' WHERE id = %s", (doc,))
    sm.lc.publish(vid, "alice")
    v = row(sm, vid)
    assert v["status"] == "searchable" and v["approved_at"] is not None
    assert current(sm, vid)["recorded_at"] == v["approved_at"]
    task = sm.conn.execute("SELECT * FROM review_task WHERE version_id = %s", (vid,)).fetchone()
    assert (task["status"], task["reviewer"]) == ("approved", "alice")
    queued = {j["stage"] for j in sm.conn.execute(
        "SELECT stage FROM ingest_job WHERE version_id = %s AND status = 'queued'", (vid,))}
    assert queued == {"publish_index", "parse_deep"}           # 6.4，排入第 6 阶段
    assert events(sm, vid) == ["published"]
    sm.worker.run_until_idle()
    assert indexed(sm, vid)


def test_row4_pending_delete_removes_everything_but_the_audit_record(sm):
    doc, vid = make(sm, "pending_review")
    uri = row(sm, vid)["raw_object_uri"]
    task = sm.conn.execute("SELECT id FROM review_task WHERE version_id = %s", (vid,)).fetchone()["id"]
    assert sm.review.decide("t1", [{"task_id": str(task), "decision": "delete"}], "bob")[0]["ok"]
    assert row(sm, vid) is None and not sm.store.exists(uri)
    for table, col in [("evidence", "version_id"), ("version_validity", "version_id"),
                       ("version_relation", "from_version_id"), ("ingest_job", "version_id")]:
        assert sm.conn.execute(f"SELECT count(*) AS n FROM {table} WHERE {col} = %s", (vid,)).fetchone()["n"] == 0
    t = sm.conn.execute("SELECT * FROM review_task WHERE id = %s", (task,)).fetchone()
    assert (t["status"], t["reviewer"], t["version_id"]) == ("deleted", "bob", None)
    assert t["decided_at"] is not None and t["corrections"] is None and t["suggestions"] is None


def test_row5_searchable_archive(sm):
    doc, vid = make(sm, "searchable")
    assert indexed(sm, vid)
    sm.lc.archive_document(doc, "carol")
    v = row(sm, vid)
    assert v["status"] == "archived" and v["archived_at"] is not None
    assert events(sm, vid)[-1] == "archived"
    sm.worker.run_until_idle()
    assert indexed(sm, vid) == []                            # 从索引删除


def test_row6_archived_unarchive(sm):
    doc, vid = make(sm, "archived")
    approved = row(sm, vid)["approved_at"]
    sm.lc.unarchive_document(doc, "carol")
    v = row(sm, vid)
    assert v["status"] == "searchable" and v["archived_at"] is None
    assert v["approved_at"] == approved                      # 不重新审核，approved_at 不变（6.6）
    assert events(sm, vid)[-1] == "unarchived"
    sm.worker.run_until_idle()
    assert indexed(sm, vid)                                  # 重新写入索引


def test_row7_searchable_acl_change(sm):
    doc, vid = make(sm, "searchable")
    old = row(sm, vid)["acl_policy_id"]
    new = sm.lc.change_acl(doc, ["R_FIN"], "dave")
    v = row(sm, vid)
    assert v["status"] == "searchable" and v["acl_policy_id"] == new != old
    assert sm.conn.execute("SELECT 1 FROM acl_policy WHERE id = %s", (old,)).fetchone()   # 旧策略保留
    ev = sm.conn.execute("SELECT * FROM lifecycle_event WHERE version_id = %s AND event = 'acl_changed'",
                         (vid,)).fetchone()
    assert ev["detail"] == {"old": ["R_STAFF"], "new": ["R_FIN"]}
    sm.worker.run_until_idle()
    assert {d.principals for d in indexed(sm, vid)} == {("R_FIN",)}   # 先数据库，再索引


def test_row8_searchable_validity_correction(sm):
    doc, vid = make(sm, "searchable")
    old = current(sm, vid)
    sm.lc.correct_validity(vid, "erin", valid_from=date(2026, 2, 1))
    assert row(sm, vid)["status"] == "searchable"
    replaced = sm.conn.execute("SELECT replaced_at FROM version_validity WHERE id = %s", (old["id"],)).fetchone()
    assert replaced["replaced_at"] is not None                # 旧行写入 replaced_at，只追加
    assert (current(sm, vid)["valid_from"], current(sm, vid)["valid_from_grade"]) == (date(2026, 2, 1), "explicit")
    assert events(sm, vid)[-1] == "validity_corrected"
    sm.worker.run_until_idle()
    assert {d.valid_from for d in indexed(sm, vid)} == {date(2026, 2, 1)}


@pytest.mark.parametrize("state", ["pending_review", "archived"])
def test_acl_change_follows_versions_that_are_not_searchable(sm, state):
    """不在 6.5 表中、但必须执行：权限是文档属性，不改变状态；否则撤权会在审核通过或取消归档后丢失。"""
    doc, vid = make(sm, state)
    new = sm.lc.change_acl(doc, ["R_FIN"], "dave")
    v = row(sm, vid)
    assert v["status"] == state and v["acl_policy_id"] == new
    if state == "archived":
        sm.lc.unarchive_document(doc, "carol")
    else:
        sm.conn.execute("UPDATE document SET doc_type = 'normative' WHERE id = %s", (doc,))
        sm.lc.publish(vid, "alice")
    sm.worker.run_until_idle()
    assert {d.principals for d in indexed(sm, vid)} == {("R_FIN",)}


# ---------- 非法转换：表外的每个（状态，事件）组合 ----------

EVENTS = {
    "审核人更正": lambda sm, d, v: sm.lc.set_draft_validity(v, valid_from=date(2026, 3, 1),
                                                         valid_from_grade="explicit", locator={"corrected_by": "bob"}),
    "审核人更正关系": lambda sm, d, v: sm.lc.clear_draft_relations(v),
    "审核通过": lambda sm, d, v: sm.lc.publish(v, "alice"),
    "删除": lambda sm, d, v: sm.lc.delete_pending(v, "bob"),
    "归档": lambda sm, d, v: sm.lc.archive_document(d, "carol"),
    "取消归档": lambda sm, d, v: sm.lc.unarchive_document(d, "carol"),
    "权限变更": lambda sm, d, v: sm.lc.change_acl(d, ["R_FIN"], "dave"),
    "有效期更正": lambda sm, d, v: sm.lc.correct_validity(v, "erin", valid_from=date(2026, 2, 1)),
}
LEGAL = {
    ("pending_review", "审核人更正"), ("pending_review", "审核人更正关系"), ("pending_review", "审核通过"),
    ("pending_review", "删除"), ("searchable", "归档"), ("archived", "取消归档"), ("searchable", "权限变更"),
    ("searchable", "有效期更正"),
    ("pending_review", "权限变更"), ("archived", "权限变更"),        # 不改状态的属性更新，见上
}
STATES = ("pending_review", "searchable", "archived", "deleted")
ILLEGAL = [(s, e) for s in STATES for e in EVENTS if (s, e) not in LEGAL]
TABLES = ("document", "document_version", "version_validity", "version_relation", "evidence", "lifecycle_event",
          "review_task", "ingest_job", "acl_policy")


def snapshot(sm) -> dict:
    return {t: sm.conn.execute(f"SELECT md5(coalesce(string_agg(x::text, '|' ORDER BY x::text), '')) AS h "
                               f"FROM {t} x").fetchone()["h"] for t in TABLES}


def test_illegal_matrix_covers_the_doc_examples():
    """6.5 明确点名的两类：删除可搜索或已归档的版本、归档待审核的版本。"""
    assert {("searchable", "删除"), ("archived", "删除"), ("pending_review", "归档")} <= set(ILLEGAL)
    assert len(ILLEGAL) == len(STATES) * len(EVENTS) - len(LEGAL) == 22


@pytest.mark.parametrize("state, event", ILLEGAL, ids=[f"{s}-{e}" for s, e in ILLEGAL])
def test_illegal_transition_is_rejected_and_changes_nothing(sm, state, event):
    doc, vid = make(sm, state)
    before = snapshot(sm)
    with pytest.raises(InvalidState):
        EVENTS[event](sm, doc, vid)
    assert snapshot(sm) == before
    if state != "deleted":
        assert row(sm, vid)["status"] == state
