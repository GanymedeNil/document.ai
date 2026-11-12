"""检索层面的时间隔离与权限（开发文档 7.4、10.1–10.3）。

事实表样例按日期完整回放（真实的入库、审核、生命周期、索引代码），然后在大量时间点、角色、问题的组合上
逐条核对检索结果，统计违规数：
- AC-7-2（硬性）回溯隔离：回溯到 T 时，T 之后审核通过的版本不出现，T 之后的更正不生效；
- AC-10-1（硬性）越权：每个角色对每个查询各执行一次，返回的证据都可搜索且该角色有权；
- AC-10-3（硬性）撤权：撤权后立即提问（索引尚未更新）与索引更新之后提问，都检索不到。
每项同时断言结果非空，避免“什么都没查到”而空洞地通过。提问接口尚未实现，回答与引用层面待第 8 节补上。
"""
from __future__ import annotations

import uuid
from datetime import date, timedelta

import psycopg
import pytest
from psycopg.rows import dict_row

from eaer.lifecycle.asof import end_of_day, validity_at
from eaer.retrieval.service import Retriever, Scope

D = date.fromisoformat
ROLES = ("R_STAFF", "R_FIN", "R_NOBODY")
TERMS = ["住宿上限", "一线城市住宿上限", "二线城市住宿上限", "采购审批上限", "部门经理单笔采购审批上限",
         "伙食补贴", "差旅总费用", "报销管理办法", "补充通知"]


@pytest.fixture(scope="module")
def replay(template_db, tmp_path_factory):
    import os

    from conftest import _admin, url_for
    from eaer.evalrun.retrieval import run
    from eaer.retrieval.embed import HashingEmbedder
    from eaer.retrieval.index import MemoryIndex
    from eaer.storage import LocalFileStore
    from factset import load
    from factset.render import Renderer
    from fixture import sample, write

    tmp = tmp_path_factory.mktemp("replay")
    admin = _admin()
    name = f"eaer_s_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}" TEMPLATE "{template_db}"')
    conn = psycopg.connect(url_for(name), autocommit=True, row_factory=dict_row)
    fs = load(write(sample(), tmp / "s.xlsx"))
    manifest = Renderer(fs).render(tmp / "corpus", "docx")
    index, emb = MemoryIndex(), HashingEmbedder()
    old = os.environ.get("EAER_ENV")
    os.environ["EAER_ENV"] = "eval"
    try:
        run(conn, fs, tmp / "corpus", manifest, store=LocalFileStore(tmp / "files"), index=index, embedder=emb)
    finally:
        if old is None:
            os.environ.pop("EAER_ENV")
        else:
            os.environ["EAER_ENV"] = old
    yield conn, fs, index, emb, Retriever(conn, index, emb), LocalFileStore(tmp / "files")
    conn.close()
    admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.close()


def in_force(doc_type, doc_date, val, on) -> bool:
    """独立于检索实现、按开发文档 7.4 写的判断：记录类看文档日期；其余看当时记录的有效期，空值不过滤。"""
    if doc_type == "record":
        return doc_date is not None and doc_date <= on
    if val is None:
        return False
    return (val["valid_from"] is None or val["valid_from"] <= on) and (val["valid_to"] is None or on < val["valid_to"])


def questions(fs) -> list[str]:
    return [q.question for q in fs.queries] + TERMS


def timeline(fs) -> list[date]:
    days = {v.uploaded_on for v in fs.versions.values()} | {v.approved_on for v in fs.versions.values()}
    days |= {e.occurred_on for e in fs.events} | {q.asked_on for q in fs.queries}
    out = set()
    for d in days:
        out |= {d - timedelta(days=1), d, d + timedelta(days=1)}
    return sorted(out)


def versions(conn) -> dict[str, dict]:
    return {str(r["id"]): r for r in conn.execute(
        "SELECT v.id, v.status, v.approved_at, v.doc_date, d.doc_type, p.principals FROM document_version v "
        "JOIN document d ON d.id = v.document_id JOIN acl_policy p ON p.id = v.acl_policy_id")}


def test_ac_7_2_backtrack_isolation(replay):
    conn, fs, _, _, retriever, _ = replay
    vs = versions(conn)
    checked, violations = 0, []
    for t in timeline(fs):
        t_end = end_of_day(t)
        for on in sorted({t, D("2025-03-05"), D("2025-12-31"), D("2026-02-01"), D("2026-06-30")}):
            for q in questions(fs):
                for h in retriever.search(Scope("eval", frozenset({"R_STAFF", "R_FIN"})), q, today=t,
                                          as_of_valid=on, as_of_system=t):
                    checked += 1
                    v = vs[h.version_id]
                    if v["approved_at"] > t_end:
                        violations.append(("T 之后审核通过", t, on, q, h.version_id))
                    elif not in_force(v["doc_type"], v["doc_date"], validity_at(conn, uuid.UUID(h.version_id), t_end), on):
                        violations.append(("T 时记录的有效期不覆盖目标日期", t, on, q, h.version_id))
    assert checked > 500, checked
    assert violations == [], violations[:5]


def test_ac_7_2_correction_after_t_does_not_apply(replay):
    """E02（2026-10-01）把 P002-v1 的生效日补录为 2025-03-01（explicit）。此前它只有上传时间：生效日为空、
    等级 inferred，按 7.4“空值不过滤”参与检索并标为未验证。问 2025-02-15 当时的规定：
    回溯到 9 月 15 日（补录之前）仍能查到、等级 inferred；按现在的认知已知它 3 月 1 日才生效，查不到。"""
    conn, fs, _, _, retriever, _ = replay
    scope = Scope("eval", frozenset({"R_FIN"}))
    q = "部门经理单笔采购审批上限是多少"
    on = D("2025-02-15")
    then = [h for h in retriever.search(scope, q, today=D("2026-11-11"), as_of_valid=on, as_of_system=D("2026-09-15"))
            if "50000" in h.content.replace(",", "")]
    now = [h for h in retriever.search(scope, q, today=D("2026-11-11"), as_of_valid=on)
           if "50000" in h.content.replace(",", "")]
    assert then and {h.valid_grade for h in then} == {"inferred"}
    assert now == []


def test_ac_10_1_no_unauthorized_retrieval(replay):
    conn, fs, _, _, retriever, _ = replay
    vs = versions(conn)
    checked, violations = 0, []
    for role in ROLES:
        for t in sorted({q.asked_on for q in fs.queries} | {D("2026-11-11")}):
            for q in questions(fs):
                for h in retriever.search(Scope("eval", frozenset({role})), q, today=t):
                    checked += 1
                    v = vs[h.version_id]
                    if v["status"] != "searchable" or role not in v["principals"]:
                        violations.append((role, t, q, h.version_id, v["status"], v["principals"]))
                for h in retriever.search(Scope("eval", frozenset({role})), q, today=t, as_of_system=t):
                    checked += 1
                    v = vs[h.version_id]
                    if v["status"] != "searchable" or role not in v["principals"]:
                        violations.append((role, t, q, h.version_id, "回溯"))
    assert checked > 100, checked
    assert violations == [], violations[:5]
    nobody = [h for q in questions(fs) for h in retriever.search(Scope("eval", frozenset({"R_NOBODY"})), q,
                                                                 today=D("2026-11-11"))]
    assert nobody == []


def test_ac_10_3_revocation_immediately_and_after_index_update(replay):
    conn, fs, index, emb, retriever, store = replay
    from dbutil import Clock
    from eaer.ingestion.indexer import Indexer
    from eaer.ingestion.pipeline import Ingestion, Worker
    from eaer.lifecycle.service import Lifecycle

    staff = Scope("eval", frozenset({"R_STAFF"}))
    q = "一线城市住宿上限是多少"
    before = retriever.search(staff, q, today=D("2026-11-11"))
    assert before
    doc = conn.execute("SELECT document_id FROM document_version WHERE id = %s",
                       (uuid.UUID(before[0].version_id),)).fetchone()["document_id"]
    clock = Clock()
    clock.set(D("2026-11-12"))
    Lifecycle(conn, clock).change_acl(doc, ["R_FIN"], "admin")
    stale = [d for d in index.docs.values() if d.document_id == str(doc) and "R_STAFF" in d.principals]
    assert stale                                              # 索引还没更新，旧权限仍在
    immediately = retriever.search(staff, q, today=D("2026-11-11"))
    assert not any(h.document_id == str(doc) for h in immediately)          # 检索后数据库核对拦住
    Worker(Ingestion(conn, store, clock), Indexer(conn, index, emb, "eval")).run_until_idle()
    assert not any("R_STAFF" in d.principals for d in index.docs.values() if d.document_id == str(doc))
    after = retriever.search(staff, q, today=D("2026-11-11"))
    assert not any(h.document_id == str(doc) for h in after)
    fin = retriever.search(Scope("eval", frozenset({"R_FIN"})), q, today=D("2026-11-11"))
    assert any(h.document_id == str(doc) for h in fin)                      # 对照：新角色能查到
