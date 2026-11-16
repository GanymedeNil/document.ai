"""AC-4-5：回溯可复现；并验证数据库中的回溯结果与事实表工具的计算一致。

评测装载入口按历史日期装载事实表样例，然后在一组时间点上逐个比较：
数据库按“已记录、未被替换”取出的有效期，必须等于 factset.Timeline 按规则推导的有效期。
"""
from __future__ import annotations

from datetime import timedelta

import psycopg
import pytest
from psycopg.rows import dict_row

from eaer.evalload.loader import EvalLoader
from eaer.lifecycle.asof import end_of_day, known_versions, validity_at
from factset import load
from factset.replay import Timeline
from fixture import sample, write


@pytest.fixture
def fs(tmp_path):
    return load(write(sample(), tmp_path / "sample.xlsx"))


def check_dates(fs):
    ds = set()
    for v in fs.versions.values():
        ds |= {v.uploaded_on, v.approved_on}
        if v.valid_from:
            ds.add(v.valid_from)
    for e in fs.events:
        ds.add(e.occurred_on)
    for q in fs.queries:
        ds |= {d for d in (q.asked_on, q.as_of_valid, q.as_of_system) if d}
    return sorted(ds | {d - timedelta(days=1) for d in ds} | {d + timedelta(days=1) for d in ds})


def snapshot(conn, fs, ids):
    """每个时间点上，每个规范类版本当时记录的有效期。"""
    out = {}
    for d in check_dates(fs):
        t = end_of_day(d)
        for v in fs.versions.values():
            if not fs.documents[v.document_id].normative:
                continue
            r = validity_at(conn, ids[v.id], t)
            out[(d, v.id)] = None if r is None else (
                r["valid_from"], r["valid_from_grade"], r["valid_to"],
                r["valid_to_grade"] if r["valid_to"] is not None else None)
    return out


def expected(fs):
    tl = Timeline(fs)
    out = {}
    for d in check_dates(fs):
        for v in fs.versions.values():
            if not fs.documents[v.document_id].normative:
                continue
            if not tl.approved(v, d):
                out[(d, v.id)] = None
                continue
            vf, g = tl.validity(v, d)
            vt, gt = tl.valid_to(v, d)
            out[(d, v.id)] = (vf, g, vt, gt)
    return out


def load_into(url, fs):
    conn = psycopg.connect(url, autocommit=True, row_factory=dict_row)
    ids = EvalLoader(conn).load(fs)
    return conn, ids


def test_loader_requires_eval_environment(conn, monkeypatch):
    monkeypatch.delenv("EAER_ENV", raising=False)
    with pytest.raises(PermissionError, match="EAER_ENV=eval"):
        EvalLoader(conn)
    monkeypatch.setenv("EAER_ENV", "production")
    with pytest.raises(PermissionError):
        EvalLoader(conn)


def test_database_matches_fact_table_rules(fs, db_url, monkeypatch):
    monkeypatch.setenv("EAER_ENV", "eval")
    conn, ids = load_into(db_url, fs)
    got, want = snapshot(conn, fs, ids), expected(fs)
    diffs = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    assert diffs == {}
    assert len(want) > 100   # 覆盖足够多的时间点
    # 回溯时可见的版本与事实表一致
    tl = Timeline(fs)
    for d in check_dates(fs):
        seen = known_versions(conn, "eval", end_of_day(d))
        assert seen == {ids[v.id] for v in fs.versions.values() if tl.approved(v, d)}
    conn.close()


def test_replay_is_reproducible(fs, make_db, monkeypatch):
    monkeypatch.setenv("EAER_ENV", "eval")
    a, ids_a = load_into(make_db(), fs)
    b, ids_b = load_into(make_db(), fs)
    snap_a, snap_b = snapshot(a, fs, ids_a), snapshot(b, fs, ids_b)
    assert snap_a == snap_b
    # 同一数据库重复查询也一致
    assert snapshot(a, fs, ids_a) == snap_a
    a.close()
    b.close()


def test_loaded_state_matches_events(fs, db_url, monkeypatch):
    monkeypatch.setenv("EAER_ENV", "eval")
    conn, ids = load_into(db_url, fs)
    status = {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM document_version")}
    assert all(s == "searchable" for s in status.values())     # P005 已取消归档
    events = [r["event"] for r in conn.execute(
        "SELECT event FROM lifecycle_event WHERE version_id = %s ORDER BY occurred_at", (ids["P005-v1"],))]
    assert events == ["published", "archived", "unarchived"]
    batch = conn.execute("SELECT import_batch_id FROM document_version WHERE id = %s", (ids["P004-v1"],)).fetchone()
    assert batch["import_batch_id"] == "legacy-import"
    n = conn.execute("SELECT count(*) AS n FROM evidence").fetchone()["n"]
    assert n == len(fs.facts)
    conn.close()
