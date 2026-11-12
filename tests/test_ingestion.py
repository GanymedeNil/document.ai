"""入库流水线第 1–3 阶段。

端到端用例把 factset 生成的合成文档真正走一遍：接收、解析、分类与抽取、审核发布。
抽取结果逐份与事实表比对，对应 AC-6-9（分类）、AC-7-7（时间抽取）和 AC-6-8（批量导入）。
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from dbutil import Clock
from eaer.ingestion.parsers import UnsupportedFormat
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.lifecycle.asof import end_of_day, validity_at
from eaer.review.service import Review
from eaer.storage import LocalFileStore
from factset import load
from factset.render import Renderer
from factset.replay import Timeline
from fixture import sample, write

DOC_TYPE = {"规范类": "normative", "记录类": "record"}
STYLE_GRADE = {"正文写明": "explicit", "仅上传时间": "inferred", "无": "unknown"}


@pytest.fixture
def env(conn, tmp_path):
    clock = Clock()
    store = LocalFileStore(tmp_path / "files")
    ing = Ingestion(conn, store, clock)
    ing.lc.ensure_tenant("t1")
    acl = ing.lc.create_acl_policy("t1", ["R_STAFF"])
    return SimpleNamespace(conn=conn, clock=clock, store=store, ing=ing, lc=ing.lc, acl=acl,
                           worker=Worker(ing), review=Review(conn, ing.lc, store), tmp=tmp_path)


def html(title: str, *paras: str) -> bytes:
    body = "".join(f"<p>{p}</p>" for p in paras)
    return f"<html><body><h1>{title}</h1>{body}</body></html>".encode()


def draft(conn, vid):
    return conn.execute("SELECT * FROM version_validity WHERE version_id = %s AND recorded_at IS NULL", (vid,)).fetchone()


# ---------- 端到端：合成语料 ----------

@pytest.mark.parametrize("fmt", ["docx", "html"])
def test_synthetic_corpus_extraction_matches_fact_table(env, fmt):
    fs = load(write(sample(), env.tmp / "sample.xlsx"))
    corpus = env.tmp / "corpus"
    manifest = Renderer(fs).render(corpus, fmt)
    entry = {m["version_id"]: m for m in manifest["versions"]}

    ops = sorted([(v.uploaded_on, 0, v.row, v) for v in fs.versions.values()] +
                 [(v.approved_on, 1, v.row, v) for v in fs.versions.values()], key=lambda o: o[:3])
    ids, checked = {}, []
    for day, kind, _, v in ops:
        env.clock.set(day)
        if kind == 0:
            r = env.ing.receive("t1", (corpus / entry[v.id]["file"]).read_bytes(), entry[v.id]["file"],
                                acl_policy_id=env.acl, source_ref=v.document_id,
                                title=fs.documents[v.document_id].title,
                                import_batch_id=entry[v.id]["import_batch_id"])
            ids[v.id] = r.version_id
            env.worker.run_until_idle()
            checked.append(_check_extraction(env, fs, v, ids))
        else:
            res = env.review.decide("t1", [{"task_id": _task(env.conn, ids[v.id]), "decision": "approve"}], "alice")
            assert res[0]["ok"], res

    assert len(checked) == len(fs.versions)
    # 发布后的有效期（含旧版本的作废日）与事实表规则一致；比较日取第一个事件之前
    t = min(e.occurred_on for e in fs.events) - timedelta(days=1)
    tl = Timeline(fs)
    for v in fs.versions.values():
        if not fs.documents[v.document_id].normative:
            continue
        r = validity_at(env.conn, ids[v.id], end_of_day(t))
        vf, g = tl.validity(v, t)
        vt, gt = tl.valid_to(v, t)
        assert (r["valid_from"], r["valid_from_grade"], r["valid_to"],
                r["valid_to_grade"] if r["valid_to"] else None) == (vf, g, vt, gt), v.id


def _task(conn, vid):
    return str(conn.execute("SELECT id FROM review_task WHERE version_id = %s", (vid,)).fetchone()["id"])


def _check_extraction(env, fs, v, ids):
    conn, vid = env.conn, ids[v.id]
    doc = fs.documents[v.document_id]
    task = conn.execute("SELECT suggestions FROM review_task WHERE version_id = %s", (vid,)).fetchone()
    assert task["suggestions"]["doc_type"] == DOC_TYPE[doc.doc_type], v.id                 # AC-6-9
    d = draft(conn, vid)
    if doc.normative:                                                                    # AC-7-7、AC-6-8
        assert (d["valid_from"], d["valid_from_grade"]) == (v.valid_from, STYLE_GRADE[v.valid_from_style]), v.id
        if v.valid_from_style == "正文写明":
            assert "起施行" in d["locator"]["sentence"]
    else:
        row = conn.execute("SELECT doc_date FROM document_version WHERE id = %s", (vid,)).fetchone()
        assert row["doc_date"] == v.doc_date
    rels = conn.execute("SELECT to_version_id, grade FROM version_relation WHERE from_version_id = %s", (vid,)).fetchall()
    assert [(r["to_version_id"], r["grade"]) for r in rels] == ([(ids[v.supersedes], "explicit")] if v.supersedes else [])
    return v.id


# ---------- 接收 ----------

def test_duplicate_upload_returns_existing_version(env):
    data = html("差旅报销管理办法", "本办法自2026年1月1日起施行。")
    a = env.ing.receive("t1", data, "a.html", acl_policy_id=env.acl)
    b = env.ing.receive("t1", data, "b.html", acl_policy_id=env.acl)
    assert b.duplicate and b.version_id == a.version_id
    assert env.conn.execute("SELECT count(*) AS n FROM document_version").fetchone()["n"] == 1


def test_unsupported_format_is_rejected_at_receive(env):
    with pytest.raises(UnsupportedFormat):
        env.ing.receive("t1", b"%PDF-1.7", "scan.pdf", acl_policy_id=env.acl)
    assert env.conn.execute("SELECT count(*) AS n FROM document_version").fetchone()["n"] == 0


def test_relative_wording_left_unresolved(env):
    r = env.ing.receive("t1", html("报销规定", "本规定自发布之日起30日后施行。"), "r.html", acl_policy_id=env.acl)
    env.worker.run_until_idle()
    d = draft(env.conn, r.version_id)
    assert d["valid_from"] is None and d["valid_from_grade"] == "unknown" and "30日后" in d["unresolved_text"]


def test_unmatched_abolition_is_left_for_reviewer(env):
    r = env.ing.receive("t1", html("报销规定", "本规定自2026年1月1日起施行，原《旧报销办法》同时废止。"),
                        "r.html", acl_policy_id=env.acl)
    env.worker.run_until_idle()
    s = env.conn.execute("SELECT suggestions FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()
    assert s["suggestions"]["unmatched_relations"][0]["title"] == "旧报销办法"


# ---------- AC-6-2：续跑 ----------

def _evidence(conn, vid):
    return conn.execute("SELECT count(*) AS n FROM evidence WHERE version_id = %s", (vid,)).fetchone()["n"]


def test_failure_is_retried_without_duplicates(env, monkeypatch):
    r = env.ing.receive("t1", html("报销规定", "第一条　总则。", "本规定自2026年1月1日起施行。"), "r.html", acl_policy_id=env.acl)
    real = env.ing.parse_text
    calls = {"n": 0}

    def flaky(vid):
        calls["n"] += 1
        real(vid)                      # 写完证据后才失败：模拟在阶段末尾中断
        if calls["n"] == 1:
            raise RuntimeError("模拟故障")

    monkeypatch.setitem(env.worker.handlers, "parse_text", flaky)
    assert env.worker.run_once()
    job = env.conn.execute("SELECT * FROM ingest_job WHERE stage = 'parse_text'").fetchone()
    assert (job["status"], job["attempts"]) == ("queued", 1) and "模拟故障" in job["last_error"]
    env.worker.run_until_idle()
    assert _evidence(env.conn, r.version_id) == 2     # 两段正文（标题只进章节路径），没有重复
    assert {j["stage"]: j["status"] for j in env.conn.execute("SELECT stage, status FROM ingest_job")} == \
        {"parse_text": "done", "classify_extract": "done"}


def test_crashed_worker_lease_expires_and_job_resumes(env):
    r = env.ing.receive("t1", html("报销规定", "本规定自2026年1月1日起施行。"), "r.html", acl_policy_id=env.acl)
    assert env.worker.claim() is not None              # 领取后进程“被杀”，任务停在 running
    assert env.worker.run_until_idle() == 0            # 租约未到期，别的进程不能领取
    env.clock.now += timedelta(minutes=6)
    assert env.worker.run_until_idle() == 2            # 租约到期后续跑两个阶段
    assert draft(env.conn, r.version_id)["valid_from"] == date(2026, 1, 1)


def test_stages_are_idempotent(env):
    r = env.ing.receive("t1", html("报销规定", "第一条　总则。", "本规定自2026年1月1日起施行。"), "r.html", acl_policy_id=env.acl)
    env.worker.run_until_idle()
    before = _evidence(env.conn, r.version_id)
    env.ing.parse_text(r.version_id)
    env.ing.classify_extract(r.version_id)
    env.ing.classify_extract(r.version_id)
    assert _evidence(env.conn, r.version_id) == before
    assert env.conn.execute("SELECT count(*) AS n FROM version_validity WHERE version_id = %s",
                            (r.version_id,)).fetchone()["n"] == 1


def test_job_fails_after_max_attempts(env, monkeypatch):
    env.ing.receive("t1", html("报销规定", "正文。"), "r.html", acl_policy_id=env.acl)
    monkeypatch.setitem(env.worker.handlers, "parse_text", lambda vid: 1 / 0)
    env.worker.run_until_idle()
    job = env.conn.execute("SELECT status, attempts FROM ingest_job WHERE stage = 'parse_text'").fetchone()
    assert (job["status"], job["attempts"]) == ("failed", 3)


def test_deleted_version_leaves_no_jobs(env):
    r = env.ing.receive("t1", html("报销规定", "正文。"), "r.html", acl_policy_id=env.acl)
    uri = env.lc.delete_pending(r.version_id, "bob")
    env.store.delete(uri)
    assert env.worker.run_until_idle() == 0
    assert not env.store.exists(uri)


def test_title_comes_from_document_when_not_given(env):
    r = env.ing.receive("t1", html("采购审批权限规定", "正文。"), "upload-0001.html", acl_policy_id=env.acl,
                        source_ref="CRM-42")
    env.worker.run_until_idle()
    doc = env.conn.execute("SELECT title, doc_type FROM document WHERE id = %s", (r.document_id,)).fetchone()
    assert (doc["title"], doc["doc_type"]) == ("采购审批权限规定", "normative")
