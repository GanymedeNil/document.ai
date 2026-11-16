"""AC-6-4：待审核不可搜（开发文档 6.5、6.7；硬性）。上传后、审核前提问，结果不涉及该文档。

端到端到检索为止：经 HTTP 接口上传，后台任务进程跑完第 1–3 阶段（以及全部可执行的任务），
再用有权限的身份检索。提问接口（第 8 节）尚未实现，“回答和引用”一半待查询执行完成后补上；
此处保证回答所依据的检索结果里没有待审核版本的证据。

三种情形：新文档待审核；已发布文档的新版本待审核；索引里意外留有待审核证据（检索后数据库核对剔除）。
每种情形都在审核通过后再查一次作为对照，证明同一个问题本来能查到。
内存索引总是运行；设置 EAER_TEST_OPENSEARCH_URL 时同样在真实 OpenSearch 上运行。
"""
from __future__ import annotations

import os
import uuid
from datetime import date
from types import SimpleNamespace

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from dbutil import Clock
from eaer.api.app import Identity, TokenAuthenticator, create_app
from eaer.ingestion.indexer import Indexer, fresh_index
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.retrieval.embed import HashingEmbedder
from eaer.retrieval.index import IndexDoc, MemoryIndex
from eaer.retrieval.opensearch import connect
from eaer.retrieval.service import Retriever, Scope
from eaer.storage import LocalFileStore

TOKENS = {"tok-ed": Identity("ed", "t1", frozenset({"editor"})),
          "tok-rv": Identity("rv", "t1", frozenset({"reviewer"}))}
SCOPE = Scope("t1", frozenset({"R_STAFF"}))
TODAY = date(2026, 6, 1)
QUESTION = "海外差旅意外保险保额是多少"
OS_URL = os.environ.get("EAER_TEST_OPENSEARCH_URL")


def page(amount: str, effective: str = "2026年1月1日") -> str:
    return (f"<h1>海外差旅保险规定</h1><p>第一条　海外差旅意外保险保额为{amount}。</p>"
            f"<p>第二条　本规定自{effective}起施行。</p>")


@pytest.fixture(params=["memory", pytest.param("opensearch", marks=pytest.mark.skipif(
    not OS_URL, reason="未设置 EAER_TEST_OPENSEARCH_URL"))])
def e2e(request, db_url, tmp_path):
    store, clock, emb = LocalFileStore(tmp_path / "files"), Clock(), HashingEmbedder()
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    client = TestClient(create_app(db_url, store, TokenAuthenticator(TOKENS), clock))

    def build(index):
        return SimpleNamespace(conn=conn, client=client, index=index,
                               worker=Worker(Ingestion(conn, store, clock), Indexer(conn, index, emb, "b1")),
                               retriever=Retriever(conn, index, emb), emb=emb)

    if request.param == "memory":
        yield build(MemoryIndex())
    else:
        with fresh_index(connect(OS_URL), f"eaer-ac64-{uuid.uuid4().hex[:8]}", build_id="b1", dims=emb.dims) as ix:
            yield build(ix)
    conn.close()


def upload(e, body: str, source_ref: str) -> dict:
    r = e.client.post("/v1/documents", headers={"Authorization": "Bearer tok-ed"},
                      files={"file": ("p.html", body.encode(), "text/html")},
                      data={"principals": "R_STAFF", "source_ref": source_ref})
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "pending_review"
    e.worker.run_until_idle()
    return r.json()


def approve(e, version_id: str) -> None:
    tasks = e.client.get("/v1/review/tasks", headers={"Authorization": "Bearer tok-rv"}).json()["items"]
    task = next(t for t in tasks if t["version_id"] == version_id)
    r = e.client.post("/v1/review/decisions", headers={"Authorization": "Bearer tok-rv"}, json={"decisions": [
        {"task_id": task["task_id"], "decision": "approve", "corrections": {"doc_type": "normative"}}]})
    assert r.json()["results"][0]["ok"], r.text
    e.worker.run_until_idle()


def versions_hit(e) -> set[str]:
    return {h.version_id for h in e.retriever.search(SCOPE, QUESTION, today=TODAY)}


def test_new_document_pending_is_not_searchable(e2e):
    up = upload(e2e, page("200万元"), "INS")
    assert versions_hit(e2e) == set()
    assert e2e.index.indexed_ids(up["version_id"]) == set()            # 待审核根本不进索引
    approve(e2e, up["version_id"])
    assert versions_hit(e2e) == {up["version_id"]}                     # 对照：审核通过后同一问题能查到


def test_pending_new_version_of_published_document_is_not_searchable(e2e):
    v1 = upload(e2e, page("200万元"), "INS")
    approve(e2e, v1["version_id"])
    v2 = upload(e2e, page("300万元", "2026年3月1日"), "INS")
    assert v2["document_id"] == v1["document_id"] and v2["version_id"] != v1["version_id"]
    hits = e2e.retriever.search(SCOPE, QUESTION, today=TODAY)
    assert {h.version_id for h in hits} == {v1["version_id"]}
    assert not any("300万元" in h.content for h in hits)
    approve(e2e, v2["version_id"])
    assert v2["version_id"] in versions_hit(e2e)


def test_pending_evidence_left_in_index_is_dropped_by_database_recheck(e2e):
    up = upload(e2e, page("200万元"), "INS")
    rows = e2e.conn.execute("SELECT e.id, e.modality, e.content, v.document_id FROM evidence e "
                            "JOIN document_version v ON v.id = e.version_id WHERE v.id = %s",
                            (up["version_id"],)).fetchall()
    assert rows
    vectors = e2e.emb.embed([r["content"] for r in rows])
    e2e.index.upsert([IndexDoc(evidence_id=r["id"], version_id=up["version_id"], document_id=str(r["document_id"]),
                               tenant_id="t1", principals=("R_STAFF",), doc_type="normative", modality=r["modality"],
                               text=r["content"], embedding=tuple(vec), approved_at=e2e.conn.execute(
                                   "SELECT now() AS t").fetchone()["t"], valid_from=None, valid_to=None,
                               valid_grade=None, doc_date=None, index_build_id="b1")
                      for r, vec in zip(rows, vectors)])
    assert e2e.index.indexed_ids(up["version_id"])                       # 索引里确实有了
    assert versions_hit(e2e) == set()                                   # 检索后核对把它剔除
