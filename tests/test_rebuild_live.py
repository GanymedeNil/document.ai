"""AC-5-3：别名切换（开发文档 5.1、5.3）。需要真实 OpenSearch，未设置 EAER_TEST_OPENSEARCH_URL 时跳过。

重建期间多个线程持续发起查询：
- 查询无中断：没有任何查询报错，且每次都有结果；
- 同一次查询不混用新旧索引：每次查询的两路检索发往同一个具体索引；查询覆盖了切换前后两个索引；
- 重建期间发布的新文档在切换后可检索（补齐）；旧索引过了保留期由 drop_retired 删除，删除时查询照常。
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections import defaultdict
from datetime import date, timedelta

import psycopg
import pytest
from psycopg.rows import dict_row

from eaer.ingestion.indexer import Indexer, drop_retired, rebuild
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.retrieval.embed import HashingEmbedder
from eaer.retrieval.opensearch import OpenSearchIndex, connect
from eaer.retrieval.service import Retriever, Scope
from eaer.storage import LocalFileStore

URL = os.environ.get("EAER_TEST_OPENSEARCH_URL")
pytestmark = pytest.mark.skipif(not URL, reason="未设置 EAER_TEST_OPENSEARCH_URL")
SCOPE = Scope("t1", frozenset({"R_STAFF"}))
TODAY = date(2026, 11, 12)


class Recording:
    """记录每次查询的检索请求发往了哪个具体索引。"""

    def __init__(self, inner):
        self.inner, self.indices = inner, inner.indices
        self.tls, self.log, self.lock = threading.local(), defaultdict(set), threading.Lock()

    def search(self, index, body, **kw):
        key = getattr(self.tls, "query", None)
        if key is not None:
            with self.lock:
                self.log[key].add(index)
        return self.inner.search(index=index, body=body, **kw)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def page(i: int, extra: str = "") -> bytes:
    return (f"<h1>住宿规定{i}</h1><p>第一条　{i}号城市住宿上限为每晚{600 + i}元。{extra}</p>"
            "<p>第二条　本规定自2026年1月1日起施行。</p>").encode()


def test_rebuild_switches_alias_without_interrupting_queries(db_url, tmp_path):
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    ing = Ingestion(conn, LocalFileStore(tmp_path / "files"))
    ing.lc.ensure_tenant("t1")
    acl = ing.lc.create_acl_policy("t1", ["R_STAFF"])
    plain = Worker(ing)

    def publish(data: bytes, name: str) -> uuid.UUID:
        r = ing.receive("t1", data, name, acl_policy_id=acl)
        plain.run_until_idle()
        conn.execute("UPDATE document SET doc_type = 'normative' WHERE id = %s", (r.document_id,))
        ing.lc.publish(r.version_id, "alice")
        return r.version_id

    for i in range(30):
        publish(page(i), f"p{i}.html")
    alias = f"eaer-ac53-{uuid.uuid4().hex[:8]}"
    client, emb = Recording(connect(URL)), HashingEmbedder()
    first = rebuild(conn, client, alias, build_id="b1", embedder=emb)
    worker = Worker(ing, Indexer(conn, OpenSearchIndex(client, alias=alias), emb, "b1"))
    worker.run_until_idle()

    stop, errors, counts = threading.Event(), [], []

    def loop():
        qconn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
        retriever = Retriever(qconn, OpenSearchIndex(client, alias=alias), emb)
        n = 0
        while not stop.is_set():
            n += 1
            client.tls.query = (threading.get_ident(), n)
            try:
                counts.append(len(retriever.search(SCOPE, "城市住宿上限是多少", today=TODAY)))
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
        qconn.close()

    threads = [threading.Thread(target=loop) for _ in range(3)]
    try:
        for t in threads:
            t.start()
        time.sleep(0.5)

        def during():   # 重建进行中发布的新文档：索引任务经别名写进旧索引
            publish(page(99, "海外差旅意外保险保额为200万元。"), "late.html")
            worker.run_until_idle()

        second = rebuild(conn, client, alias, build_id="b2", embedder=emb, analyzer="cjk", before_switch=during)
        time.sleep(0.5)
        dropped = drop_retired(client, alias, older_than=timedelta(0))      # 模拟保留期已过
        time.sleep(0.5)
    finally:
        stop.set()
        for t in threads:
            t.join()

    try:
        assert errors == [], errors[:3]
        assert counts and min(counts) > 0
        used = list(client.log.values())
        assert used and all(len(s) == 1 for s in used), [s for s in used if len(s) != 1][:3]
        assert {f"{alias}-b1", f"{alias}-b2"} <= set().union(*used)
        assert first["retired"] == [] and second["retired"] == [f"{alias}-b1"] and dropped == [f"{alias}-b1"]
        assert second["caught_up"] >= 1
        late = Retriever(conn, OpenSearchIndex(client, alias=alias), emb).search(SCOPE, "海外差旅意外保险保额", today=TODAY)
        assert any("200万元" in h.content for h in late), [h.content for h in late]
        assert OpenSearchIndex(client, alias=alias).current() == {f"{alias}-b2"}
    finally:
        OpenSearchIndex(client, alias=alias).drop_all()
        conn.close()
