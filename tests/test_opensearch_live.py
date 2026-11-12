"""OpenSearch 集成测试：设置 EAER_TEST_OPENSEARCH_URL 时运行（CI 中由 OpenSearch 3.9.0 服务提供），否则跳过。

在真实引擎上验证：同一组过滤条件下，OpenSearch 返回的证据集合与内存实现的判断完全一致；
写入、更新、按版本删除、别名切换和两路检索都能正常执行。
"""
from __future__ import annotations

import os
import random
import uuid
from dataclasses import replace

import pytest

from eaer.retrieval.index import MemoryIndex
from eaer.retrieval.opensearch import OpenSearchIndex
from test_retrieval import random_doc, random_filters

URL = os.environ.get("EAER_TEST_OPENSEARCH_URL")
pytestmark = pytest.mark.skipif(not URL, reason="未设置 EAER_TEST_OPENSEARCH_URL")


@pytest.fixture
def live():
    from opensearchpy import OpenSearch

    client = OpenSearch(URL)
    alias = f"eaer-test-{uuid.uuid4().hex[:8]}"
    ix = OpenSearchIndex(client, alias=alias)
    name = ix.create("b1", dims=4, analyzer=os.environ.get("EAER_TEST_OPENSEARCH_ANALYZER", "standard"))
    ix.switch_alias(name)
    yield client, ix, alias
    client.indices.delete(index=f"{alias}-*", ignore_unavailable=True)


def test_filters_match_memory_implementation(live):
    client, ix, alias = live
    rng = random.Random(7)
    docs = []
    for i in range(300):
        d = random_doc(rng)
        docs.append(replace(d, evidence_id=f"ev_{i:016x}", version_id=f"v{i % 30}",
                            embedding=tuple(rng.uniform(-1, 1) for _ in range(4))))
    ix.upsert(docs)
    mem = MemoryIndex()
    mem.upsert(docs)
    for _ in range(60):
        f = random_filters(rng)
        resp = client.search(index=alias, body={"size": 1000, "_source": False, "query": f.to_dsl()})
        got = {h["_id"] for h in resp["hits"]["hits"]}
        assert got == {d.evidence_id for d in docs if f.matches(d)}


def test_write_search_and_delete(live):
    client, ix, alias = live
    d = replace(random_doc(random.Random(3)), evidence_id="ev_" + "a" * 16, version_id="v1", tenant_id="t1",
                principals=("A",), doc_type="normative", valid_from=None, valid_to=None,
                text="一线城市住宿上限 650", embedding=(1.0, 0.0, 0.0, 0.0))
    ix.upsert([d])
    f = random_filters(random.Random(5))
    f = replace(f, tenant_id="t1", principals=frozenset({"A"}), system_t=None)
    assert ix.search_text("住宿上限", f, 10)[0][0] == d.evidence_id
    assert ix.search_vector([1.0, 0.0, 0.0, 0.0], f, 10)[0][0] == d.evidence_id
    ix.update_fields([d.evidence_id], {"principals": ("B",)})
    assert ix.search_text("住宿上限", f, 10) == []
    assert ix.indexed_ids("v1") == {d.evidence_id} and ix.version_ids("t1") == {"v1"}
    ix.delete_version("v1")
    assert ix.indexed_ids("v1") == set()
