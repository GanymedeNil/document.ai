"""OpenSearch 实现（开发文档 5.1–5.3）。

整个部署一个索引，名为 eaer-<构建编号>，查询经别名 eaer 访问；更换 Embedding 模型或分词器时
新建索引全量重建，完成后切换别名（AC-5-3）。关键词检索与向量检索各发一次查询，在应用侧用 RRF 融合，
两路都带同一组过滤条件。

注意：本实现的查询语句与映射已有单元测试，并用求值器证明过滤语义与内存实现一致，
但尚未连接真实的 OpenSearch 3.9.0 运行；接入开发环境后需跑一遍集成测试。
"""
from __future__ import annotations

from typing import Any

from eaer.retrieval.index import Filters, IndexDoc

ALIAS = "eaer"


def mapping(dims: int, analyzer: str = "smartcn") -> dict:
    kw = {"type": "keyword"}
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {
            "evidence_id": kw, "version_id": kw, "document_id": kw, "tenant_id": kw,
            "principals": kw, "doc_type": kw, "modality": kw, "valid_grade": kw, "index_build_id": kw,
            "text": {"type": "text", "analyzer": analyzer},
            "embedding": {"type": "knn_vector", "dimension": dims,
                          "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}},
            "approved_at": {"type": "date"}, "valid_from": {"type": "date"},
            "valid_to": {"type": "date"}, "doc_date": {"type": "date"},
        }},
    }


def text_query(query: str, f: Filters, k: int) -> dict:
    return {"size": k, "_source": False,
            "query": {"bool": {"must": [{"match": {"text": query}}], "filter": [f.to_dsl()]}}}


def vector_query(vector: list[float], f: Filters, k: int) -> dict:
    # lucene 引擎的 knn 在检索过程中应用过滤条件，不会先取 k 条再过滤导致结果变少
    return {"size": k, "_source": False,
            "query": {"knn": {"embedding": {"vector": list(vector), "k": k, "filter": f.to_dsl()}}}}


class OpenSearchIndex:
    def __init__(self, client: Any, alias: str = ALIAS):
        self.client = client
        self.alias = alias

    # ---------- 索引管理 ----------

    def create(self, build_id: str, dims: int, analyzer: str = "smartcn") -> str:
        name = f"{self.alias}-{build_id}"
        self.client.indices.create(index=name, body=mapping(dims, analyzer))
        return name

    def switch_alias(self, new_index: str) -> None:
        """原子切换别名：同一次请求里移除旧索引、加上新索引，查询不中断。"""
        actions: list[dict] = []
        if self.client.indices.exists_alias(name=self.alias):
            current = self.client.indices.get_alias(name=self.alias)
            actions += [{"remove": {"index": i, "alias": self.alias}} for i in current if i != new_index]
        actions.append({"add": {"index": new_index, "alias": self.alias}})
        self.client.indices.update_aliases(body={"actions": actions})

    # ---------- 写入 ----------

    def upsert(self, docs: list[IndexDoc]) -> None:
        if not docs:
            return
        body: list[dict] = []
        for d in docs:
            body += [{"index": {"_index": self.alias, "_id": d.evidence_id}}, d.to_source()]
        self._bulk(body)

    def update_fields(self, evidence_ids: list[str], fields: dict) -> None:
        if not evidence_ids:
            return
        doc = {k: (v.isoformat() if hasattr(v, "isoformat") else list(v) if isinstance(v, tuple) else v)
               for k, v in fields.items()}
        body: list[dict] = []
        for i in evidence_ids:
            body += [{"update": {"_index": self.alias, "_id": i}}, {"doc": doc}]
        self._bulk(body)

    def delete_version(self, version_id: str) -> None:
        self.client.delete_by_query(index=self.alias, body={"query": {"term": {"version_id": version_id}}},
                                    refresh=True, conflicts="proceed")

    def _bulk(self, body: list[dict]) -> None:
        resp = self.client.bulk(body=body, refresh=True)
        if resp.get("errors"):
            failed = [it for it in resp.get("items", []) if any(v.get("error") for v in it.values())]
            raise RuntimeError(f"OpenSearch 批量写入有 {len(failed)} 条失败：{failed[:3]}")

    # ---------- 读取 ----------

    def indexed_ids(self, version_id: str) -> set[str]:
        resp = self.client.search(index=self.alias, body={
            "size": 10000, "_source": False, "query": {"term": {"version_id": version_id}}})
        return {h["_id"] for h in resp["hits"]["hits"]}

    def version_ids(self, tenant_id: str) -> set[str]:
        resp = self.client.search(index=self.alias, body={
            "size": 0, "query": {"term": {"tenant_id": tenant_id}},
            "aggs": {"v": {"terms": {"field": "version_id", "size": 100000}}}})
        return {b["key"] for b in resp["aggregations"]["v"]["buckets"]}

    def search_text(self, query: str, f: Filters, k: int) -> list[tuple[str, float]]:
        resp = self.client.search(index=self.alias, body=text_query(query, f, k))
        return [(h["_id"], h["_score"]) for h in resp["hits"]["hits"]]

    def search_vector(self, vector: list[float], f: Filters, k: int) -> list[tuple[str, float]]:
        resp = self.client.search(index=self.alias, body=vector_query(vector, f, k))
        return [(h["_id"], h["_score"]) for h in resp["hits"]["hits"]]
