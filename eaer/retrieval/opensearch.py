"""OpenSearch 实现（开发文档 5.1–5.3）。

整个部署一个索引，名为 eaer-<构建编号>，查询经别名 eaer 访问；更换 Embedding 模型或分词器时
新建索引全量重建，完成后切换别名（AC-5-3）。关键词检索与向量检索各发一次查询，在应用侧用 RRF 融合，
两路都带同一组过滤条件。

中文分词有两种方式：由引擎的分词器处理（默认 standard，内置，中文按字切分；smartcn、icu_analyzer 需
docker/opensearch 镜像中的插件）；传入 segmenter 时在应用侧切词（如 jieba），text 字段改用内置 whitespace
分词器，见 segment.py。默认取 standard 是 2026-11-13 用户依据合成集 v2 的对比选定的（AC-5-4）：问法与原文
用词不同时它的召回最高。
2026-11-12 起集成测试在真实 OpenSearch 3.9.0 上运行。
"""
from __future__ import annotations

import json
from typing import Any

from dataclasses import replace

from eaer.retrieval.index import Filters, IndexDoc
from eaer.retrieval.segment import Segmenter

ALIAS = "eaer"
DEFAULT_ANALYZER = "standard"      # 部署的默认分词方式（EAER_TEXT_ANALYZER 可改；换分词方式须重建索引）
PRE_SEGMENTED = "whitespace"       # 应用侧已切好词时 text 字段使用的分词器
# 一次批量写入请求的上限（字节）。一份大文档（上万个表格单元格）的全部证据连同向量可达数百 MB，
# 一次发出去会超过 http.max_content_length（默认 100 MB，413）或协调节点的内存上限（堆的 10%，429）
MAX_BULK_BYTES = 5 * 1024 * 1024


def mapping(dims: int, analyzer: str = DEFAULT_ANALYZER, segmenter: str | None = None) -> dict:
    """segmenter 不为空时表示 text 已在应用侧切词，记入 _meta，便于核对索引与应用配置一致。"""
    kw = {"type": "keyword"}
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {"_meta": {"text_analysis": segmenter or analyzer}, "properties": {
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


def connect(url: str) -> Any:
    """创建 OpenSearch 客户端。按边界规则 3.2，只有 retrieval 导入检索引擎客户端，其余模块经此取得。"""
    from opensearchpy import OpenSearch

    return OpenSearch(url)


class OpenSearchIndex:
    def __init__(self, client: Any, alias: str = ALIAS, segmenter: Segmenter | None = None,
                 max_bulk_bytes: int = MAX_BULK_BYTES):
        self.client = client
        self.alias = alias
        self.segmenter = segmenter
        self.max_bulk_bytes = max_bulk_bytes

    # ---------- 索引管理 ----------

    def create(self, build_id: str, dims: int, analyzer: str = DEFAULT_ANALYZER) -> str:
        """应用侧切词时忽略 analyzer，text 字段固定用 whitespace。"""
        name = f"{self.alias}-{build_id}"
        if self.segmenter is not None:
            body = mapping(dims, PRE_SEGMENTED, self.segmenter.name)
        else:
            body = mapping(dims, analyzer)
        self.client.indices.create(index=name, body=body)
        return name

    def _seg(self, text: str) -> str:
        return " ".join(self.segmenter.segment(text)) if self.segmenter is not None else text

    def switch_alias(self, new_index: str) -> None:
        """原子切换别名：同一次请求里移除旧索引、加上新索引，查询不中断。"""
        actions: list[dict] = []
        if self.client.indices.exists_alias(name=self.alias):
            current = self.client.indices.get_alias(name=self.alias)
            actions += [{"remove": {"index": i, "alias": self.alias}} for i in current if i != new_index]
        actions.append({"add": {"index": new_index, "alias": self.alias}})
        self.client.indices.update_aliases(body={"actions": actions})

    def snapshot(self) -> "OpenSearchIndex":
        """把别名解析成它此刻指向的索引；本次查询的所有请求都发往这个索引（AC-5-3）。
        已经是具体索引名、或别名同时指向多个索引（不应出现）时，原样返回。"""
        if not self.client.indices.exists_alias(name=self.alias):
            return self
        names = list(self.client.indices.get_alias(name=self.alias))
        return OpenSearchIndex(self.client, alias=names[0], segmenter=self.segmenter) if len(names) == 1 else self

    def builds(self) -> dict[str, int]:
        """本别名下的全部索引版本及其创建时间（毫秒时间戳）。"""
        got = self.client.indices.get(index=f"{self.alias}-*", ignore_unavailable=True, allow_no_indices=True)
        return {name: int(v["settings"]["index"]["creation_date"]) for name, v in got.items()}

    def text_analysis(self) -> str | None:
        """别名当前指向的索引在映射 _meta 中记录的分词方式（standard、smartcn、jieba-0.42.1 等）。"""
        names = sorted(self.current())
        if not names:
            return None
        got = self.client.indices.get_mapping(index=names[0])
        return (got[names[0]]["mappings"].get("_meta") or {}).get("text_analysis")

    def current(self) -> set[str]:
        if not self.client.indices.exists_alias(name=self.alias):
            return set()
        return set(self.client.indices.get_alias(name=self.alias))

    def drop_index(self, name: str) -> None:
        if not name.startswith(f"{self.alias}-"):
            raise ValueError(f"{name} 不属于别名 {self.alias}")
        self.client.indices.delete(index=name, ignore_unavailable=True)

    def drop_all(self) -> None:
        """删除本别名下的全部索引版本（评测与测试清理用）。"""
        self.client.indices.delete(index=f"{self.alias}-*", ignore_unavailable=True)

    # ---------- 写入 ----------

    def upsert(self, docs: list[IndexDoc]) -> None:
        if not docs:
            return
        body: list[dict] = []
        for d in docs:
            if self.segmenter is not None:
                d = replace(d, text=self._seg(d.text))
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
        """body 是“动作、内容”成对的列表；按 MAX_BULK_BYTES 分成几次请求，最后一次刷新（之前的随之可见）。
        中途失败时任务重试，已写入的证据下次按 indexed_ids 跳过。"""
        chunks, cur, size = [], [], 0
        for i in range(0, len(body), 2):
            pair = body[i:i + 2]
            n = sum(len(json.dumps(x, ensure_ascii=False, default=str).encode()) + 1 for x in pair)
            if cur and size + n > self.max_bulk_bytes:
                chunks.append(cur)
                cur, size = [], 0
            cur += pair
            size += n
        if cur:
            chunks.append(cur)
        for k, chunk in enumerate(chunks):
            resp = self.client.bulk(body=chunk, refresh=k == len(chunks) - 1)
            if resp.get("errors"):
                failed = [it for it in resp.get("items", []) if any(v.get("error") for v in it.values())]
                raise RuntimeError(f"OpenSearch 批量写入有 {len(failed)} 条失败：{failed[:3]}")

    # ---------- 读取 ----------

    def indexed_ids(self, version_id: str) -> set[str]:
        """一个版本已写入索引的全部证据 id。用滚动查询取全：一份大文档的证据可以超过单次查询的 10,000 条上限
        （超过时原先只取到前 10,000 条，其余每次都当成新证据重写）。"""
        from opensearchpy.helpers import scan
        return {h["_id"] for h in scan(self.client, index=self.alias, _source=False, size=5000,
                                       query={"query": {"term": {"version_id": version_id}}})}

    def version_ids(self, tenant_id: str) -> set[str]:
        resp = self.client.search(index=self.alias, body={
            "size": 0, "query": {"term": {"tenant_id": tenant_id}},
            "aggs": {"v": {"terms": {"field": "version_id", "size": 100000}}}})
        return {b["key"] for b in resp["aggregations"]["v"]["buckets"]}

    def search_text(self, query: str, f: Filters, k: int) -> list[tuple[str, float]]:
        q = self._seg(query)
        if not q.strip():
            return []
        resp = self.client.search(index=self.alias, body=text_query(q, f, k))
        return [(h["_id"], h["_score"]) for h in resp["hits"]["hits"]]

    def search_vector(self, vector: list[float], f: Filters, k: int) -> list[tuple[str, float]]:
        resp = self.client.search(index=self.alias, body=vector_query(vector, f, k))
        return [(h["_id"], h["_score"]) for h in resp["hits"]["hits"]]
