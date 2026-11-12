"""索引任务（开发文档 5.3）：处理 publish_index 与 index_update。

数据库是唯一的事实来源。每次任务都按数据库当前状态把一个版本“对齐”到索引：
- 不可搜索（待审核、已归档、已删除）：从索引删除；
- 可搜索：新证据写入，向量优先取自 evidence_embedding 缓存，缓存里没有的才调用模型；
  已在索引中的证据只更新权限、有效期等字段（证据 id 由内容决定，id 相同即内容相同）；
  索引中多出来的证据删除。
任务因此可以重复执行，也不怕乱序。
"""
from __future__ import annotations

import uuid

import psycopg

from eaer.retrieval.embed import Embedder
from eaer.retrieval.index import IndexDoc, SearchIndex

RANK = {"explicit": 2, "inferred": 1, "unknown": 0}


class Indexer:
    def __init__(self, conn: psycopg.Connection, index: SearchIndex, embedder: Embedder, build_id: str):
        self.conn, self.index, self.embedder, self.build_id = conn, index, embedder, build_id

    def sync_version(self, version_id: uuid.UUID) -> None:
        v = self.conn.execute(
            "SELECT v.id, v.status, v.approved_at, v.doc_date, d.id AS document_id, d.tenant_id, d.doc_type, "
            "  p.principals, val.valid_from, val.valid_from_grade, val.valid_to, val.valid_to_grade "
            "FROM document_version v JOIN document d ON d.id = v.document_id "
            "JOIN acl_policy p ON p.id = v.acl_policy_id "
            "LEFT JOIN version_validity val ON val.version_id = v.id AND val.recorded_at IS NOT NULL AND val.replaced_at IS NULL "
            "WHERE v.id = %s", (version_id,)).fetchone()
        vid = str(version_id)
        if v is None or v["status"] != "searchable":
            self.index.delete_version(vid)
            return
        grades = [v["valid_from_grade"]] + ([v["valid_to_grade"]] if v["valid_to"] else [])
        grade = None if v["doc_type"] == "record" else min((g for g in grades if g), key=RANK.get, default=None)
        meta = {"principals": tuple(sorted(v["principals"])), "doc_type": v["doc_type"],
                "approved_at": v["approved_at"], "valid_from": v["valid_from"], "valid_to": v["valid_to"],
                "valid_grade": grade, "doc_date": v["doc_date"], "index_build_id": self.build_id}
        evidence = self.conn.execute(
            "SELECT id, modality, content FROM evidence WHERE version_id = %s ORDER BY id", (version_id,)).fetchall()
        present = self.index.indexed_ids(vid)
        wanted = {e["id"] for e in evidence}
        new = [e for e in evidence if e["id"] not in present]
        vectors = self._vectors(new)
        self.index.upsert([IndexDoc(evidence_id=e["id"], version_id=vid, document_id=str(v["document_id"]),
                                    tenant_id=v["tenant_id"], modality=e["modality"], text=e["content"],
                                    embedding=tuple(vec), **meta) for e, vec in zip(new, vectors)])
        self.index.update_fields(sorted(present & wanted), meta)
        stale = present - wanted
        if stale:  # 极少见：证据被重新解析替换。整版删除后重建，避免逐条删除的接口差异
            self.index.delete_version(vid)
            self.sync_version(version_id)


    def _vectors(self, evidence: list[dict]) -> list[list[float]]:
        """先查缓存，缺的才调用模型并写回缓存。"""
        if not evidence:
            return []
        cached = {r["evidence_id"]: r["vector"] for r in self.conn.execute(
            "SELECT evidence_id, vector FROM evidence_embedding WHERE model = %s AND evidence_id = ANY(%s)",
            (self.embedder.name, [e["id"] for e in evidence]))}
        missing = [e for e in evidence if e["id"] not in cached]
        if missing:
            fresh = self.embedder.embed([e["content"] for e in missing])
            for e, vec in zip(missing, fresh):
                self.conn.execute(
                    "INSERT INTO evidence_embedding (evidence_id, model, vector) VALUES (%s, %s, %s) "
                    "ON CONFLICT DO NOTHING", (e["id"], self.embedder.name, vec))
                cached[e["id"]] = vec
        return [cached[e["id"]] for e in evidence]


def reconcile(conn: psycopg.Connection, index: SearchIndex, tenant_id: str) -> dict[str, set[str]]:
    """对账（AC-5-2）：数据库中可搜索的版本与索引中的版本应完全一致。"""
    db = {str(r["id"]) for r in conn.execute(
        "SELECT v.id FROM document_version v JOIN document d ON d.id = v.document_id "
        "WHERE d.tenant_id = %s AND v.status = 'searchable' "
        "AND EXISTS (SELECT 1 FROM evidence e WHERE e.version_id = v.id)", (tenant_id,))}
    ix = index.version_ids(tenant_id)
    return {"missing_in_index": db - ix, "unexpected_in_index": ix - db}
