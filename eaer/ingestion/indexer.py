"""索引任务（开发文档 5.3）：处理 publish_index 与 index_update。

属于 ingestion（开发文档 3.1“建索引”）：按边界规则 3.2，只有 ingestion 与 lifecycle 写检索索引。

数据库是唯一的事实来源。每次任务都按数据库当前状态把一个版本“对齐”到索引：
- 不可搜索（待审核、已归档、已删除）：从索引删除；
- 可搜索：新证据写入，向量优先取自 evidence_embedding 缓存，缓存里没有的才调用模型；
  已在索引中的证据只更新权限、有效期等字段（证据 id 由内容决定，id 相同即内容相同）；
  索引中多出来的证据删除。
任务因此可以重复执行，也不怕乱序。
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator

import psycopg

from eaer.retrieval.embed import Embedder
from eaer.retrieval.index import IndexDoc, SearchIndex
from eaer.retrieval.opensearch import DEFAULT_ANALYZER, OpenSearchIndex
from eaer.retrieval.segment import Segmenter

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


@contextmanager
def fresh_index(client: Any, alias: str, *, build_id: str, dims: int, analyzer: str = DEFAULT_ANALYZER,
                segmenter: Segmenter | None = None) -> Iterator[OpenSearchIndex]:
    """新建一个空索引并挂上别名，用完删除。供评测与对比工具使用；生产的全量重建应先写入、再切换别名（5.3）。"""
    ix = OpenSearchIndex(client, alias=alias, segmenter=segmenter)
    ix.switch_alias(ix.create(build_id, dims=dims, analyzer=analyzer))
    try:
        yield ix
    finally:
        ix.drop_all()


def rebuild(conn: psycopg.Connection, client: Any, alias: str, *, build_id: str, embedder: Embedder,
            analyzer: str = DEFAULT_ANALYZER, segmenter: Segmenter | None = None,
            before_switch: Callable[[], None] | None = None,
            clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> dict:
    """全量重建（开发文档 5.3：更换 Embedding 模型或分词器）。

    1. 新建索引 <alias>-<build_id>，把数据库中全部可搜索版本写进去（向量取自 evidence_embedding 缓存，
       换分词器不重新调用模型）；查询这时仍经别名访问旧索引；
    2. 原子切换别名；
    3. 补齐重建期间发生的变化（发布、归档、权限与有效期变更都会排入索引任务或记生命周期事件），
       这些版本按数据库当前状态再对齐一次，此时写入经别名已到新索引；
    4. 旧索引不在这里删除：保留 24 小时，由 drop_retired 定期清理，进行中的查询不会中断。
    """
    # 生命周期事件与索引任务用应用时钟记时间；留出余量，宁可多对齐几个版本（对齐可重复执行）
    started = clock() - timedelta(minutes=5)
    via_alias = OpenSearchIndex(client, alias=alias, segmenter=segmenter)
    retired = via_alias.current()
    name = via_alias.create(build_id, dims=embedder.dims, analyzer=analyzer)
    building = Indexer(conn, OpenSearchIndex(client, alias=name, segmenter=segmenter), embedder, build_id)
    versions = [r["id"] for r in conn.execute("SELECT id FROM document_version WHERE status = 'searchable'")]
    for vid in versions:
        building.sync_version(vid)
    if before_switch is not None:
        before_switch()
    via_alias.switch_alias(name)
    touched = [r["vid"] for r in conn.execute(
        "SELECT version_id AS vid FROM lifecycle_event WHERE occurred_at >= %s UNION "
        "SELECT version_id FROM ingest_job WHERE created_at >= %s AND stage IN ('publish_index', 'index_update')",
        (started, started))]
    for vid in touched:
        building.sync_version(vid)
    return {"index": name, "retired": sorted(retired - {name}), "synced": len(versions), "caught_up": len(touched)}


def drop_retired(client: Any, alias: str, *, older_than: timedelta = timedelta(hours=24),
                 now: datetime | None = None) -> list[str]:
    """删除别名不再指向、且创建已超过保留期的旧索引（5.3：旧索引保留 24 小时后删除）。"""
    ix = OpenSearchIndex(client, alias=alias)
    current = ix.current()
    cutoff = (now or datetime.now(timezone.utc)) - older_than
    dropped = []
    for name, created_ms in sorted(ix.builds().items()):
        if name in current:
            continue
        if datetime.fromtimestamp(created_ms / 1000, timezone.utc) <= cutoff:
            ix.drop_index(name)
            dropped.append(name)
    return dropped
