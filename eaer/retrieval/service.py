"""检索服务（开发文档 5.2、7.4、10.1 第 4–5 步）。

1. 入口检查授权范围，缺失即拒绝；
2. 授权范围与时间过滤编译成 Filters，下推到索引；
3. 关键词检索与向量检索各取若干条，RRF 融合；
4. 在数据库中核对每条候选的版本状态、权限和有效期，剔除不通过的——覆盖索引更新的延迟窗口；
   回溯查询在这一步把有效期换成回溯时间点当时记录的值；
5. 有重排模型时重排，取前 top_m 条。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Protocol

import psycopg

from eaer.domain.identity import Scope
from eaer.lifecycle.asof import end_of_day
from eaer.retrieval.embed import Embedder
from eaer.retrieval.index import Filters, MissingScope, SearchIndex, rrf

RANK = {"explicit": 2, "inferred": 1, "unknown": 0}


@dataclass(frozen=True)
class Hit:
    evidence_id: str
    version_id: str
    document_id: str
    modality: str
    content: str
    locator: dict
    score: float
    valid_grade: str | None


class Reranker(Protocol):
    def rerank(self, query: str, passages: list[str]) -> list[float]: ...


class Retriever:
    def __init__(self, conn: psycopg.Connection, index: SearchIndex, embedder: Embedder,
                 reranker: Reranker | None = None, *, k_lexical: int = 50, k_vector: int = 50,
                 rrf_k: int = 60, top_n: int = 50, top_m: int = 10, backtrack_factor: int = 3):
        self.conn, self.index, self.embedder, self.reranker = conn, index, embedder, reranker
        self.k_lexical, self.k_vector, self.rrf_k = k_lexical, k_vector, rrf_k
        self.top_n, self.top_m, self.backtrack_factor = top_n, top_m, backtrack_factor

    def search(self, scope: Scope | None, question: str, *, today: date,
               as_of_valid: date | None = None, as_of_system: date | None = None) -> list[Hit]:
        if scope is None or not scope.tenant_id or not scope.principals:
            raise MissingScope("检索请求必须带授权范围")
        system_t = end_of_day(as_of_system) if as_of_system else None
        on = as_of_valid or as_of_system or today
        f = Filters(scope.tenant_id, frozenset(scope.principals), on, system_t)
        factor = self.backtrack_factor if system_t else 1      # 回溯在检索后才筛有效期，候选放大
        index = self.index.snapshot()                          # 两路检索发往同一个索引，别名切换不混用（AC-5-3）
        lexical = index.search_text(question, f, self.k_lexical * factor)
        qv = (self.embedder.embed_query(question) if hasattr(self.embedder, "embed_query")
              else self.embedder.embed([question])[0])            # 查询可带指令前缀，文档不带
        vector = index.search_vector(qv, f, self.k_vector * factor)
        fused = rrf(lexical, vector, k=self.rrf_k)[: self.top_n * factor]
        hits = self._recheck(scope, fused, on, system_t)[: self.top_n]
        if self.reranker is not None and hits:
            scores = self.reranker.rerank(question, [h.content for h in hits])
            hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]
        return hits[: self.top_m]

    def _recheck(self, scope: Scope, fused: list[tuple[str, float]], on: date, system_t: datetime | None) -> list[Hit]:
        if not fused:
            return []
        ids = [i for i, _ in fused]
        rows = {r["id"]: r for r in self.conn.execute(
            "SELECT e.id, e.version_id, e.modality, e.content, e.locator, v.status, v.approved_at, v.doc_date, "
            "  d.id AS document_id, d.tenant_id, d.doc_type, p.principals "
            "FROM evidence e JOIN document_version v ON v.id = e.version_id JOIN document d ON d.id = v.document_id "
            "JOIN acl_policy p ON p.id = v.acl_policy_id WHERE e.id = ANY(%s)", (ids,))}
        versions = list({r["version_id"] for r in rows.values()})
        if system_t is None:
            validity = {r["version_id"]: r for r in self.conn.execute(
                "SELECT * FROM version_validity WHERE version_id = ANY(%s) AND recorded_at IS NOT NULL "
                "AND replaced_at IS NULL", (versions,))}
        else:
            validity = {r["version_id"]: r for r in self.conn.execute(
                "SELECT * FROM version_validity WHERE version_id = ANY(%s) AND recorded_at IS NOT NULL "
                "AND recorded_at <= %s AND (replaced_at IS NULL OR replaced_at > %s)", (versions, system_t, system_t))}
        out = []
        for eid, score in fused:
            r = rows.get(eid)
            if r is None or r["tenant_id"] != scope.tenant_id or r["status"] != "searchable":
                continue
            if not (set(r["principals"]) & scope.principals):
                continue
            if system_t is not None and r["approved_at"] > system_t:
                continue
            val = validity.get(r["version_id"])
            if not _in_force(r["doc_type"], r["doc_date"], val, on):
                continue
            out.append(Hit(eid, str(r["version_id"]), str(r["document_id"]), r["modality"], r["content"],
                           r["locator"], score, _grade(r["doc_type"], val)))
        return out


def _in_force(doc_type: str | None, doc_date: date | None, val: dict | None, on: date) -> bool:
    if doc_type == "record":
        return doc_date is not None and doc_date <= on
    if val is None:
        return False
    return ((val["valid_from"] is None or val["valid_from"] <= on)
            and (val["valid_to"] is None or val["valid_to"] > on))


def _grade(doc_type: str | None, val: dict | None) -> str | None:
    if doc_type == "record" or val is None:
        return None
    grades = [val["valid_from_grade"]] + ([val["valid_to_grade"]] if val["valid_to"] else [])
    return min(grades, key=RANK.get)
