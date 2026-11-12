"""检索引擎接口（开发文档第 5 节）。

过滤条件只在这里定义一次，同时给出两种形式：
- matches()：Python 判断，供内存参考实现使用；
- to_dsl()：OpenSearch 查询语句，供生产实现使用。
测试用一个查询语句求值器证明两者等价，这样到真实服务器上要验证的只剩引擎本身的行为。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Protocol


@dataclass(frozen=True)
class IndexDoc:
    evidence_id: str
    version_id: str
    document_id: str
    tenant_id: str
    principals: tuple[str, ...]
    doc_type: str | None
    modality: str
    text: str
    embedding: tuple[float, ...]
    approved_at: datetime
    valid_from: date | None
    valid_to: date | None
    valid_grade: str | None
    doc_date: date | None
    index_build_id: str

    def to_source(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["principals"] = list(self.principals)
        d["embedding"] = list(self.embedding)
        for k in ("approved_at", "valid_from", "valid_to", "doc_date"):
            d[k] = d[k].isoformat() if d[k] is not None else None
        return d


class MissingScope(Exception):
    """检索请求缺少授权范围，在入口拒绝（开发文档 5.2 第 1 步、AC-5-1）。"""


@dataclass(frozen=True)
class Filters:
    """每次检索都必须带的过滤条件。

    on：判断生效的日期。system_t 不为空时是回溯查询：索引里只按审核通过时间过滤，
    有效期在检索后换成 system_t 时刻记录的值再判断（开发文档 7.4）。
    """
    tenant_id: str
    principals: frozenset[str]
    on: date
    system_t: datetime | None = None

    def __post_init__(self):
        if not self.tenant_id or not self.principals:
            raise MissingScope("检索请求必须带租户和可访问角色")

    def matches(self, d: IndexDoc) -> bool:
        if d.tenant_id != self.tenant_id or not (set(d.principals) & self.principals):
            return False
        if self.system_t is not None:
            return d.approved_at <= self.system_t
        if d.doc_type == "record":
            return d.doc_date is not None and d.doc_date <= self.on
        # 规范类与类型未定的文档按有效期判断；空值不过滤（开发文档 7.4）
        return ((d.valid_from is None or d.valid_from <= self.on)
                and (d.valid_to is None or d.valid_to > self.on))

    def to_dsl(self) -> dict:
        scope = [{"term": {"tenant_id": self.tenant_id}},
                 {"terms": {"principals": sorted(self.principals)}}]
        if self.system_t is not None:
            return {"bool": {"filter": scope + [{"range": {"approved_at": {"lte": self.system_t.isoformat()}}}]}}
        on = self.on.isoformat()
        absent = lambda f: {"bool": {"must_not": [{"exists": {"field": f}}]}}  # noqa: E731
        normative = {"bool": {
            "must_not": [{"term": {"doc_type": "record"}}],
            "filter": [
                {"bool": {"should": [absent("valid_from"), {"range": {"valid_from": {"lte": on}}}], "minimum_should_match": 1}},
                {"bool": {"should": [absent("valid_to"), {"range": {"valid_to": {"gt": on}}}], "minimum_should_match": 1}},
            ]}}
        record = {"bool": {"filter": [{"term": {"doc_type": "record"}}, {"range": {"doc_date": {"lte": on}}}]}}
        return {"bool": {"filter": scope + [{"bool": {"should": [normative, record], "minimum_should_match": 1}}]}}


class SearchIndex(Protocol):
    def upsert(self, docs: list[IndexDoc]) -> None: ...
    def update_fields(self, evidence_ids: list[str], fields: dict) -> None: ...
    def delete_version(self, version_id: str) -> None: ...
    def indexed_ids(self, version_id: str) -> set[str]: ...
    def version_ids(self, tenant_id: str) -> set[str]: ...
    def search_text(self, query: str, f: Filters, k: int) -> list[tuple[str, float]]: ...
    def search_vector(self, vector: list[float], f: Filters, k: int) -> list[tuple[str, float]]: ...


def rrf(*rankings: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    """倒数排名融合：只看名次，不看各路分数的量纲。"""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: (-x[1], x[0]))


_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*|[\u4e00-\u9fff]")


def tokens(text: str) -> list[str]:
    """内存实现用的分词：英文与数字按词，中文按单字加相邻二字。生产环境由 OpenSearch 的中文分词器负责。"""
    units = _WORD.findall(text.lower().replace(",", ""))
    out = list(units)
    out += [a + b for a, b in zip(units, units[1:]) if len(a) == 1 and len(b) == 1]
    return out


class MemoryIndex:
    """内存参考实现：BM25 加余弦相似度。用于测试与评测流程，不用于生产。"""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.docs: dict[str, IndexDoc] = {}
        self.k1, self.b = k1, b

    def upsert(self, docs: list[IndexDoc]) -> None:
        for d in docs:
            self.docs[d.evidence_id] = d

    def update_fields(self, evidence_ids: list[str], fields: dict) -> None:
        for i in evidence_ids:
            if i in self.docs:
                self.docs[i] = replace(self.docs[i], **fields)

    def delete_version(self, version_id: str) -> None:
        for i in [i for i, d in self.docs.items() if d.version_id == version_id]:
            del self.docs[i]

    def indexed_ids(self, version_id: str) -> set[str]:
        return {i for i, d in self.docs.items() if d.version_id == version_id}

    def version_ids(self, tenant_id: str) -> set[str]:
        return {d.version_id for d in self.docs.values() if d.tenant_id == tenant_id}

    def search_text(self, query: str, f: Filters, k: int) -> list[tuple[str, float]]:
        pool = [d for d in self.docs.values() if f.matches(d)]
        if not pool:
            return []
        q = Counter(tokens(query))
        toks = {d.evidence_id: Counter(tokens(d.text)) for d in pool}
        avg = sum(sum(t.values()) for t in toks.values()) / len(pool)
        n = len(pool)
        df = Counter(t for c in toks.values() for t in c)
        scores = []
        for d in pool:
            c, length, s = toks[d.evidence_id], sum(toks[d.evidence_id].values()), 0.0
            for t in q:
                if t in c:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * c[t] * (self.k1 + 1) / (c[t] + self.k1 * (1 - self.b + self.b * length / avg))
            if s > 0:
                scores.append((d.evidence_id, s))
        return sorted(scores, key=lambda x: (-x[1], x[0]))[:k]

    def search_vector(self, vector: list[float], f: Filters, k: int) -> list[tuple[str, float]]:
        scored = [(d.evidence_id, sum(a * b for a, b in zip(vector, d.embedding)))
                  for d in self.docs.values() if f.matches(d)]
        return sorted(scored, key=lambda x: (-x[1], x[0]))[:k]
