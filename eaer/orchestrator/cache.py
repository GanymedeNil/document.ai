"""回答缓存（开发文档 10.1 第 7 步、10.2、AC-10-4）。

同一租户、同一授权范围、同一索引构建下，同一天对同一问题（时间参数、模式提示相同）的提问直接复用已完成的回答。
- 键：租户、授权范围哈希（principals 排序后取哈希）、索引构建编号（别名此刻指向的具体索引，重建切换后键自然变化）、
  规范化的问题、提问日、时间参数、模式提示、模型与 Deep、Verified 开关。权限不同的用户永远落在不同的键上；
- 失效：版本状态、权限、已发布的有效期与关系、权限策略、租户配置一有变化，数据库触发器就清除该租户的全部缓存
  （迁移 0006），索引任务完成时也清（迁移 0009，索引跟上之前得到的回答不再沿用）；另有有效期
  （budgets.cache_ttl_seconds，初值 600 秒，0 表示不缓存）；
- 先算后写的竞争：每次清除都把租户的失效序号加一（迁移 0009）。提问开始时读序号（epoch），写缓存时锁住序号行再比较，
  期间发生过失效就不写——否则旧状态算出的回答会在清除之后写进去，一直留到过期；
- 命中后返回前仍按当前授权范围核对所引证据（授权接口 check），有一条不通过就作废这条缓存、重新回答——
  即使失效漏了，也不会把别人的或已撤权的内容交出去；
- 只缓存在同步时限内完成的回答；每次命中照样写一条 query_log（cost 中注明来源）。
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from datetime import date, datetime, timedelta

import psycopg
from psycopg.types.json import Jsonb

from eaer.domain.identity import Scope
from eaer.orchestrator import codec
from eaer.orchestrator.engine import QueryResult


def scope_hash(scope: Scope) -> str:
    return hashlib.sha256("\n".join(sorted(scope.principals)).encode()).hexdigest()


def key(tenant_id: str, scope: Scope, build_id: str, question: str, *, today: date, as_of_valid: date | None,
        as_of_system: date | None, mode_hint: str | None, model: str, deep_enabled: bool,
        verified_enabled: bool = True) -> str:
    parts = [tenant_id, scope_hash(scope), build_id, " ".join(question.split()), today.isoformat(),
             codec.iso(as_of_valid), codec.iso(as_of_system), mode_hint, model, deep_enabled, verified_enabled]
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def _dump(r: QueryResult) -> dict:
    d = asdict(r)
    d.pop("session_id")
    return d


def _load(d: dict, session_id: str) -> QueryResult:
    return QueryResult(**{**d, "session_id": session_id, "claims": codec.claims_in(d["claims"])})


def epoch(conn: psycopg.Connection, tenant_id: str) -> int:
    """租户当前的失效序号；提问开始时（检索之前）读一次，写缓存时交给 put。"""
    conn.execute("INSERT INTO query_cache_epoch (tenant_id) VALUES (%s) ON CONFLICT DO NOTHING", (tenant_id,))
    return conn.execute("SELECT epoch FROM query_cache_epoch WHERE tenant_id = %s", (tenant_id,)).fetchone()["epoch"]


def put(conn: psycopg.Connection, k: str, r: QueryResult, *, tenant_id: str, scope: Scope, build_id: str,
        ttl_seconds: float, source_query: uuid.UUID, now: datetime, since_epoch: int) -> bool:
    """写缓存；提问开始之后发生过失效（序号变了）就不写，返回是否写入。"""
    if ttl_seconds <= 0 or r.status != "completed":
        return False
    with conn.transaction():
        # FOR SHARE：正在提交的失效（它已锁住序号行）先完成，再按它提交后的序号判断
        row = conn.execute("SELECT epoch FROM query_cache_epoch WHERE tenant_id = %s FOR SHARE", (tenant_id,)).fetchone()
        if row is None or row["epoch"] != since_epoch:
            return False
        conn.execute(
            "INSERT INTO query_cache (key, tenant_id, scope_hash, index_build_id, result, source_query, created_at, expires_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (key) DO UPDATE SET result = EXCLUDED.result, "
            "source_query = EXCLUDED.source_query, created_at = EXCLUDED.created_at, expires_at = EXCLUDED.expires_at",
            (k, tenant_id, scope_hash(scope), build_id, Jsonb(_dump(r)), source_query, now,
             now + timedelta(seconds=ttl_seconds)))
    return True


def get(conn: psycopg.Connection, k: str, *, tenant_id: str, scope: Scope, authorizer, now: datetime,
        session_id: str) -> QueryResult | None:
    """取缓存的回答；没有、已过期，或所引证据按当前授权范围核对不通过时返回 None（后者同时作废这条缓存）。"""
    row = conn.execute("SELECT result, source_query FROM query_cache WHERE key = %s AND tenant_id = %s "
                       "AND scope_hash = %s AND expires_at > %s", (k, tenant_id, scope_hash(scope), now)).fetchone()
    if row is None:
        return None
    r = _load(row["result"], session_id)
    ids = list(dict.fromkeys(x.evidence_id for c in r.claims if c.status != "unsupported" for x in c.citations))
    if ids and not all(authorizer.check(scope, ids)):
        conn.execute("DELETE FROM query_cache WHERE key = %s", (k,))
        return None
    r.cost = {"by_kind": {}, "failed": 0, "models": [], "cache": {"source_query": str(row["source_query"])}}
    r.retrieval_rounds, r.seconds = 0, 0.0
    return r
