"""对账补写（开发文档 5.4、6.2）：把对账发现的索引与数据库不一致交回索引任务修复。

对账（eaer/retrieval/reconcile.py）只报告差异。索引任务用完全部尝试次数（模型服务长时间过载、OpenSearch 不可用）后
标为 failed，之后不再有什么会补写：版本可搜索、索引里却没有它（2026-11-15 MultiHop-RAG 重跑），或者已归档、撤权的版本
留在索引里白白占检索名额（结果的正确仍由检索后的数据库核对保证，5.2）。这里按差异补任务：

- missing_in_index（可搜索、索引里没有）与 unexpected_in_index（不可搜索、索引里还有），数据库里有这个版本的：补一个
  index_update（origin = 'reconcile'）。索引任务按执行时数据库的状态对齐（indexer.py），两种差异是同一种修法；对账与
  执行之间状态又变了也不会改错。该版本已有排队中（含退避中）或执行中的 publish_index / index_update 时不再补：
  差异多半就是它还没做完。
- 数据库里已经没有的版本（待审核时被删除，或索引来自别的库）：ingest_job 引用版本，排不了任务，直接从索引删除；
  版本 id 不复用，删除不会与之后的变化冲突。

后台任务进程每 EAER_RECONCILE_SECONDS 秒（默认 3600，与 Trace 清理同一个节奏）对每个租户做一次；几个后台进程同时
运行时用咨询锁，同一时刻只有一个在做。手动：python -m eaer.ingestion reconcile [--repair]。
有差异时写 WARNING 日志、每个租户记一个 span（只有数量）；补的任务带 origin，`python -m eaer.metrics` 的
index_repair 据此统计补了多少、成没成（eaer/ingestion/stats.py）。

一个总也写不进去的版本（例如内容让 Embedding 服务每次都报错）每轮对账会补一个任务、照样失败：失败数在监控里逐轮增加，
需要人工处理；这里不设上限，免得把真正的故障藏起来。
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import psycopg

from eaer.domain.jobs import PRIORITY
from eaer.ingestion.stats import INDEX_STAGES
from eaer.lifecycle.service import utcnow
from eaer.observability import set_attrs, span
from eaer.retrieval.index import SearchIndex
from eaer.retrieval.reconcile import reconcile

log = logging.getLogger("eaer.ingestion.repair")

_LOCK = "hashtext('eaer.reconcile_repair')"


@dataclass
class Repair:
    tenant_id: str
    missing_in_index: list[str]
    unexpected_in_index: list[str]
    enqueued: list[str] = field(default_factory=list)       # 补了 index_update 的版本
    pending: list[str] = field(default_factory=list)        # 已有排队中或执行中的索引任务，没有再补
    deleted: list[str] = field(default_factory=list)        # 数据库里没有、直接从索引删除的版本

    @property
    def clean(self) -> bool:
        return not (self.missing_in_index or self.unexpected_in_index)

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in ("missing_in_index", "unexpected_in_index", "enqueued", "pending", "deleted")}

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.to_json().items()}


def repair(conn: psycopg.Connection, index: SearchIndex, tenant_id: str, *,
           clock: Callable[[], datetime] = utcnow) -> Repair:
    """对一个租户对账并补任务（见模块说明）。调用方负责互斥（repair_all 用咨询锁）。"""
    diff = reconcile(conn, index, tenant_id)
    out = Repair(tenant_id, sorted(diff["missing_in_index"]), sorted(diff["unexpected_in_index"]))
    with span("eaer.ingest.reconcile") as sp:
        for vid in sorted(diff["missing_in_index"] | diff["unexpected_in_index"]):
            key = _uuid(vid)
            with conn.transaction():
                exists = key is not None and conn.execute(
                    "SELECT 1 FROM document_version WHERE id = %s FOR SHARE", (key,)).fetchone() is not None
                if not exists:
                    index.delete_version(vid)
                    out.deleted.append(vid)
                    continue
                if conn.execute("SELECT 1 FROM ingest_job WHERE version_id = %s AND stage = ANY(%s) "
                                "AND status IN ('queued', 'running') LIMIT 1", (key, list(INDEX_STAGES))).fetchone():
                    out.pending.append(vid)
                    continue
                now = clock()
                conn.execute(
                    "INSERT INTO ingest_job (id, version_id, stage, status, priority, origin, created_at, updated_at) "
                    "VALUES (%s, %s, 'index_update', 'queued', %s, 'reconcile', %s, %s)",
                    (uuid.uuid4(), key, PRIORITY["index_update"], now, now))
                out.enqueued.append(vid)
        set_attrs(sp, {f"eaer.reconcile.{k}": n for k, n in out.counts().items()})
    if not out.clean:
        log.warning("索引对账（租户 %s）：索引缺 %d 个版本、多 %d 个版本；补排 index_update %d 个，已有索引任务在做 %d 个，"
                    "数据库里没有、直接从索引删除 %d 个。版本：%s", tenant_id, len(out.missing_in_index),
                    len(out.unexpected_in_index), len(out.enqueued), len(out.pending), len(out.deleted),
                    _sample(out.missing_in_index + out.unexpected_in_index))
    return out


def repair_all(conn: psycopg.Connection, index: SearchIndex, *, tenants: list[str] | None = None,
               clock: Callable[[], datetime] = utcnow) -> list[Repair] | None:
    """对每个租户（或给定的几个）做 repair。另一个进程正在做时返回 None、什么都不做。"""
    if not conn.execute(f"SELECT pg_try_advisory_lock({_LOCK}) AS ok").fetchone()["ok"]:
        log.info("另一个进程正在做索引对账，本轮跳过")
        return None
    try:
        if tenants is None:
            tenants = [r["tenant_id"] for r in conn.execute("SELECT tenant_id FROM tenant_config ORDER BY tenant_id")]
        return [repair(conn, index, t, clock=clock) for t in tenants]
    finally:
        conn.execute(f"SELECT pg_advisory_unlock({_LOCK})")


def _uuid(vid: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(vid)
    except ValueError:
        return None


def _sample(ids: list[str], n: int = 10) -> str:
    return "、".join(ids[:n]) + (f" 等 {len(ids)} 个" if len(ids) > n else "")
