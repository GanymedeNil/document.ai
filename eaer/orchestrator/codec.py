"""查询状态与检查点之间的转换（开发文档 8.4）。

图的状态每个节点之后写入 PostgreSQL 检查点。状态里只放 JSON 能表示的值（字符串、数字、列表、字典），
不依赖检查点序列化器对自定义类的支持：库升级或类改名都不会让旧检查点无法读出，检查点内容也可以直接查看。
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import date, datetime

from eaer.adapters.model import Usage
from eaer.adapters.online import CallRecord
from eaer.orchestrator.fast import Citation, ClaimOut, Draft
from eaer.orchestrator.timeparse import TimeBasis
from eaer.retrieval.service import Hit
from eaer.verifier.deterministic import Target


def iso(d: date | datetime | None) -> str | None:
    return d.isoformat() if d is not None else None


def day(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def hits_out(hits: list[Hit]) -> list[dict]:
    return [asdict(h) for h in hits]


def hits_in(rows: list[dict] | None) -> list[Hit]:
    return [Hit(**r) for r in rows or []]


def meta_out(meta: dict) -> dict:
    """Steps.meta 的结果：日期转成 ISO 字符串（evidence_block 与信号判断按字符串使用同样成立）。"""
    return {k: {f: (v.isoformat() if isinstance(v, (date, datetime)) else str(v) if isinstance(v, uuid.UUID) else v)
                for f, v in row.items()} for k, row in meta.items()}


def draft_out(d: Draft | None) -> dict | None:
    return None if d is None else asdict(d)


def draft_in(d: dict | None) -> Draft | None:
    if d is None:
        return None
    return Draft(d["claims"], d["abstain"], d["missing"], d["conflicts"], d["model"], Usage(**d["usage"]))


def claims_out(claims: list[ClaimOut]) -> list[dict]:
    return [asdict(c) for c in claims]


def claims_in(rows: list[dict] | None) -> list[ClaimOut]:
    return [ClaimOut(**{**r, "citations": [Citation(**c) for c in r["citations"]]}) for r in rows or []]


def tb_in(d: dict) -> TimeBasis:
    return TimeBasis(day(d["as_of_valid"]), day(d["as_of_system"]), d["source"], list(d["unresolved"]))


def targets_out(vt: dict[str, Target]) -> dict:
    return {k: {"on": iso(t.on), "system_t": iso(t.system_t)} for k, t in vt.items()}


def targets_in(d: dict | None) -> dict[str, Target]:
    return {k: Target(day(v["on"]), datetime.fromisoformat(v["system_t"]) if v["system_t"] else None)
            for k, v in (d or {}).items()}


def calls_out(calls: list[CallRecord]) -> list[dict]:
    return [asdict(c) for c in calls]


def calls_in(rows: list[dict] | None) -> list[CallRecord]:
    return [CallRecord(**r) for r in rows or []]
