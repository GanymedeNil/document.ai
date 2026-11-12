"""按系统时间查询当时的认知（开发文档 7.4 的回溯查询）。

t 为时间点（含）。按日期回溯时用 end_of_day(日期)，表示“截至该日结束”。
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, time, timezone

import psycopg


def end_of_day(d: date) -> datetime:
    return datetime.combine(d, time.max, tzinfo=timezone.utc)


def validity_at(conn: psycopg.Connection, version_id: uuid.UUID, t: datetime) -> dict | None:
    """某版本在系统时间 t 时记录的有效期；t 时尚未发布则为 None。"""
    return conn.execute(
        "SELECT valid_from, valid_from_grade, valid_to, valid_to_grade, valid_to_reason "
        "FROM version_validity WHERE version_id = %s AND recorded_at IS NOT NULL AND recorded_at <= %s "
        "AND (replaced_at IS NULL OR replaced_at > %s)", (version_id, t, t)).fetchone()


def supersedes_at(conn: psycopg.Connection, t: datetime) -> list[tuple[uuid.UUID, uuid.UUID]]:
    """t 时已知的取代关系 (新版本, 旧版本)。"""
    rows = conn.execute(
        "SELECT from_version_id, to_version_id FROM version_relation WHERE relation = 'supersedes' "
        "AND recorded_at IS NOT NULL AND recorded_at <= %s AND (retracted_at IS NULL OR retracted_at > %s)",
        (t, t)).fetchall()
    return [(r["from_version_id"], r["to_version_id"]) for r in rows]


def known_versions(conn: psycopg.Connection, tenant_id: str, t: datetime) -> set[uuid.UUID]:
    """t 时已审核通过的版本（回溯查询只看这些）。"""
    rows = conn.execute(
        "SELECT v.id FROM document_version v JOIN document d ON d.id = v.document_id "
        "WHERE d.tenant_id = %s AND v.approved_at IS NOT NULL AND v.approved_at <= %s", (tenant_id, t)).fetchall()
    return {r["id"] for r in rows}
