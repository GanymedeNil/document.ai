"""回答方式：严格模式与宽松模式（迁移 0015，设计变更记录 2026-11-16）。

租户默认为宽松模式（tenant_config.strict_mode = false），管理员可改；提问接口可按次覆盖。两种模式的区别见
eaer/orchestrator/fast.py 的模块说明。
"""
from __future__ import annotations

from datetime import datetime

import psycopg


def tenant_strict(conn: psycopg.Connection, tenant_id: str) -> bool:
    row = conn.execute("SELECT strict_mode FROM tenant_config WHERE tenant_id = %s", (tenant_id,)).fetchone()
    return bool(row["strict_mode"]) if row else False


def set_strict(conn: psycopg.Connection, tenant_id: str, strict: bool, now: datetime) -> bool:
    conn.execute("INSERT INTO tenant_config (tenant_id, strict_mode, updated_at) VALUES (%s, %s, %s) "
                 "ON CONFLICT (tenant_id) DO UPDATE SET strict_mode = EXCLUDED.strict_mode, updated_at = EXCLUDED.updated_at",
                 (tenant_id, strict, now))
    return strict
