"""模型准入的结论与模式开关（开发文档 12.3、AC-12-3）。

准入检查本身在评测模块里跑（eaer/evalrun/admission.py，客户在自己的环境中运行，数据不出域）；这里只负责：
- record：写入准入结论（model_admission，只追加）与标定出的阈值（tenant_config.thresholds）；
- gate：部署时按租户与生成模型取最近一次准入结论，决定 Verified、Deep 是否开启。没有准入记录的模型两者都关闭
  （“未达标的模式保持关闭”），并给出原因，回答中说明；EAER_ADMISSION=off 可跳过（只用于本地开发）。
准入门槛（DEFAULT_CRITERIA）可由 tenant_config.thresholds 中的同名键覆盖，基线评测后与用户一起回填（16.2）。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import psycopg
from psycopg.types.json import Jsonb

DEFAULT_CRITERIA = {
    "verified_max_false_pass": 0.05,     # 验证器误通过率上限：不该支持的结论被判为支持的比例
    "verified_max_false_reject": 0.30,   # 验证器误拒率上限：该支持的结论没被判为支持的比例（防止“一律未验证”蒙混过关）
    "verified_min_positive": 8,          # 评测集最少的正例、反例数，太少不下结论
    "verified_min_negative": 16,
    "deep_min_accuracy": 0.70,           # Deep 答案正确率下限
    "deep_min_questions": 5,
}


@dataclass
class Gate:
    verified: bool
    deep: bool
    reasons: dict[str, str] = field(default_factory=dict)


def criteria(conn: psycopg.Connection, tenant_id: str) -> dict:
    row = conn.execute("SELECT thresholds FROM tenant_config WHERE tenant_id = %s", (tenant_id,)).fetchone()
    t = (row or {}).get("thresholds") or {}
    return {k: t.get(k, v) for k, v in DEFAULT_CRITERIA.items()}


def latest(conn: psycopg.Connection, tenant_id: str, model_id: str) -> dict | None:
    return conn.execute("SELECT * FROM model_admission WHERE tenant_id = %s AND model_id = %s "
                        "ORDER BY created_at DESC LIMIT 1", (tenant_id, model_id)).fetchone()


def gate(conn: psycopg.Connection, tenant_id: str, model_id: str, *, required: bool = True) -> Gate:
    if not required:
        return Gate(True, True)
    row = latest(conn, tenant_id, model_id)
    if row is None:
        why = f"所用模型 {model_id} 尚未通过准入检查"
        return Gate(False, False, {"verified": why, "deep": why})
    reasons = {}
    if not row["verified_ok"]:
        reasons["verified"] = f"所用模型 {model_id} 未通过 Verified 准入（{row['report'].get('verified', {}).get('reason', '')}）"
    if not row["deep_ok"]:
        reasons["deep"] = f"所用模型 {model_id} 未通过 Deep 准入（{row['report'].get('deep', {}).get('reason', '')}）"
    return Gate(row["verified_ok"], row["deep_ok"], reasons)


def record(conn: psycopg.Connection, tenant_id: str, model_id: str, *, verified_ok: bool, deep_ok: bool, report: dict,
           now: datetime, thresholds: dict | None = None) -> uuid.UUID:
    """写入准入结论；thresholds（标定出的阈值，如 semantic_min_confidence）合并进 tenant_config.thresholds。"""
    aid = uuid.uuid4()
    with conn.transaction():
        conn.execute("INSERT INTO model_admission (id, tenant_id, model_id, verified_ok, deep_ok, report, created_at) "
                     "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                     (aid, tenant_id, model_id, verified_ok, deep_ok, Jsonb(report), now))
        if thresholds:
            conn.execute("INSERT INTO tenant_config (tenant_id, thresholds, updated_at) VALUES (%s, %s, %s) "
                         "ON CONFLICT (tenant_id) DO UPDATE SET thresholds = tenant_config.thresholds || EXCLUDED.thresholds, "
                         "updated_at = EXCLUDED.updated_at", (tenant_id, Jsonb(thresholds), now))
    return aid
