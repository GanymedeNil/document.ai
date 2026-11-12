"""强制验证清单（开发文档 8.1、11.2、AC-11-5）。

内置规则总是生效、不可关闭：答案引用规范类文档；问题带时间点或版本用语、或接口传入了时间参数。
租户只能追加规则，不能删除或缩小（服务层只提供追加；数据库触发器 forced_verify_append_only 是第二道防线）。
规则每次判断时从数据库读取、不缓存，所以追加后下一次提问立即生效。

租户规则的种类：
  keyword   问题包含该词
  document  答案引用该文档（值为文档 id）
  doc_type  答案引用该类型的文档（normative / record）
  all       所有问题
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

KINDS = {"keyword", "document", "doc_type", "all"}
BUILTIN = [
    {"id": "builtin-normative", "kind": "builtin", "description": "答案引用规范类文档"},
    {"id": "builtin-time-version", "kind": "builtin",
     "description": "问题含明确日期、时间点或版本用语（截至、当时、修订前、旧版、第 N 版等），或接口传入了时间参数"},
]
_TIME_OR_VERSION = re.compile(
    r"\d{4}\s*年|\d{4}-\d{1,2}(?:-\d{1,2})?|\d{1,2}\s*月\s*\d{1,2}\s*[日号]|截至|截止|当时|修订前|修订后|旧版|新版|"
    r"历史版本|之前的版本|第\s*[0-9一二三四五六七八九十]+\s*版")


class ForcedVerifyError(Exception):
    """规则不合法，对应接口的 400。"""


@dataclass(frozen=True)
class Cited:
    document_id: str
    doc_type: str | None


def mentions_time_or_version(question: str) -> bool:
    return bool(_TIME_OR_VERSION.search(question))


def tenant_rules(conn: psycopg.Connection, tenant_id: str) -> list[dict]:
    row = conn.execute("SELECT forced_verify_rules FROM tenant_config WHERE tenant_id = %s", (tenant_id,)).fetchone()
    return list(row["forced_verify_rules"]) if row else []


def add_rule(conn: psycopg.Connection, tenant_id: str, kind: str, value: Any, actor: str, now: datetime,
             description: str | None = None) -> tuple[dict, bool]:
    """追加一条规则；同种类同取值的规则已存在时原样返回。返回 (规则, 是否新增)。"""
    if kind not in KINDS:
        raise ForcedVerifyError(f"kind 只能是 {'、'.join(sorted(KINDS))}")
    if kind == "all":
        value = None
    elif not isinstance(value, str) or not value.strip() or len(value) > 100:
        raise ForcedVerifyError("value 必须是 1–100 个字符的字符串")
    else:
        value = value.strip()
    if kind == "doc_type" and value not in ("normative", "record"):
        raise ForcedVerifyError("doc_type 规则的 value 只能是 normative 或 record")
    if kind == "document":
        try:
            did = uuid.UUID(value)
        except ValueError:
            raise ForcedVerifyError("document 规则的 value 必须是文档 id")
        if conn.execute("SELECT 1 FROM document WHERE id = %s AND tenant_id = %s", (did, tenant_id)).fetchone() is None:
            raise ForcedVerifyError("文档不存在")
        value = str(did)
    with conn.transaction():
        conn.execute("INSERT INTO tenant_config (tenant_id, updated_at) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                     (tenant_id, now))
        rules = conn.execute("SELECT forced_verify_rules FROM tenant_config WHERE tenant_id = %s FOR UPDATE",
                             (tenant_id,)).fetchone()["forced_verify_rules"]
        for r in rules:
            if r["kind"] == kind and r.get("value") == value:
                return r, False
        rule = {"id": f"fv_{uuid.uuid4().hex[:12]}", "kind": kind, "value": value, "description": description,
                "created_by": actor, "created_at": now.isoformat()}
        conn.execute("UPDATE tenant_config SET forced_verify_rules = forced_verify_rules || %s, updated_at = %s "
                     "WHERE tenant_id = %s", (Jsonb([rule]), now, tenant_id))
        return rule, True


def must_verify(conn: psycopg.Connection, tenant_id: str, *, question: str, time_params: bool = False,
                cited: list[Cited] = ()) -> list[str]:
    """返回命中的规则 id；非空即必须走 Verified（AC-8-1）。检索前（cited 为空）与检索后各判断一次（8.1）。"""
    hit: list[str] = []
    if any(c.doc_type == "normative" for c in cited):
        hit.append("builtin-normative")
    if time_params or mentions_time_or_version(question):
        hit.append("builtin-time-version")
    for r in tenant_rules(conn, tenant_id):
        k, v = r["kind"], r.get("value")
        if (k == "all" or (k == "keyword" and v in question)
                or (k == "document" and any(c.document_id == v for c in cited))
                or (k == "doc_type" and any(c.doc_type == v for c in cited))):
            hit.append(r["id"])
    return hit
