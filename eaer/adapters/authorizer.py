"""授权接口与参考实现（开发文档 10.1、11.3）。

resolve：身份 -> 授权范围（租户、可访问角色、过期时间）。不认识的身份抛 AuthorizationError，
授权服务不可用抛 AuthorizerUnavailable；调用方遇到两者都必须拒绝请求，不能退化为空范围或宽范围（AC-10-5）。
check：返回前在数据库中逐条核对证据（存在、同租户、版本可搜索、角色有交集）；不存在与无权一律为 False，
不透露存在性（AC-10-6）。范围过期抛 ScopeExpired。

参考实现 LocalAuthorizer：本地账号与角色，配置文件（JSON）维护：
    {"accounts": {"alice": {"tenant": "t1", "principals": ["R_STAFF"]}}, "ttl_seconds": 900}
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Protocol

import psycopg

from eaer.domain.identity import Identity, Scope


class AuthorizationError(Exception):
    """身份无法解析为授权范围。"""


class AuthorizerUnavailable(Exception):
    """授权服务不可用。调用方必须拒绝请求。"""


class ScopeExpired(Exception):
    pass


class Authorizer(Protocol):
    def resolve(self, identity: Identity) -> Scope: ...
    def check(self, scope: Scope, evidence_ids: list[str]) -> list[bool]: ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def check_in_database(conn: psycopg.Connection, scope: Scope, evidence_ids: list[str]) -> list[bool]:
    """各实现共用的数据库核对：证据存在、同租户、版本可搜索、角色有交集。"""
    if not evidence_ids:
        return []
    ok = {r["id"] for r in conn.execute(
        "SELECT e.id FROM evidence e JOIN document_version v ON v.id = e.version_id "
        "JOIN document d ON d.id = v.document_id JOIN acl_policy p ON p.id = v.acl_policy_id "
        "WHERE e.id = ANY(%s) AND d.tenant_id = %s AND v.status = 'searchable' AND p.principals && %s",
        (list(evidence_ids), scope.tenant_id, sorted(scope.principals)))}
    return [i in ok for i in evidence_ids]


class LocalAuthorizer:
    def __init__(self, conn: psycopg.Connection, accounts: dict[str, dict], *, ttl_seconds: int = 900,
                 clock: Callable[[], datetime] = _utcnow):
        self.conn, self.accounts, self.ttl, self.clock = conn, accounts, timedelta(seconds=ttl_seconds), clock

    @classmethod
    def from_file(cls, conn: psycopg.Connection, path: str | Path, **kw) -> "LocalAuthorizer":
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(conn, cfg["accounts"], ttl_seconds=cfg.get("ttl_seconds", 900), **kw)

    def resolve(self, identity: Identity) -> Scope:
        acc = self.accounts.get(identity.user)
        if acc is None or acc.get("tenant") != identity.tenant_id or not acc.get("principals"):
            raise AuthorizationError("无法确定授权范围")
        return Scope(identity.tenant_id, frozenset(acc["principals"]), self.clock() + self.ttl)

    def check(self, scope: Scope, evidence_ids: list[str]) -> list[bool]:
        if scope.expires_at is not None and scope.expires_at <= self.clock():
            raise ScopeExpired("授权范围已过期，请重新解析")
        return check_in_database(self.conn, scope, evidence_ids)
