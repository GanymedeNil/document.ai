"""身份与授权范围（开发文档 10.1、11.3）。

Identity 是身份校验的结果（谁、哪个租户、接口角色）；Scope 是授权接口 resolve 的结果
（租户、可访问的角色、过期时间），只以句柄形式在模块间传递，不进入模型上下文（3.2 第 4 条）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Identity:
    user: str
    tenant_id: str
    roles: frozenset[str]          # 接口角色：editor、reviewer 等，与文档权限无关


@dataclass(frozen=True)
class Scope:
    tenant_id: str
    principals: frozenset[str]     # 文档权限角色（acl_policy.principals）
    expires_at: datetime | None = None
