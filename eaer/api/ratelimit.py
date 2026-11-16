"""频率限制（开发文档 11.1：429 rate_limited）。

每个用户（按“租户:用户”）每分钟的提问与上传次数有上限，超过时返回 429，响应头 Retry-After 给出到下一分钟的秒数。
计数在 PostgreSQL（迁移 0010）：多个接口实例共用，横向扩展后限制照样准确。上限来自部署环境变量：
  EAER_RATE_QUERY_PER_MINUTE   每个用户每分钟最多提问次数，默认 30；0 表示不限
  EAER_RATE_UPLOAD_PER_MINUTE  每个用户每分钟最多上传次数，默认 120；0 表示不限
只限制会花钱或占资源的两个接口（提问调用模型，上传触发解析与索引）；查看会话、证据、审核不限。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import psycopg


@dataclass(frozen=True)
class Limits:
    query: int = 30
    upload: int = 120

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "Limits":
        return cls(int(env.get("EAER_RATE_QUERY_PER_MINUTE", "30")), int(env.get("EAER_RATE_UPLOAD_PER_MINUTE", "120")))

    def of(self, kind: str) -> int:
        return {"query": self.query, "upload": self.upload}[kind]


def hit(conn: psycopg.Connection, subject: str, kind: str, limit: int, now: datetime) -> int | None:
    """记一次；超过上限时返回需要等待的秒数（到下一分钟），否则返回 None。limit 为 0 时不限。"""
    if limit <= 0:
        return None
    window = now.replace(second=0, microsecond=0)
    row = conn.execute(
        "INSERT INTO rate_limit_counter (subject, kind, window_start, count) VALUES (%s, %s, %s, 1) "
        "ON CONFLICT (subject, kind, window_start) DO UPDATE SET count = rate_limit_counter.count + 1 RETURNING count",
        (subject, kind, window)).fetchone()
    count = row["count"] if isinstance(row, dict) else row[0]
    if count == 1:                                # 新的一分钟：删掉自己早于一小时的计数
        conn.execute("DELETE FROM rate_limit_counter WHERE subject = %s AND window_start < %s",
                     (subject, window - timedelta(hours=1)))
    if count <= limit:
        return None
    return max(1, math.ceil((window + timedelta(minutes=1) - now).total_seconds()))
