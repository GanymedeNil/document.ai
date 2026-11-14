"""频率限制（开发文档 11.1：429 rate_limited，eaer/api/ratelimit.py、迁移 0010）。接口层面的 429 见 tests/test_contract.py。

- 每个用户每分钟计数，超过上限时给出到下一分钟的秒数；上限 0 表示不限；不同用户、不同接口类别各自计数；
- 新的一分钟顺手删掉自己早于一小时的计数；
- 两个接口实例（各自的 create_app）共用数据库里的计数：集群版横向扩展后限制照样准确。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from eaer.api.app import TokenAuthenticator, create_app
from eaer.api.ratelimit import Limits, hit
from eaer.domain.identity import Identity
from eaer.storage import LocalFileStore

T = datetime(2026, 11, 14, 9, 30, 15, tzinfo=timezone.utc)


def test_counts_per_user_kind_and_minute(conn):
    assert [hit(conn, "t1:a", "query", 2, T) for _ in range(3)] == [None, None, 45]     # 到 9:31:00 还有 45 秒
    assert hit(conn, "t1:b", "query", 2, T) is None                                     # 别的用户
    assert hit(conn, "t1:a", "upload", 2, T) is None                                    # 别的接口
    assert hit(conn, "t1:a", "query", 2, T + timedelta(seconds=40)) == 5                # 9:30:55，同一分钟内仍超
    assert hit(conn, "t1:a", "query", 2, T + timedelta(minutes=1)) is None              # 下一分钟重新计
    assert hit(conn, "t1:a", "query", 0, T) is None and Limits(0, 0).of("upload") == 0  # 0 不限
    assert Limits.from_env({}) == Limits(30, 120)
    assert Limits.from_env({"EAER_RATE_QUERY_PER_MINUTE": "5"}).query == 5


def test_old_windows_are_cleaned_up(conn):
    for k in range(5):
        hit(conn, "t1:a", "query", 10, T - timedelta(hours=3) + timedelta(minutes=k))
    hit(conn, "t1:b", "query", 10, T - timedelta(hours=3))
    hit(conn, "t1:a", "query", 10, T)                                                   # a 的新一分钟
    rows = conn.execute("SELECT subject, count(*) AS n FROM rate_limit_counter GROUP BY subject ORDER BY subject").fetchall()
    assert [(r["subject"], r["n"]) for r in rows] == [("t1:a", 1), ("t1:b", 1)]         # 只清自己的


def test_instances_share_the_count(db_url, tmp_path):
    tokens = {"tok": Identity("u", "t1", frozenset({"editor"}))}
    apps = [TestClient(create_app(db_url, LocalFileStore(tmp_path), TokenAuthenticator(tokens), lambda: T,
                                  rate_limits=Limits(query=10, upload=2))) for _ in range(2)]

    def up(client, k):
        return client.post("/v1/documents", headers={"Authorization": "Bearer tok"}, data={"principals": "R_STAFF"},
                           files={"file": (f"{k}.html", f"<h1>文档{k}</h1><p>正文{k}</p>".encode(), "text/html")})
    assert [up(apps[0], 1).status_code, up(apps[1], 2).status_code, up(apps[0], 3).status_code] == [201, 201, 429]
