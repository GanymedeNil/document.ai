"""数据库测试夹具。

需要一个可用的 PostgreSQL（生产锁定 18.6）。管理连接由 EAER_TEST_ADMIN_URL 指定，
默认连本机 /tmp 下的套接字；连不上时数据库相关测试跳过，事实表工具的测试照常运行。
每个测试使用从已迁移模板复制出的独立数据库，测试之间互不影响。
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ADMIN_URL = os.environ.get("EAER_TEST_ADMIN_URL", "postgresql://postgres@/postgres?host=/tmp")


def url_for(dbname: str) -> str:
    p = urlsplit(ADMIN_URL)
    return urlunsplit((p.scheme, p.netloc, "/" + dbname, p.query, p.fragment))


def _admin():
    try:
        return psycopg.connect(ADMIN_URL, autocommit=True)
    except psycopg.OperationalError as e:
        pytest.skip(f"没有可用的 PostgreSQL：{e}")


@pytest.fixture(scope="session")
def template_db():
    from eaer.db import migrate

    admin = _admin()
    name = f"eaer_tpl_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    migrate(url_for(name))
    yield name
    admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.close()


@pytest.fixture
def make_db(template_db):
    """返回一个函数，每次调用创建一个已迁移的新数据库并返回其连接串。"""
    admin = _admin()
    created = []

    def make() -> str:
        name = f"eaer_t_{uuid.uuid4().hex[:8]}"
        admin.execute(f'CREATE DATABASE "{name}" TEMPLATE "{template_db}"')
        created.append(name)
        return url_for(name)

    yield make
    for name in created:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.close()


@pytest.fixture
def db_url(make_db) -> str:
    return make_db()


@pytest.fixture
def conn(db_url):
    c = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    yield c
    c.close()


@pytest.fixture
def empty_db_url():
    """未迁移的空数据库，用于测试迁移本身。"""
    admin = _admin()
    name = f"eaer_e_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    yield url_for(name)
    admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.close()
