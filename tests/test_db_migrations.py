"""AC-4-1：迁移脚本在空库执行成功，回滚脚本可恢复上一版本。"""
from __future__ import annotations

import psycopg

from eaer.db import downgrade, migrate

TABLES = {"tenant_config", "acl_policy", "document", "document_version", "version_validity",
          "version_relation", "evidence", "lifecycle_event", "review_task", "ingest_job", "query_log", "evidence_embedding"}
TRIGGERS = {"version_delete_guard", "version_update_guard", "validity_guard", "relation_guard",
            "lifecycle_event_append_only"}


def _state(url):
    with psycopg.connect(url) as c:
        tables = {r[0] for r in c.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
        triggers = {r[0] for r in c.execute("SELECT tgname FROM pg_trigger WHERE NOT tgisinternal")}
    return tables - {"alembic_version"}, triggers


def test_upgrade_downgrade_upgrade(empty_db_url):
    migrate(empty_db_url)
    assert _state(empty_db_url) == (TABLES, TRIGGERS)
    downgrade(empty_db_url)
    assert _state(empty_db_url) == (set(), set())
    migrate(empty_db_url)
    assert _state(empty_db_url) == (TABLES, TRIGGERS)
