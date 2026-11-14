"""AC-4-1：迁移脚本在空库执行成功，回滚脚本可恢复上一版本。
另核对 0005 建的检查点表与 langgraph-checkpoint-postgres 自己的 setup() 建出的完全一致（列、类型、默认值、主键、索引），
且库认为全部迁移步骤已执行（8.4）。"""
from __future__ import annotations

import psycopg
import pytest

from eaer.db import downgrade, migrate

TABLES = {"tenant_config", "acl_policy", "document", "document_version", "version_validity",
          "version_relation", "evidence", "lifecycle_event", "review_task", "ingest_job", "query_log", "evidence_embedding"}
CHECKPOINT_TABLES = {"checkpoint_migrations", "checkpoints", "checkpoint_blobs", "checkpoint_writes"}
TABLES_0005 = {"query_task"} | CHECKPOINT_TABLES
TABLES_0006 = {"query_cache"}
TABLES_0007 = {"model_admission"}
TRIGGERS = {"version_delete_guard", "version_update_guard", "validity_guard", "relation_guard",
            "lifecycle_event_append_only", "forced_verify_append_only"}
TRIGGERS_0005 = {"query_log_guard", "query_task_guard"}
TRIGGERS_0006 = {"cache_on_version", "cache_on_validity", "cache_on_relation", "cache_on_acl_policy", "cache_on_tenant_config"}
TRIGGERS_0007 = {"model_admission_append_only", "cache_on_model_admission"}
TRIGGERS_0008 = {"review_task_tenant_guard"}
TRIGGERS_0009 = {"cache_on_index_job"}
TABLES_0009 = {"query_cache_epoch"}
ALL = (TABLES | TABLES_0005 | TABLES_0006 | TABLES_0007 | TABLES_0009,
       TRIGGERS | TRIGGERS_0005 | TRIGGERS_0006 | TRIGGERS_0007 | TRIGGERS_0008 | TRIGGERS_0009)


def _state(url):
    with psycopg.connect(url) as c:
        tables = {r[0] for r in c.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
        triggers = {r[0] for r in c.execute("SELECT tgname FROM pg_trigger WHERE NOT tgisinternal")}
    return tables - {"alembic_version"}, triggers


def test_upgrade_downgrade_upgrade(empty_db_url):
    full = ALL
    migrate(empty_db_url)
    assert _state(empty_db_url) == full
    downgrade(empty_db_url)
    assert _state(empty_db_url) == (set(), set())
    migrate(empty_db_url)
    assert _state(empty_db_url) == full


def test_each_revision_downgrades_one_step(empty_db_url):
    """逐个版本回滚：0009 去掉索引任务完成时清缓存的触发器；0008 去掉审核任务的租户与其触发器；0007 去掉准入记录；
    0006 去掉回答缓存；0005 去掉查询任务与检查点表；0004 只去掉强制验证清单的触发器。"""
    migrate(empty_db_url)
    downgrade(empty_db_url, "0008")
    assert _state(empty_db_url) == (ALL[0] - TABLES_0009, ALL[1] - TRIGGERS_0009)
    with psycopg.connect(empty_db_url) as c:                    # 0006 的清缓存函数恢复原样（不再引用失效序号）
        assert "eaer_cache_invalidate" not in c.execute(
            "SELECT prosrc FROM pg_proc WHERE proname = 'eaer_cache_clear_for_version'").fetchone()[0]
    downgrade(empty_db_url, "0007")
    assert _state(empty_db_url) == (TABLES | TABLES_0005 | TABLES_0006 | TABLES_0007,
                                    TRIGGERS | TRIGGERS_0005 | TRIGGERS_0006 | TRIGGERS_0007)
    with psycopg.connect(empty_db_url) as c:
        assert c.execute("SELECT count(*) FROM information_schema.columns "
                         "WHERE table_name = 'review_task' AND column_name = 'tenant_id'").fetchone()[0] == 0
    downgrade(empty_db_url, "0006")
    assert _state(empty_db_url) == (TABLES | TABLES_0005 | TABLES_0006, TRIGGERS | TRIGGERS_0005 | TRIGGERS_0006)
    downgrade(empty_db_url, "0005")
    assert _state(empty_db_url) == (TABLES | TABLES_0005, TRIGGERS | TRIGGERS_0005)
    downgrade(empty_db_url, "0004")
    assert _state(empty_db_url) == (TABLES, TRIGGERS)
    downgrade(empty_db_url, "0003")
    assert _state(empty_db_url) == (TABLES, TRIGGERS - {"forced_verify_append_only"})
    migrate(empty_db_url)
    assert _state(empty_db_url) == ALL


def test_0008_backfills_review_task_tenant_without_guessing(empty_db_url):
    """0008 从版本所属文档回填审核任务的租户；迁移前已删除的任务没有依据，保持为空、不推断，
    且之后新写入的任务必须带租户。"""
    from psycopg.rows import dict_row

    from dbutil import seed

    migrate(empty_db_url)
    with psycopg.connect(empty_db_url, autocommit=True, row_factory=dict_row) as c:
        lc, _, ids = seed(c)                                     # t1：v1 已通过、v2 待审核
        doc = lc.create_document("t1", "X", "临时文件")
        lc.delete_pending(lc.add_version(doc, content_hash="h", raw_object_uri="file://x", acl_policy_id=ids["acl"],
                                         parser_version="t"), "bob")
    downgrade(empty_db_url, "0007")
    migrate(empty_db_url)
    with psycopg.connect(empty_db_url, autocommit=True, row_factory=dict_row) as c:
        rows = c.execute("SELECT status, tenant_id FROM review_task ORDER BY status").fetchall()
        assert rows == [{"status": "approved", "tenant_id": "t1"}, {"status": "deleted", "tenant_id": None},
                        {"status": "open", "tenant_id": "t1"}]
        with pytest.raises(psycopg.errors.CheckViolation, match="review_task_tenant_required"):
            c.execute("INSERT INTO review_task (id, status, reviewer, decided_at) "
                      "VALUES (gen_random_uuid(), 'deleted', 'bob', now())")


def _checkpoint_schema(url):
    q = ("SELECT c.table_name, c.column_name, c.data_type, c.is_nullable, c.column_default "
         "FROM information_schema.columns c WHERE c.table_schema = 'public' AND c.table_name = ANY(%s) "
         "ORDER BY 1, 2")
    with psycopg.connect(url) as c:
        cols = c.execute(q, (sorted(CHECKPOINT_TABLES),)).fetchall()
        idx = c.execute("SELECT tablename, indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = ANY(%s) "
                        "ORDER BY 1, 2", (sorted(CHECKPOINT_TABLES),)).fetchall()
        versions = [r[0] for r in c.execute("SELECT v FROM checkpoint_migrations ORDER BY v")]
    return cols, idx, versions


def test_checkpoint_tables_match_the_library(empty_db_url, make_empty_db):
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    migrate(empty_db_url)
    ours = _checkpoint_schema(empty_db_url)
    lib_url = make_empty_db()
    with psycopg.connect(lib_url, autocommit=True, row_factory=dict_row) as c:
        PostgresSaver(c).setup()
    assert ours == _checkpoint_schema(lib_url)
    assert ours[2] == list(range(len(PostgresSaver.MIGRATIONS)))      # setup() 不会再执行任何步骤
