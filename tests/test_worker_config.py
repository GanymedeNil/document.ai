"""后台任务进程的配置（eaer/ingestion/__main__.py）：处理索引任务所需的配置缺一不可，缺少即启动报错；
进程的分词方式必须与索引记录的一致。涉及真实 OpenSearch 的用例在未设置 EAER_TEST_OPENSEARCH_URL 时跳过。"""
from __future__ import annotations

import os
import uuid

import pytest

from eaer.ingestion.__main__ import ConfigError, build_worker, main
from eaer.storage import LocalFileStore

OS_URL = os.environ.get("EAER_TEST_OPENSEARCH_URL")
EMBED = {"EAER_EMBED_BASE_URL": "http://127.0.0.1:1/v1", "EAER_EMBED_API_KEY": "k",
         "EAER_EMBED_MODEL": "Qwen3-Embedding-0.6B", "EAER_EMBED_DIMS": "1024"}


@pytest.mark.parametrize("env, message", [
    ({}, "EAER_OPENSEARCH_URL"),
    ({"EAER_OPENSEARCH_URL": "http://127.0.0.1:1"}, "Embedding"),
])
def test_missing_index_config_is_an_error(conn, tmp_path, env, message):
    with pytest.raises(ConfigError, match=message):
        build_worker(conn, LocalFileStore(tmp_path), env)


def test_parse_only_worker_skips_index_jobs(conn, tmp_path):
    w = build_worker(conn, LocalFileStore(tmp_path), {}, with_index=False)
    assert set(w.handlers) == {"parse_text", "classify_extract", "parse_deep"}


def test_main_reports_config_errors(monkeypatch, capsys, db_url, tmp_path):
    monkeypatch.setenv("EAER_DATABASE_URL", db_url)
    monkeypatch.setenv("EAER_STORAGE_DIR", str(tmp_path))
    monkeypatch.delenv("EAER_OPENSEARCH_URL", raising=False)
    assert main([]) == 2 and "配置错误" in capsys.readouterr().err
    assert main(["rebuild"]) == 2


@pytest.mark.skipif(not OS_URL, reason="未设置 EAER_TEST_OPENSEARCH_URL")
def test_worker_requires_an_index_with_the_same_text_analysis(conn, tmp_path):
    from eaer.ingestion.indexer import fresh_index
    from eaer.retrieval.opensearch import connect

    alias = f"eaer-wk-{uuid.uuid4().hex[:8]}"
    env = {**EMBED, "EAER_OPENSEARCH_URL": OS_URL, "EAER_INDEX_ALIAS": alias}
    with pytest.raises(ConfigError, match="rebuild"):
        build_worker(conn, LocalFileStore(tmp_path), env)                  # 别名还不存在
    with fresh_index(connect(OS_URL), alias, build_id="b7", dims=1024):          # 默认分词方式 standard
        w = build_worker(conn, LocalFileStore(tmp_path), env)
        assert {"publish_index", "index_update"} <= set(w.handlers)
        assert w.handlers["publish_index"].__self__.build_id == "b7"
        with pytest.raises(ConfigError, match="分词方式"):
            build_worker(conn, LocalFileStore(tmp_path), {**env, "EAER_TEXT_ANALYZER": "jieba"})


# ---------- 后台执行提问（8.5）：与接口进程相同的查询配置 ----------

def test_query_runner_config(tmp_path):
    from eaer.runtime import query_runner

    base = {"EAER_DATABASE_URL": "postgresql://x/y"}
    assert query_runner({**base, "EAER_DISABLE_QUERY": "1"}, LocalFileStore(tmp_path)) is None
    with pytest.raises(ConfigError, match="EAER_TOKENS_FILE"):
        query_runner(base, LocalFileStore(tmp_path))
    tokens = tmp_path / "tokens.json"
    tokens.write_text('{"t": {"user": "a", "tenant": "t1", "roles": [], "principals": ["R"]}}', encoding="utf-8")
    with pytest.raises(ConfigError, match="提问服务缺少配置"):
        query_runner({**base, "EAER_TOKENS_FILE": str(tokens)}, LocalFileStore(tmp_path))
