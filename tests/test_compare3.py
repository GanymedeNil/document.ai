"""AC-8-11 三方对比的运行器（eaer/evalrun/compare3.py）：三种执行方式确实按各自的路径执行，报告齐全。
模型服务用 OpenAI 兼容的替身（走线上客户端的代码），数字只说明流程；真实模型上的报告见 docs/三方对比报告.md。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from eaer.adapters.online import OnlineModelClient
from eaer.evalrun.compare3 import CONFIGS, report, run_config, summarize_config
from eaer.retrieval.index import MemoryIndex
from eaer.storage import LocalFileStore
from factset import load
from factset.render import Renderer
from stubmodel import StubModelServer

IDS = ["Q01", "Q02", "Q05", "Q07", "Q09", "Q16"]        # Q16 是“验证器”类：三种方式都不计


def test_three_configurations_take_their_paths(make_db, tmp_path, monkeypatch):
    import psycopg
    from psycopg.rows import dict_row

    monkeypatch.setenv("EAER_ENV", "eval")
    full = load(Path("docs/事实表样例.xlsx"))
    fs = replace(full, queries=[q for q in full.queries if q.id in IDS])
    stats = {}
    with StubModelServer() as stub:
        model = OnlineModelClient.from_env(stub.env())
        for key, _ in CONFIGS:
            manifest = Renderer(fs).render(tmp_path / key / "corpus", "html")
            conn = psycopg.connect(make_db(), autocommit=True, row_factory=dict_row)
            rows = run_config(key, conn, fs, tmp_path / key / "corpus", manifest, model=model,
                              store=LocalFileStore(tmp_path / key / "files"), index=MemoryIndex(),
                              embedder=model.embedder(), reranker=model)
            conn.close()
            paths = [r.result.mode_path for r in rows if r.result.type != "验证器"]
            assert len(paths) == len(IDS) - 1 and all(paths), (key, paths)
            if key == "hybrid":
                assert all(p == ["fast"] for p in paths)
            elif key == "agent":
                assert all(p[0] == "deep" and p[-1] == "verified" for p in paths), paths
            else:
                assert any(p[0] == "fast" for p in paths)
            assert all(r.result.tokens > 0 for r in rows if r.result.outcome != "error" and r.result.type != "验证器")
            stats[key] = summarize_config(rows)
            assert stats[key]["n"] == len(IDS) - 1 and "" not in stats[key]["paths"]      # 分母一致
    text = report(stats, model="stub-gen", embedder="stub-embed", reranker="stub-rerank", dataset="事实表样例")
    assert all(t in text for _, t in CONFIGS) and "按查询类型的正确数" in text
