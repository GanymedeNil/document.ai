"""AC-7-1（硬性）样例正确性：事实表样例 Q01–Q08 端到端冒烟，全部正确。

走完整的产品链路：按日期回放 docs/事实表样例.xlsx（真实的入库、审核、生命周期与索引代码），每个查询在它的提问日
由编排器回答（Router、Deep、Verified；时间点由编排器从问题中识别，不传参数），用线上生成服务与配置的 Embedding、
重排服务。正确的标准：factset 计算的标准答案中的每个取值都出现在回答里、没有拒答；标准答案要求“未验证”的
（Q07：回溯时只有上传时间），回答带“未验证”。

需要真实的模型服务：EAER_GEN_*、EAER_EMBED_*、EAER_RERANK_*（见 eaer/adapters/online.py；本地开发的配置在
未纳入版本控制的 .env，`set -a; source .env; set +a` 后运行）。三者缺一即跳过——CI 里没有模型服务，
这一项在有模型服务的环境里跑，结果记入 CLAUDE.md。
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from eaer.evalrun.fast import run_engine
from eaer.retrieval.index import MemoryIndex
from eaer.storage import LocalFileStore
from factset import load
from factset.render import Renderer

SAMPLE = Path("docs/事实表样例.xlsx")
IDS = [f"Q0{i}" for i in range(1, 9)]


def _model():
    if not os.environ.get("EAER_GEN_BASE_URL"):
        return None, "未配置生成服务（EAER_GEN_*）"
    from eaer.adapters.online import OnlineModelClient
    m = OnlineModelClient.from_env()
    missing = m.missing_for_fast()
    return (None, f"未配置 {'、'.join(missing)}") if missing else (m, "")


def test_q01_to_q08_end_to_end(conn, tmp_path, monkeypatch):
    model, why = _model()
    if model is None:
        pytest.skip(f"AC-7-1 冒烟需要真实的模型服务：{why}")
    monkeypatch.setenv("EAER_ENV", "eval")
    full = load(SAMPLE)
    fs = replace(full, queries=[q for q in full.queries if q.id in IDS])      # 文档、版本、事件全部保留，按日期回放
    assert [q.id for q in fs.queries] == IDS
    manifest = Renderer(fs).render(tmp_path / "corpus", "docx")
    results = run_engine(conn, fs, tmp_path / "corpus", manifest, model=model, store=LocalFileStore(tmp_path / "files"),
                         index=MemoryIndex(), embedder=model.embedder(), reranker=model)
    assert [r.query_id for r in results] == IDS
    bad = [r for r in results if r.outcome != "correct" or (r.unverified_expected and not r.unverified_labelled)]
    assert not bad, "\n".join(f"{r.query_id}（{r.outcome}，路径 {r.mode_path}）预期：{r.expected}\n  回答：{r.answer[:300]}"
                              for r in bad)
    q07 = next(r for r in results if r.query_id == "Q07")
    assert q07.unverified_expected and q07.unverified_labelled
