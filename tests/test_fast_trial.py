"""Fast 试跑运行器（eaer/evalrun/fast.py）的流程与计分，离线运行：用一个总是拒答的模型。

只验证回放、计分与报表：拒答类计为 refused_ok、应答类计为 wrong、验证器类单列为 not_applicable。
真实生成服务的试跑见 `python -m eaer.evalrun.fast`（需要 EAER_GEN_* 环境变量）。
"""
from __future__ import annotations

from eaer.adapters.model import Capabilities, Completion, Usage
from eaer.evalrun.fast import report, run_fast
from eaer.retrieval.embed import HashingEmbedder
from eaer.retrieval.index import MemoryIndex
from eaer.storage import LocalFileStore
from factset import load
from factset.render import Renderer
from fixture import sample, write


class AlwaysAbstain:
    def capabilities(self):
        return Capabilities(True, 32000, False, 0)

    def generate(self, messages, schema):
        return Completion("{}", {"abstain": True, "claims": [], "missing": "测试"}, Usage(10, 2), "always-abstain")


def test_trial_runner_scores_and_reports(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("EAER_ENV", "eval")
    fs = load(write(sample(), tmp_path / "s.xlsx"))
    manifest = Renderer(fs).render(tmp_path / "corpus", "html")
    results = run_fast(conn, fs, tmp_path / "corpus", manifest, model=AlwaysAbstain(),
                       store=LocalFileStore(tmp_path / "files"), index=MemoryIndex(), embedder=HashingEmbedder())
    by = {r.query_id: r.outcome for r in results}
    assert {q for q, o in by.items() if o == "refused_ok"} == {"Q09", "Q12", "Q14"}
    assert by["Q16"] == "not_applicable"
    assert all(o == "wrong" for q, o in by.items() if q not in {"Q09", "Q12", "Q14", "Q16"})
    text = report(results, model="always-abstain", embedder="hashing")
    assert "正确 3 / 15" in text and "| Q16 | 验证器 | not_applicable |" in text


def test_engine_runner_records_mode_paths(conn, tmp_path, monkeypatch):
    from eaer.evalrun.fast import run_engine
    monkeypatch.setenv("EAER_ENV", "eval")
    fs = load(write(sample(), tmp_path / "s.xlsx"))
    manifest = Renderer(fs).render(tmp_path / "corpus", "html")
    results = run_engine(conn, fs, tmp_path / "corpus", manifest, model=AlwaysAbstain(),
                         store=LocalFileStore(tmp_path / "files"), index=MemoryIndex(), embedder=HashingEmbedder())
    by = {r.query_id: r for r in results}
    assert by["Q02"].router == "verified" and by["Q02"].mode_path == ["verified"]          # 时间点从问题中识别
    assert by["Q05"].router == "deep" and by["Q15"].mode_path == ["fast"]
    assert {q for q, r in by.items() if r.outcome == "refused_ok"} == {"Q09", "Q12", "Q14"}
    assert "编排器试跑" in report(results, model="always-abstain", embedder="hashing")
