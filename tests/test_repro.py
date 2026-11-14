"""AC-14-5 可复现（开发文档 14.5）：同一模型、索引、评测集版本连续评测两次，指标差异在随机波动范围内。

- 判定（eaer/evalrun/repro.py）：两次的运行配置必须完全相同，否则不比；逐条配对，McNemar 精确检验的双侧 p ≥ 0.05
  为在随机波动范围内；
- 端到端：`python -m eaer.evalrun.repro run` 连续两次经完整编排器回答事实表样例（模型服务用 OpenAI 兼容的替身，
  走与线上相同的客户端代码），两次逐条相同，判定为可复现。真实模型上的两次结果记入 CLAUDE.md。
"""
from __future__ import annotations

import copy
import json
import os

import pytest

from eaer.evalrun.repro import ConfigMismatch, compare, mcnemar_exact, report
from stubmodel import StubModelServer

CFG = {"engine": True, "model": "m", "temperature": "0", "thinking": "disabled", "embedder": "e", "reranker": "r",
       "index": "MemoryIndex", "format": "html", "split": None, "per_type": None,
       "dataset": {"name": "自建合成集", "modified": False, "detail": "", "sha256": "ab" * 32}}


def run(outcomes: list[str]) -> dict:
    return {"config": copy.deepcopy(CFG), "results": [
        {"query_id": f"Q{i:03d}", "outcome": o, "answer": o, "seconds": 1.0 + i / 100, "mode_path": ["fast"]}
        for i, o in enumerate(outcomes)]}


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 2) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(6, 1) == pytest.approx(0.125)
    assert mcnemar_exact(1, 6) == mcnemar_exact(6, 1)


def test_identical_runs_are_reproducible():
    a = run(["correct"] * 40 + ["wrong"] * 10 + ["refused_ok"] * 5 + ["not_applicable"])
    c = compare(a, copy.deepcopy(a))
    assert c["n"] == 55 and c["p_value"] == 1.0 and c["within_noise"]
    assert c["same_outcome"] == c["same_answer"] == 55 and "随机波动范围内" in report(c)


def test_small_flips_are_noise_but_a_systematic_drop_is_not():
    base = ["correct"] * 80 + ["wrong"] * 20
    flipped = list(base)
    flipped[0], flipped[85] = "wrong", "correct"                 # 一对来回：随机波动
    assert compare(run(base), run(flipped))["within_noise"]
    dropped = ["wrong"] * 12 + base[12:]                       # 12 条只在第一次答对
    c = compare(run(base), run(dropped))
    assert not c["within_noise"] and c["p_value"] < 0.001
    assert len(c["discordant"]["first_only"]) == 12 and "不可复现" in report(c)


@pytest.mark.parametrize("key, value", [("model", "other"), ("temperature", "0.7"), ("embedder", "x"),
                                        ("index", "OpenSearchIndex"), ("per_type", 5)])
def test_different_configurations_are_not_compared(key, value):
    a, b = run(["correct"] * 5), run(["correct"] * 5)
    b["config"][key] = value
    with pytest.raises(ConfigMismatch, match=key):
        compare(a, b)
    b = run(["correct"] * 5)
    b["config"]["dataset"]["sha256"] = "cd" * 32                # 评测集改过一个字节也不是同一版本
    with pytest.raises(ConfigMismatch, match="dataset"):
        compare(a, b)


def test_run_twice_end_to_end(monkeypatch, tmp_path, capsys):
    from eaer.evalrun.repro import main

    admin = os.environ.get("EAER_TEST_ADMIN_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
    with StubModelServer() as stub:
        for k in [k for k in os.environ if k.startswith(("EAER_GEN_", "EAER_EMBED_", "EAER_RERANK_"))]:
            monkeypatch.delenv(k)
        for k, v in stub.env().items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("EAER_GEN_TEMPERATURE", "0")
        monkeypatch.setenv("EAER_ENV", "eval")
        code = main(["run", "docs/事实表样例.xlsx", "--admin-url", admin, "--engine", "--format", "html",
                     "--keep", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "复现检查（AC-14-5）" in out and "差异在随机波动范围内" in out and "数据集：自建合成集" in out
    a, b = (json.loads((tmp_path / f).read_text(encoding="utf-8")) for f in ("first.json", "second.json"))
    assert a["config"] == b["config"] and a["config"]["model"] == "stub-gen" and a["config"]["temperature"] == "0"
    c = compare(a, b)
    assert c["n"] >= 12 and c["same_outcome"] == c["same_answer"] == c["same_mode_path"] == c["n"]
