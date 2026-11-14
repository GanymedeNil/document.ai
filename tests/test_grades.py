"""AC-7-3（硬性）来源等级如实：回答标注的等级与事实表一致；仅由 inferred 或空值支撑的时效结论标为“未验证”。

用 eaer/evalrun/grades.py 按日期回放事实表、经完整编排器回答（检索与模型用固定依据的替身，等级与标注全由真实的
数据库核对与验证器决定），逐条比对。这里跑事实表样例与冻结的合成集 v2（测试集每类抽样）；
全量（2000 条）用 `EAER_ENV=eval python -m eaer.evalrun.grades docs/评测集/合成集-v2.xlsx --admin-url …`。
"""
from __future__ import annotations

from pathlib import Path

from eaer.evalrun.grades import report, run_grades
from eaer.storage import LocalFileStore
from factset import load
from factset.render import Renderer
from factset.subset import subset
from fixture import sample, write

FROZEN = Path("docs/评测集/合成集-v2.xlsx")


def _run(conn, fs, tmp_path):
    manifest = Renderer(fs).render(tmp_path / "corpus", "html")
    return run_grades(conn, fs, tmp_path / "corpus", manifest, store=LocalFileStore(tmp_path / "files"))


def _bad(checks):
    return [c for c in checks if c.outcome in ("grade_mismatch", "label_mismatch")]


def test_fact_table_sample(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("EAER_ENV", "eval")
    checks = _run(conn, load(write(sample(), tmp_path / "s.xlsx")), tmp_path)
    assert checks and not _bad(checks), report(checks)
    assert not [c for c in checks if c.outcome == "uncovered"], report(checks)
    assert {c.expected_grade for c in checks} >= {"explicit", "inferred", "unknown"}  # 三种等级都覆盖到
    q07 = [c for c in checks if c.query_id == "Q07"]                                   # 回溯时只有上传时间：未验证
    assert q07 and all(c.expected_grade == "inferred" and c.status == "unverified" and c.labelled for c in q07)
    q06 = [c for c in checks if c.query_id == "Q06"]                                   # 补录之后：explicit，不标
    assert q06 and all(c.expected_grade == "explicit" and c.status == "supported" and not c.labelled for c in q06)


def test_synthetic_set_sampled(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("EAER_ENV", "eval")
    fs = subset(load(FROZEN), split="测试", per_type=3)
    checks = _run(conn, fs, tmp_path)
    assert len({c.query_id for c in checks}) >= 15 and not _bad(checks), report(checks)
    assert {c.expected_grade for c in checks} >= {"explicit", "inferred", "unknown"}, report(checks)


def test_a_wrong_grade_is_caught(conn, tmp_path, monkeypatch):
    """核对本身要能发现问题：让检索把等级一律报成 explicit，样例中的 inferred 依据必须计为不一致。"""
    import eaer.retrieval.service as service

    monkeypatch.setenv("EAER_ENV", "eval")
    monkeypatch.setattr(service, "_grade", lambda doc_type, val: None if doc_type == "record" else "explicit")
    checks = _run(conn, load(write(sample(), tmp_path / "s.xlsx")), tmp_path)
    assert {c.query_id for c in _bad(checks)} >= {"Q07"}
