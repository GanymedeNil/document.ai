"""对应开发文档 AC-14-1（事实表校验）与 AC-14-3（答案计算）。"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from factset import Calculator, compare, load, validate  # noqa: E402
from factset.cli import main  # noqa: E402
from fixture import sample, set_cell, write  # noqa: E402


@pytest.fixture
def sample_path(tmp_path: Path) -> Path:
    return write(sample(), tmp_path / "sample.xlsx")


def messages(data, tmp_path) -> list[str]:
    fs = load(write(data, tmp_path / "case.xlsx"))
    return [str(i) for i in validate(fs)]


# ---------- AC-14-1：事实表校验 ----------

def test_sample_is_valid(sample_path):
    assert validate(load(sample_path)) == []


@pytest.mark.parametrize("mutate, expected", [
    (lambda d: d["版本"].append(list(d["版本"][1])), "重复"),
    (lambda d: set_cell(d, "事实", "F01", "版本编号", "P009-v1"), "版本 P009-v1 不存在"),
    (lambda d: set_cell(d, "版本", "P001-v1", "取代的版本", "P001-v3"), "版本号不小于本版本"),
    (lambda d: set_cell(d, "查询", "Q09", "依据事实", "F05"), "应拒答的查询不填依据事实"),
    (lambda d: set_cell(d, "版本", "P003-v1", "生效日写法", "正文写明"), "记录类文档的生效日写法应为不适用"),
    (lambda d: set_cell(d, "查询", "Q04", "回溯时间点", None), "回溯查询必须填写回溯时间点"),
    (lambda d: set_cell(d, "查询", "Q01", "依据事实", "F05"), "事实键与目标事实键不一致"),
    (lambda d: set_cell(d, "事件", "E03", "事件类型", "归档"), "已经处于归档"),
    (lambda d: set_cell(d, "文档", "P002", "可访问角色", "R_BOSS"), "R_BOSS 不在角色表中"),
    (lambda d: set_cell(d, "版本", "P001-v2", "生效日", None), "生效日为空"),
])
def test_broken_data_is_rejected(tmp_path, mutate, expected):
    data = sample()
    mutate(data)
    found = messages(data, tmp_path)
    assert any(expected in m for m in found), found


def test_supersede_twice_is_rejected(tmp_path):
    data = sample()
    set_cell(data, "版本", "P001-v3", "取代的版本", "P001-v1")
    assert any("不能被取代两次" in m for m in messages(data, tmp_path))


# ---------- AC-14-3：答案计算 ----------

def test_all_sample_queries_match(sample_path):
    fs = load(sample_path)
    calc = Calculator(fs)
    mismatches = {q.id: compare(q, calc.compute(q)) for q in fs.queries}
    assert {k: v for k, v in mismatches.items() if v} == {}
    assert len(fs.queries) == 16


@pytest.mark.parametrize("qid, refuse, basis, grade", [
    ("Q01", False, ("F03",), "explicit"),
    ("Q02", False, ("F02",), "explicit"),
    ("Q03", False, ("F03",), "explicit"),           # 实际生效：7 月 1 日起是 v3
    ("Q04", False, ("F02",), "explicit"),           # 回溯到 7 月 2 日：v3 尚未审核通过
    ("Q05", False, ("F02", "F03"), "explicit"),
    ("Q06", False, ("F05",), "explicit"),           # 10 月 1 日补录后为 explicit
    ("Q07", False, ("F05",), "inferred"),           # 回溯到 9 月 15 日：补录尚未发生
    ("Q08", False, ("F06",), "explicit"),
    ("Q09", True, (), None),                        # 无权查看
    ("Q10", False, ("F04", "F08"), "explicit"),     # 冲突，explicit 优先
    ("Q11", False, ("F09",), "explicit"),           # 归档前
    ("Q12", True, (), None),                        # 归档中
    ("Q13", False, ("F09",), "explicit"),           # 取消归档后
    ("Q14", True, (), None),                        # 语料中没有
    ("Q15", False, ("F07",), None),                 # 记录类没有来源等级
    ("Q16", False, ("F03",), "explicit"),
])
def test_each_query(sample_path, qid, refuse, basis, grade):
    fs = load(sample_path)
    q = next(q for q in fs.queries if q.id == qid)
    e = Calculator(fs).compute(q)
    assert (e.refuse, e.basis, e.grade) == (refuse, basis, grade)


def test_q07_is_unverified_and_q16_is_unsupported(sample_path):
    fs = load(sample_path)
    calc = Calculator(fs)
    by_id = {q.id: calc.compute(q) for q in fs.queries}
    assert by_id["Q07"].unverified
    assert by_id["Q16"].answer == "不支持"
    assert by_id["Q10"].conflict and by_id["Q10"].values == ("450 元/晚", "500 元/晚")


def test_wrong_manual_answer_is_reported(tmp_path):
    data = sample()
    set_cell(data, "查询", "Q02", "预期答案", "650 元/晚")
    set_cell(data, "查询", "Q02", "依据事实", "F03")
    fs = load(write(data, tmp_path / "case.xlsx"))
    q = next(q for q in fs.queries if q.id == "Q02")
    diffs = compare(q, Calculator(fs).compute(q))
    assert any("依据事实" in d for d in diffs) and any("600 元/晚" in d for d in diffs)


def test_permission_change_event(tmp_path):
    """10 月 20 日起 P002 对普通员工开放，Q09 不再应拒答。"""
    data = sample()
    data["事件"].append(["E04", date(2026, 10, 20), "权限变更", "P002", None, None, None, "R_STAFF; R_FIN"])
    fs = load(write(data, tmp_path / "case.xlsx"))
    assert validate(fs) == []
    q = next(q for q in fs.queries if q.id == "Q09")
    e = Calculator(fs).compute(q)
    assert not e.refuse and e.basis == ("F06",)


# ---------- 命令行 ----------

def test_cli_check_and_export(sample_path, tmp_path, capsys):
    assert main(["check", str(sample_path)]) == 0
    out = tmp_path / "evalset.jsonl"
    assert main(["export", str(sample_path), "-o", str(out)]) == 0
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 16 and all(r["manual_matches"] for r in lines)
    assert lines[3]["expected"]["basis"] == ["F02"]


def test_cli_rejects_invalid_sheet(tmp_path, capsys):
    data = sample()
    set_cell(data, "事实", "F01", "版本编号", "P009-v1")
    assert main(["check", str(write(data, tmp_path / "bad.xlsx"))]) == 1
    assert "校验未通过" in capsys.readouterr().out
