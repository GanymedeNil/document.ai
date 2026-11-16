"""对应开发文档 14.2 第 3、4 步与 AC-14-2 的程序检查部分。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from factset import load  # noqa: E402
from factset.cli import main  # noqa: E402
from factset.render import Renderer  # noqa: E402
from factset.verify_docs import FULL_DATE, read_docx, read_html, verify  # noqa: E402
from fixture import sample, write  # noqa: E402


@pytest.fixture
def fs(tmp_path):
    return load(write(sample(), tmp_path / "sample.xlsx"))


@pytest.mark.parametrize("fmt", ["docx", "html"])
def test_generated_documents_pass(fs, tmp_path, fmt):
    out = tmp_path / fmt
    manifest = Renderer(fs).render(out, fmt)
    assert len(manifest["versions"]) == 8
    errors, notes = verify(fs, out)
    assert errors == [] and notes == []


@pytest.mark.parametrize("fmt", ["docx", "html"])
def test_chart_value_only_in_image(fs, tmp_path, fmt):
    out = tmp_path / fmt
    Renderer(fs).render(out, fmt)
    c = (read_docx if fmt == "docx" else read_html)(out / f"P003-v1.{fmt}")
    assert "128" not in c.all_text and len(c.images) == 1
    assert (out / "P003-v1-fig1.png").exists()


def test_undated_versions_have_no_full_dates(fs, tmp_path):
    out = tmp_path / "html"
    Renderer(fs).render(out, "html")
    for vid in ("P002-v1", "P004-v1"):
        assert FULL_DATE.findall(read_html(out / f"{vid}.html").all_text) == []
    assert "2026年7月1日" in read_html(out / "P001-v3.html").all_text


def test_manifest_marks_legacy_batch_and_events(fs, tmp_path):
    out = tmp_path / "html"
    Renderer(fs).render(out, "html")
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    batch = {v["version_id"]: v["import_batch_id"] for v in m["versions"]}
    assert batch["P004-v1"] == "legacy-import" and batch["P002-v1"] is None
    assert [e["event_id"] for e in m["events"]] == ["E01", "E02", "E03"]
    # 按审核通过日排序，评测时按此顺序装载
    dates = [v["approved_on"] for v in m["versions"]]
    assert dates == sorted(dates)


def _tamper(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


@pytest.mark.parametrize("vid, old, new, expected", [
    ("P002-v1", "50,000元", "60,000元", "不在正文中"),
    ("P001-v3", "<td>650元/晚</td>", "<td>待定</td>", "不在表格中"),
    ("P002-v1", "本规定未尽事宜", "本规定自2025年3月1日起施行，未尽事宜", "出现日期"),
    ("P003-v1", "<p>相关数据见图1。</p>", "<p>总费用为128万元，见图1。</p>", "不应出现在文字中"),
    ("P001-v3", "本办法自2026年7月1日起施行", "本办法自发布之日起施行", "文中没有 2026年7月1日"),
    ("P003-v1", "报告日期：2026年7月10日", "报告日期：近期", "缺少报告日期"),
])
def test_tampered_documents_fail(fs, tmp_path, vid, old, new, expected):
    out = tmp_path / "html"
    Renderer(fs).render(out, "html")
    _tamper(out / f"{vid}.html", old, new)
    errors, _ = verify(fs, out)
    assert any(expected in str(e) for e in errors), [str(e) for e in errors]


def test_old_value_in_new_version_is_noted(fs, tmp_path):
    out = tmp_path / "html"
    Renderer(fs).render(out, "html")
    _tamper(out / "P001-v3.html", "<p>第三条　各项标准见下表。</p>", "<p>第三条　原标准为600元/晚，现调整如下表。</p>")
    errors, notes = verify(fs, out)
    assert errors == [] and any("F02" in str(n) for n in notes)


def test_cli_render_and_verify(tmp_path):
    xlsx = write(sample(), tmp_path / "sample.xlsx")
    out = tmp_path / "docs"
    assert main(["render", str(xlsx), "-o", str(out)]) == 0
    assert main(["verify-docs", str(xlsx), str(out)]) == 0
    (out / "P005-v1.docx").unlink()
    assert main(["verify-docs", str(xlsx), str(out)]) == 1


def test_mixed_unit_charts_label_each_bar_with_its_unit():
    """不同单位的柱子不能共用一个写着第一项单位的坐标轴：每根柱子标自己的单位，纵轴不写单位。"""
    from factset.render import chart_labels

    assert chart_labels([("Q2 总费用", 128, "万元")]) == (["128"], "万元")
    assert chart_labels([("总费用", 509, "万元"), ("预算执行率", 54, "%"), ("报销笔数", 2770, "笔")]) == (
        ["509万元", "54%", "2770笔"], "")
