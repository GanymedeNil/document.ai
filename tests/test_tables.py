"""表格的深度解析（开发文档 6.2 第 6 阶段）：每个数据单元格一条证据（带行名、列名）；只有第一格有内容的行与单列的表
整格一条证据，不丢内容。

起因（2026-11-14 入库吞吐测量）：Docling 把电子表格的每个工作表解析成一张表，一列写的条款只有第一格，原先的规则
只取第二列起的单元格，这些条款全被丢掉，XLSX 的正文一条都检索不到。
"""
from __future__ import annotations

import importlib.util
from datetime import date

import pytest

from test_ingestion import env, html  # noqa: F401
from test_retrieval import SCOPE, ix, publish  # noqa: F401


def table_evidence(ix, vid) -> list[str]:
    return [r["content"] for r in ix.conn.execute(
        "SELECT content FROM evidence WHERE version_id = %s AND modality = 'table' ORDER BY content", (vid,))]


def test_first_column_only_rows_and_single_column_tables_are_kept(ix):
    body = ("<h1>差旅标准</h1><p>第一条　本标准自2026年1月1日起施行。</p>"
            "<table><tr><th>项目</th><th>标准</th></tr><tr><td>一线城市住宿</td><td>650元/晚</td></tr>"
            "<tr><td>注：出差超过30天的按八折执行</td><td></td></tr></table>"
            "<table><tr><td>须知</td></tr><tr><td>住宿须取得正规发票</td></tr><tr><td>超标部分自理</td></tr></table>")
    r = publish(ix, body.encode(), "t.html")
    ix.worker.run_until_idle()
    ev = table_evidence(ix, r.version_id)
    assert "一线城市住宿｜标准：650元/晚" in ev                           # 原来的单元格证据不变
    assert "注：出差超过30天的按八折执行" in ev                           # 只有第一格的行
    assert {"须知", "住宿须取得正规发票", "超标部分自理"} <= set(ev)          # 单列的表：每行一条（含第一行）


@pytest.mark.skipif(importlib.util.find_spec("docling") is None, reason="没有安装 Docling")
def test_spreadsheet_clauses_become_evidence(ix):
    from eaer.evalrun.throughput import make_doc, render

    doc = make_doc(7, 2, chart=False)
    r = ix.ing.receive("t1", render(doc, "xlsx"), "clauses.xlsx", acl_policy_id=ix.acl)
    ix.worker.run_until_idle()
    task = str(ix.conn.execute("SELECT id FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()["id"])
    assert ix.review.decide("t1", [{"task_id": task, "decision": "approve",
                                    "corrections": {"doc_type": "normative"}}], "rv")[0]["ok"]   # 表格标题判不出类型
    ix.worker.run_until_idle()
    ev = table_evidence(ix, r.version_id)
    clauses = [x for p in doc.pages for x in p.lines]
    assert all(c.replace("　", " ") in {e.replace("　", " ") for e in ev} for c in clauses), ev[:5]


def test_table_caption_prefixes_every_cell(ix):
    """有表题的表格，每个单元格的证据前加“表N《表题》｜”：单元格要靠表题才检索得到、才知道单位（ViDoRe V3 发现：
    “Switzerland｜2022：12.3”这样的单元格不知道说的是什么，问题检索到的只有表题段落，取不到数值）。
    没有表题的表格照旧，已有文档的证据 id 不变。"""
    body = ("<h1>劳动力流动管理办法</h1><p>第一条　本办法自2026年1月1日起施行。</p>"
            "<table><caption>Table 3: Part-time contracts by country, 2022 (%)</caption>"
            "<tr><th>Country</th><th>2022</th></tr><tr><td>Switzerland</td><td>12.3</td></tr></table>"
            "<table><tr><th>项目</th><th>标准</th></tr><tr><td>一线城市住宿</td><td>650元/晚</td></tr></table>")
    r = publish(ix, body.encode(), "t.html")
    ix.worker.run_until_idle()
    ev = table_evidence(ix, r.version_id)
    assert "表1《Table 3: Part-time contracts by country, 2022 (%)》｜Switzerland｜2022：12.3" in ev
    assert "一线城市住宿｜标准：650元/晚" in ev
    hits = ix.retriever.search(SCOPE, "part-time contracts Switzerland", today=date(2026, 3, 1))
    assert any("Switzerland" in h.content for h in hits)


def test_multi_row_header_names_the_columns_and_is_not_data(ix):
    """多层表头（“EU movers”下分 2021、2022）：列名是各行表头连起来，第二行表头不当作数据。原先只把第一行当表头，
    第二行变成“表N《表题》｜｜EU movers：2021”这样只有表题、没有数值的单元格，与问题用词相近时占满检索结果
    （ViDoRe Q22、Q80 因此拒答），数据单元格“EU-27｜EU movers：762”也看不出是哪一年。HTML 的 colspan 按跨的列数
    重复，两行表头才对得上列。经过解析结果暂存（文本解析存、深度解析取），表头行数随暂存保留。"""
    body = ("<h1>劳动力流动年度报告</h1><p>报告日期：2026年1月5日</p>"
            "<table><caption>Table 18: Composition of inflows of movers (1 000s), 2021-2022</caption>"
            "<tr><th></th><th colspan=\"2\">EU movers</th><th>Third-country nationals</th></tr>"
            "<tr><th></th><th>2021</th><th>2022</th><th>2022</th></tr>"
            "<tr><td>EU-27</td><td>762</td><td>860</td><td>3 324</td></tr></table>"
            "<table><tr><th>项目</th><th>标准</th></tr><tr><td>住宿</td><td>650元/晚</td></tr></table>")
    r = publish(ix, body.encode(), "t.html")
    ix.worker.run_until_idle()
    cap = "表1《Table 18: Composition of inflows of movers (1 000s), 2021-2022》｜"
    assert table_evidence(ix, r.version_id) == sorted([
        cap + "EU-27｜EU movers / 2021：762", cap + "EU-27｜EU movers / 2022：860",
        cap + "EU-27｜Third-country nationals / 2022：3 324", "住宿｜标准：650元/晚"])


def test_header_rows_and_column_names():
    from types import SimpleNamespace as C

    from eaer.ingestion.docling_parser import header_flags
    from eaer.ingestion.parsers import header_rows
    from eaer.ingestion.pipeline import column_names

    assert header_rows([True, True, False], 3) == 2
    assert header_rows([False, True, False], 3) == 1        # 没有标记时与原先一样，第一行当表头
    assert header_rows([True, True], 2) == 1                # 全部标成表头时也按 1 行，不让整张表没有证据
    assert column_names([["", "EU movers", "EU movers"], ["", "2021", "2022"]]) == ["", "EU movers / 2021",
                                                                                   "EU movers / 2022"]
    assert column_names([["Total"], ["Total"]]) == ["Total"]  # 跨行合并的表头只取一次
    assert column_names([["项目", "标准"]]) == ["项目", "标准"]
    grid = [[C(text="", column_header=False), C(text="EU movers", column_header=True)],
            [C(text="", column_header=False), C(text="2021", column_header=True)],
            [C(text="EU-27", column_header=False), C(text="762", column_header=False)]]
    assert header_flags(grid) == [True, True, False]


def test_docling_table_caption_falls_back_to_the_title_paragraph_before_it():
    from eaer.ingestion.docling_parser import table_caption
    from eaer.ingestion.parsers import Block

    title = Block("paragraph", "Table 5: Stock of movers in the EU by age group, 2018-2023 (1 000s)", page=39)
    assert table_caption("Table 1: Main indicators", [title], 39) == "Table 1: Main indicators"   # Docling 关联的图注优先
    assert table_caption("", [title], 39) == title.text
    assert table_caption("", [title], 40) == ""                                     # 不在同一页
    assert table_caption("", [Block("paragraph", "Table 5 shows that movers are older.", page=39)], 39) == ""
    assert table_caption("", [Block("paragraph", "表2　各城市住宿标准", page=3)], 3) == "表2　各城市住宿标准"
    assert len(table_caption("Table 9: " + "x" * 400, [], 1)) == 200
