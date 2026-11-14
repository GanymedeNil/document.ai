"""表格的深度解析（开发文档 6.2 第 6 阶段）：每个数据单元格一条证据（带行名、列名）；只有第一格有内容的行与单列的表
整格一条证据，不丢内容。

起因（2026-11-14 入库吞吐测量）：Docling 把电子表格的每个工作表解析成一张表，一列写的条款只有第一格，原先的规则
只取第二列起的单元格，这些条款全被丢掉，XLSX 的正文一条都检索不到。
"""
from __future__ import annotations

import importlib.util

import pytest

from test_ingestion import env, html  # noqa: F401
from test_retrieval import ix, publish  # noqa: F401


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
