"""入库吞吐的测量工具（eaer/evalrun/throughput.py，AC-6-10、AC-13-6）。真实机器上的数字记入 CLAUDE.md 与机器规格对照表。

- 七种格式都能按预定页数渲染（PDF 与扫描件按物理页、PPTX 按幻灯片、XLSX 按工作表、图片一页、Word 按分页符）；
- 端到端小样（HTML、Word；模型服务用 OpenAI 兼容的替身，真实 OpenSearch）：每种格式份数、页数、各阶段耗时、每小时页数、
  存储大小、模型调用都有记录，没有失败。需要 EAER_TEST_OPENSEARCH_URL。
"""
from __future__ import annotations

import io
import os
import re
import uuid

import pytest

from eaer.evalrun.throughput import FORMATS, make_doc, render, report, run

OS_URL = os.environ.get("EAER_TEST_OPENSEARCH_URL")


def test_every_format_renders_with_the_intended_pages():
    pytest.importorskip("pptx", reason="渲染 PPTX 要 python-pptx（requirements-docling.txt；CI 的 docling 任务跑这条）")
    doc = make_doc(3, 4, chart=True)
    assert len(doc.pages) == 4 and doc.pages[-1].lines[-1] == "本办法自2026年1月1日起施行。"
    assert doc.pages[2].table and doc.chart and any(u == "%" for *_, u in doc.chart)
    out = {f: render(doc, f) for f in FORMATS}
    assert out["png"].startswith(b"\x89PNG")
    for f in ("pdf", "scan"):
        assert out[f].startswith(b"%PDF") and len(re.findall(rb"/Type\s*/Page[^s]", out[f])) == 4, f
    from pptx import Presentation
    assert len(Presentation(io.BytesIO(out["pptx"])).slides) == 4
    from openpyxl import load_workbook
    assert len(load_workbook(io.BytesIO(out["xlsx"])).sheetnames) == 4
    from docx import Document
    d = Document(io.BytesIO(out["docx"]))
    assert sum(r._element.xml.count('w:type="page"') for p in d.paragraphs for r in p.runs) == 3
    assert out["html"].count(b"page-break") == 4 and b"data:image/png;base64," in out["html"]


@pytest.mark.skipif(not OS_URL, reason="未设置 EAER_TEST_OPENSEARCH_URL")
def test_small_end_to_end_run(db_url, tmp_path):
    from eaer.adapters.online import OnlineModelClient
    from eaer.ingestion.indexer import fresh_index
    from eaer.retrieval.opensearch import connect
    from stubmodel import StubModelServer

    alias = f"eaer-tp-{uuid.uuid4().hex[:8]}"
    with StubModelServer() as stub, fresh_index(connect(OS_URL), alias, build_id="tp", dims=stub.dims):
        model = OnlineModelClient.from_env(stub.env())
        r = run(db_url, connect(OS_URL), alias, tmp_path / "files", model.embedder(), pages=20, pages_per_doc=5,
                formats=["html", "docx"])
    assert r["pages"] == 20 and set(r["by_format"]) == {"html", "docx"}
    for s in r["by_format"].values():
        assert s["docs"] == 2 and s["pages"] == 10 and s["failed"] == 0 and s["pages_per_hour"] > 0
        assert {"receive", "parse_text", "classify_extract", "approve", "publish_index", "parse_deep"} <= set(s["stages"])
    assert r["evidence"]["text"] >= 20 * 10 and r["evidence"]["table"] > 0
    assert all(v > 0 for v in r["bytes"].values()) and r["model_calls"]["embed"]["calls"] > 0
    assert "每小时" in report(r)
