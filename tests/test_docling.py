"""Docling 解析器（开发文档 6.1、6.2）：PDF、扫描件（OCR）、PPTX、XLSX、图片，以及经入库流水线后的证据定位（页码）。

需要安装 requirements-docling.txt；未安装时整个文件跳过（CI 中由单独的 docling 任务运行）。
首次运行会从 Hugging Face 下载版面与表格模型；离线环境用 EAER_DOCLING_ARTIFACTS 指向预先下载的模型目录。
"""
from __future__ import annotations

import pytest

from eaer.ingestion.docling_parser import available

pytestmark = pytest.mark.skipif(not available(), reason="未安装 Docling（requirements-docling.txt）")


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    from docsamples import make_all
    return make_all(tmp_path_factory.mktemp("docling"))


@pytest.fixture(scope="module")
def parser():
    from eaer.ingestion.docling_parser import DoclingParser
    return DoclingParser()


def parse(parser, path):
    return parser.parse(path.read_bytes(), path.name)


def squash(s: str) -> str:
    return "".join(s.split())


def test_supports_formats_of_section_6_1(parser):
    for name in ("a.pdf", "a.PPTX", "a.xlsx", "a.png", "a.jpg", "a.tiff"):
        assert parser.supports(name), name
    for name in ("a.docx", "a.html", "a.dwg"):        # Word 与 HTML 走原生解析器
        assert not parser.supports(name), name


def test_text_pdf_paragraphs_with_pages(parser, samples):
    doc = parse(parser, samples["pdf"])
    assert doc.parser == "docling-2.143.1"
    paras = {squash(b.text): b.page for b in doc.blocks if b.kind == "paragraph"}
    assert paras.get("第二条一线城市住宿上限为每晚650元。") == 1, paras
    assert paras.get("第四条本办法由财务部负责解释。") == 2, paras
    assert squash(doc.title or "") == "差旅报销管理办法"
    sec = {squash(b.text): b.section for b in doc.blocks if b.kind == "paragraph"}
    assert sec["第四条本办法由财务部负责解释。"] == ("附则",)


@pytest.mark.parametrize("key", ["scanned", "png"])
def test_scanned_pages_go_through_ocr(parser, samples, key):
    doc = parse(parser, samples[key])
    text = squash(" ".join(b.text for b in doc.blocks))
    assert "650" in text and "住宿" in text and "2026" in text, text
    assert {b.page for b in doc.blocks if b.text} == {1}


def test_pptx_slides_as_pages_and_tables(parser, samples):
    doc = parse(parser, samples["pptx"])
    texts = {squash(b.text): b.page for b in doc.blocks if b.text}
    assert texts.get("一线城市住宿上限为每晚650元") == 1, texts
    tables = [b for b in doc.blocks if b.kind == "table"]
    assert len(tables) == 1 and tables[0].page == 2
    assert tables[0].rows == [["项目", "标准"], ["伙食补贴", "80元/天"], ["交通补贴", "50元/天"]]


def test_xlsx_sheet_as_table(parser, samples):
    doc = parse(parser, samples["xlsx"])
    tables = [b for b in doc.blocks if b.kind == "table"]
    assert len(tables) == 1
    assert tables[0].rows == [["城市等级", "住宿上限（元/晚）"], ["一线城市", "650"], ["二线城市", "450"]]


def test_pipeline_records_page_in_locators_and_extracts_dates(conn, tmp_path, samples):
    """经入库流水线：PDF 证据带页码，时间抽取照常；发布后 PPTX 表格按单元格生成深度证据，也带页码。"""
    from dbutil import Clock
    from eaer.ingestion.pipeline import Ingestion, Worker
    from eaer.storage import LocalFileStore

    ing = Ingestion(conn, LocalFileStore(tmp_path / "files"), Clock())
    ing.lc.ensure_tenant("t1")
    acl = ing.lc.create_acl_policy("t1", ["R_STAFF"])
    worker = Worker(ing)
    r = ing.receive("t1", samples["pdf"].read_bytes(), "text.pdf", acl_policy_id=acl)
    worker.run_until_idle()
    rows = conn.execute("SELECT locator, content FROM evidence WHERE version_id = %s", (r.version_id,)).fetchall()
    by_text = {squash(x["content"]): x["locator"] for x in rows}
    assert by_text["第二条一线城市住宿上限为每晚650元。"]["page"] == 1
    assert by_text["第四条本办法由财务部负责解释。"]["page"] == 2
    v = conn.execute("SELECT parser_version FROM document_version WHERE id = %s", (r.version_id,)).fetchone()
    assert v["parser_version"] == "docling-2.143.1"
    draft = conn.execute("SELECT valid_from FROM version_validity WHERE version_id = %s AND recorded_at IS NULL",
                         (r.version_id,)).fetchone()
    assert str(draft["valid_from"]) == "2026-01-01"

    deck = ing.receive("t1", samples["pptx"].read_bytes(), "deck.pptx", acl_policy_id=acl)
    worker.run_until_idle()
    conn.execute("UPDATE document SET doc_type = 'normative' WHERE id = %s", (deck.document_id,))
    ing.lc.publish(deck.version_id, "alice")
    worker.run_until_idle()
    cells = conn.execute("SELECT locator, content FROM evidence WHERE version_id = %s AND modality = 'table'",
                         (deck.version_id,)).fetchall()
    assert {c["content"] for c in cells} == {"伙食补贴｜标准：80元/天", "交通补贴｜标准：50元/天"}
    assert {c["locator"]["page"] for c in cells} == {2}
