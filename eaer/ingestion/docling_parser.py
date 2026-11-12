"""Docling 解析器（开发文档 6.1、6.2）：PDF（含扫描件，走 OCR）、PPTX、XLSX、图片。

按 parsers.Parser 接口接入；Word 与 HTML 仍用原生解析器。Docling 及其依赖（PyTorch、版面模型）较重，
单独放在 requirements-docling.txt；没有安装时 supports() 返回 False，这些格式在上传时就被拒绝，
不会进入入库任务后才失败。

版面、表格与 OCR 模型首次使用时从 Hugging Face 下载；私有化环境无法联网时，先用
`docling-tools models download` 下载，再用环境变量 EAER_DOCLING_ARTIFACTS 指向模型目录。

映射到 Block：标题与章节标题 → heading；正文、列表项、图注 → paragraph；表格 → table（行列文本）；
图片 → figure。页眉页脚丢弃。页码来自 Docling 的 prov，PPTX 为幻灯片序号，XLSX 为工作表序号。
"""
from __future__ import annotations

import importlib.util
import io
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eaer.ingestion.parsers import ParsedDocument

SUFFIXES = {".pdf", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
VERSION = "2.143.1"
_SKIP_LABELS = {"page_header", "page_footer"}


def available() -> bool:
    return importlib.util.find_spec("docling") is not None


class DoclingParser:
    name = f"docling-{VERSION}"

    def __init__(self, *, ocr: bool = True, artifacts_path: str | None = None):
        self.ocr = ocr
        self.artifacts_path = artifacts_path or os.environ.get("EAER_DOCLING_ARTIFACTS")
        self._converter: Any = None
        self._lock = threading.Lock()
        self._available = available()

    def supports(self, filename: str) -> bool:
        return self._available and Path(filename).suffix.lower() in SUFFIXES

    def _conv(self):
        with self._lock:
            if self._converter is None:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

                opts = PdfPipelineOptions(do_ocr=self.ocr, do_table_structure=True)
                if self.artifacts_path:
                    opts.artifacts_path = self.artifacts_path
                self._converter = DocumentConverter(
                    allowed_formats=[InputFormat.PDF, InputFormat.IMAGE, InputFormat.PPTX, InputFormat.XLSX],
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
                                    InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)})
            return self._converter

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        from docling.datamodel.base_models import DocumentStream
        from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TextItem, TitleItem

        from eaer.ingestion.parsers import Block, _finish   # 在这里导入：parsers 在模块末尾注册本解析器

        res = self._conv().convert(DocumentStream(name=Path(filename).name, stream=io.BytesIO(data)))
        doc = res.document
        items = [it for it, _ in doc.iterate_items()]
        # PDF 通常没有 Title，首个一级章节标题就是文档标题；有 Title 时（PPTX 每页标题）章节标题降一级
        shift = 1 if any(isinstance(it, TitleItem) for it in items) else 0
        blocks: list[Block] = []
        for item in items:
            page = item.prov[0].page_no if getattr(item, "prov", None) else None
            label = str(getattr(item, "label", "")).rsplit(".", 1)[-1].lower()
            if label in _SKIP_LABELS:
                continue
            if isinstance(item, TableItem):
                rows = [[" ".join(c.text.split()) for c in row] for row in item.data.grid]
                if rows:
                    blocks.append(Block("table", rows=rows, page=page))
            elif isinstance(item, PictureItem):
                blocks.append(Block("figure", page=page))
            elif isinstance(item, TitleItem):
                blocks.append(Block("heading", _clean(item.text), 1, page=page))
            elif isinstance(item, SectionHeaderItem):
                blocks.append(Block("heading", _clean(item.text), min(int(item.level or 1) + shift, 6), page=page))
            elif isinstance(item, TextItem) and _clean(item.text):
                blocks.append(Block("paragraph", _clean(item.text), page=page))
        return _finish(blocks, self.name)


def _clean(text: str) -> str:
    return " ".join((text or "").split())
