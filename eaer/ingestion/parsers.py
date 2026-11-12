"""文档解析器接口与原生实现（开发文档 6.1、6.2 第 2 阶段）。

生产环境的首选解析器是 Docling（PDF、扫描件、PPTX、XLSX、图片），见 docling_parser.py；
这里的 Word 与 HTML 原生解析器不依赖模型，用于测试，也作为这两种格式的轻量实现。
解析结果按文档顺序给出块：标题、段落、表格、图片；页码对 Word 和 HTML 不适用。
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol


class UnsupportedFormat(Exception):
    pass


@dataclass
class Block:
    kind: str                         # heading / paragraph / table / figure
    text: str = ""
    level: int = 0                    # 标题层级
    rows: list[list[str]] = field(default_factory=list)
    section: tuple[str, ...] = ()     # 所在章节的标题路径
    order: int = 0                    # 在文档中的顺序
    page: int | None = None


@dataclass
class ParsedDocument:
    title: str | None
    blocks: list[Block]
    parser: str

    def paragraphs(self) -> list[Block]:
        return [b for b in self.blocks if b.kind == "paragraph"]


class Parser(Protocol):
    name: str
    def supports(self, filename: str) -> bool: ...
    def parse(self, data: bytes, filename: str) -> ParsedDocument: ...


def _finish(blocks: list[Block], parser: str) -> ParsedDocument:
    """补上顺序号和章节路径；第一个一级标题作为文档标题。"""
    path: list[tuple[int, str]] = []
    title = None
    for i, b in enumerate(blocks):
        b.order = i
        if b.kind == "heading":
            level = b.level or 1
            if title is None and level == 1:
                title = b.text
            path = [p for p in path if p[0] < level] + [(level, b.text)]
            b.section = tuple(t for _, t in path[:-1])
        else:
            b.section = tuple(t for _, t in path)
    return ParsedDocument(title, blocks, parser)


class DocxParser:
    name = "native-docx-1"

    def supports(self, filename: str) -> bool:
        return Path(filename).suffix.lower() == ".docx"

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        d = Document(io.BytesIO(data))
        blocks: list[Block] = []
        for el in d.element.body.iterchildren():
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "p":
                p = Paragraph(el, d)
                if el.xpath(".//w:drawing"):
                    blocks.append(Block("figure", p.text.strip()))
                    continue
                text = p.text.strip()
                if not text:
                    continue
                style = (p.style.name or "") if p.style is not None else ""
                if style == "Title":
                    blocks.append(Block("heading", text, 1))
                elif style.startswith("Heading"):
                    try:
                        level = int(style.split()[-1])
                    except ValueError:
                        level = 1
                    blocks.append(Block("heading", text, level))
                else:
                    blocks.append(Block("paragraph", text))
            elif tag == "tbl":
                t = Table(el, d)
                blocks.append(Block("table", rows=[[c.text.strip() for c in r.cells] for r in t.rows]))
        return _finish(blocks, self.name)


class _HTMLCollector(HTMLParser):
    HEADINGS = {f"h{i}": i for i in range(1, 7)}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._buf: list[str] = []
        self._tag: str | None = None
        self._rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "title", "head"):
            self._skip += 1
        elif tag == "table":
            self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "img":
            self.blocks.append(Block("figure", dict(attrs).get("alt", "") or ""))
        elif tag in self.HEADINGS or tag in ("p", "li", "figcaption"):
            self._tag, self._buf = tag, []

    def handle_endtag(self, tag):
        if tag in ("script", "style", "title", "head"):
            self._skip = max(0, self._skip - 1)
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._rows is not None:
            self._rows.append(self._row)
            self._row = None
        elif tag == "table" and self._rows is not None:
            self.blocks.append(Block("table", rows=self._rows))
            self._rows = None
        elif tag == self._tag:
            text = " ".join("".join(self._buf).split())
            if text and tag != "figcaption":
                if tag in self.HEADINGS:
                    self.blocks.append(Block("heading", text, self.HEADINGS[tag]))
                else:
                    self.blocks.append(Block("paragraph", text))
            self._tag = None

    def handle_data(self, data):
        if self._skip:
            return
        if self._cell is not None:
            self._cell.append(data)
        elif self._tag is not None:
            self._buf.append(data)


class HtmlParser:
    name = "native-html-1"

    def supports(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in (".html", ".htm")

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        c = _HTMLCollector()
        c.feed(data.decode("utf-8", errors="replace"))
        c.close()
        return _finish(c.blocks, self.name)


DEFAULT_PARSERS: list[Parser] = [DocxParser(), HtmlParser()]


def _register_docling() -> None:
    # 未安装 Docling 时它不认领任何格式，PDF 等在上传时即被拒绝
    from eaer.ingestion.docling_parser import DoclingParser
    DEFAULT_PARSERS.append(DoclingParser())


_register_docling()


def parser_for(filename: str, parsers: list[Parser] | None = None) -> Parser:
    for p in parsers or DEFAULT_PARSERS:
        if p.supports(filename):
            return p
    raise UnsupportedFormat(f"不支持的格式：{Path(filename).suffix or filename}")
