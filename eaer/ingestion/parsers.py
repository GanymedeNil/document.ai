"""文档解析器接口与原生实现（开发文档 6.1、6.2 第 2 阶段）。

生产环境的首选解析器是 Docling（PDF、扫描件、PPTX、XLSX、图片），见 docling_parser.py；
这里的 Word 与 HTML 原生解析器不依赖模型，用于测试，也作为这两种格式的轻量实现。
解析结果按文档顺序给出块：标题、段落、表格、图片；页码对 Word 和 HTML 不适用。
图片块带上图片本身（image、image_type），供深度解析交给视觉模型读取（6.2 第 6 阶段、12.1）：Word 取嵌入的图片；
HTML 只取内嵌的 data URI 图片——外部地址不去下载（入库进程不访问文档里写的任意网址），这类图片只留替代文字。
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
    image: bytes | None = field(default=None, repr=False)   # 图片块：图片内容
    image_type: str | None = None     # 图片块：image/png、image/jpeg、image/gif、image/webp
    header_rows: int = 1              # 表格块：表头的行数（多层表头时大于 1，如“EU movers”下分“1 000s”“%”，再分年份）


IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def sniff_image(data: bytes) -> str | None:
    """按内容判断图片格式（不信扩展名与声明的类型）；不是支持的格式返回 None。"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def figure(text: str = "", data: bytes | None = None, **kw) -> Block:
    """图片块；data 不是支持的图片格式或过大时只留文字。"""
    mime = sniff_image(data) if data and len(data) <= MAX_IMAGE_BYTES else None
    return Block("figure", text, image=data if mime else None, image_type=mime, **kw)


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
                    rids = el.xpath(".//a:blip/@r:embed")
                    part = d.part.related_parts.get(rids[0]) if rids else None
                    blocks.append(figure(p.text.strip(), getattr(part, "blob", None)))
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
        self._row_th: list[bool] = []     # 当前行各格是不是 th
        self._head: list[bool] = []       # 已收的各行是不是全为 th（空格不计）
        self._is_th = False
        self._span = 1                    # 当前格的 colspan：跨几列就在每列重复一次，多层表头才对得上列
        self._cell: list[str] | None = None
        self._caption: list[str] = []
        self._in_caption = False
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "title", "head"):
            self._skip += 1
        elif tag == "table":
            self._rows, self._caption, self._head = [], [], []
        elif tag == "caption" and self._rows is not None:
            self._in_caption = True
        elif tag == "tr" and self._rows is not None:
            self._row, self._row_th = [], []
        elif tag in ("td", "th") and self._row is not None:
            self._cell, self._is_th = [], tag == "th"
            span = dict(attrs).get("colspan") or "1"
            self._span = min(max(int(span), 1), 50) if span.strip().isdigit() else 1
        elif tag == "img":
            a = dict(attrs)
            self.blocks.append(figure(a.get("alt", "") or "", _data_uri(a.get("src") or "")))
        elif tag in self.HEADINGS or tag in ("p", "li", "figcaption"):
            self._tag, self._buf = tag, []

    def handle_endtag(self, tag):
        if tag in ("script", "style", "title", "head"):
            self._skip = max(0, self._skip - 1)
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row += [" ".join("".join(self._cell).split())] * self._span
            self._row_th += [self._is_th] * self._span
            self._cell = None
        elif tag == "tr" and self._row is not None and self._rows is not None:
            filled = [th for x, th in zip(self._row, self._row_th) if x]
            self._head.append(bool(filled) and all(filled))
            self._rows.append(self._row)
            self._row = None
        elif tag == "caption":
            self._in_caption = False
        elif tag == "table" and self._rows is not None:
            caption = " ".join("".join(self._caption).split())
            self.blocks.append(Block("table", caption, rows=self._rows,       # 表格块的 text 是表题
                                     header_rows=header_rows(self._head, len(self._rows))))
            self._rows, self._caption = None, []
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
        elif self._in_caption:
            self._caption.append(data)
        elif self._tag is not None:
            self._buf.append(data)


def _data_uri(src: str) -> bytes | None:
    """data:image/…;base64,… 的图片内容；其他（外部地址、相对路径）返回 None，不去下载。"""
    import base64
    import binascii

    if not src.startswith("data:image/") or ";base64," not in src[:64]:
        return None
    try:
        return base64.b64decode(src.split(",", 1)[1], validate=False)
    except (binascii.Error, ValueError):
        return None


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


def header_rows(is_header: list[bool], n_rows: int) -> int:
    """表头行数：从第一行起连续的表头行；至少 1 行（没有标记时与原先一样把第一行当表头），
    全部行都标成表头时也按 1 行（不能让一张表一条证据都没有）。"""
    n = 0
    for h in is_header:
        if not h:
            break
        n += 1
    return n if 1 <= n < n_rows else 1


def parser_for(filename: str, parsers: list[Parser] | None = None) -> Parser:
    for p in parsers or DEFAULT_PARSERS:
        if p.supports(filename):
            return p
    raise UnsupportedFormat(f"不支持的格式：{Path(filename).suffix or filename}")
