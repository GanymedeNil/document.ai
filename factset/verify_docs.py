"""检查合成文档（开发文档 14.2 第 4 步）。

模板生成的文档和以后经模型改写的文档都要过这道检查：
- 正文类事实的取值出现在正文段落中，表格类事实出现在表格单元格中；
- 图表类事实的取值不出现在任何文字中（否则不需要看图就能答），图片存在且数据与事实一致；
- “正文写明”的版本写有生效日；“仅上传时间”和“无”的规范类版本全文没有完整日期；
- 记录类文档写有报告日期；
- 出现同一事实键在其他版本中的取值时给出提示（新旧取值混在一份文档里，答案可能有歧义）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from .model import Fact, FactSet, Issue
from .render import cn_date

FULL_DATE = re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
FULLWIDTH = str.maketrans("０１２３４５６７８９，．", "0123456789,.")


def norm(s: str) -> str:
    """去空白、全角转半角、去掉数字间的千分位逗号。"""
    s = re.sub(r"\s+", "", s.translate(FULLWIDTH))
    return re.sub(r"(?<=\d),(?=\d{3})", "", s)


def value_text(f: Fact) -> str:
    return norm(f"{f.value}{f.unit or ''}")


@dataclass
class Content:
    paragraphs: list[str]
    cells: list[str]
    images: list[str]

    @property
    def all_text(self) -> str:
        return norm("\n".join(self.paragraphs + self.cells))


def read_docx(path: Path) -> Content:
    from docx import Document as Docx
    d = Docx(str(path))
    cells = [c.text for t in d.tables for row in t.rows for c in row.cells]
    return Content([p.text for p in d.paragraphs], cells, [f"inline-{i}" for i, _ in enumerate(d.inline_shapes)])


class _HTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paragraphs, self.cells, self.images, self._buf, self._in_cell = [], [], [], [], False

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self._flush()
            self._in_cell = True
        elif tag == "img":
            a = dict(attrs)
            self.images.append(a.get("src", ""))
            if a.get("alt"):
                self.paragraphs.append(a["alt"])      # 替代文字也算文字

    def handle_endtag(self, tag):
        if tag in ("td", "th", "p", "h1", "h2", "h3", "figcaption", "title", "li"):
            self._flush()

    def handle_data(self, data):
        self._buf.append(data)

    def _flush(self):
        text = "".join(self._buf).strip()
        self._buf = []
        if text:
            (self.cells if self._in_cell else self.paragraphs).append(text)
        self._in_cell = False


def read_html(path: Path) -> Content:
    p = _HTML()
    p.feed(path.read_text(encoding="utf-8"))
    p._flush()
    return Content(p.paragraphs, p.cells, p.images)


def verify(fs: FactSet, out_dir: Path) -> tuple[list[Issue], list[Issue]]:
    """返回（错误，提示）。错误必须修正；提示供人工抽检时留意。"""
    errors: list[Issue] = []
    notes: list[Issue] = []
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = {e["version_id"]: e for e in manifest["versions"]}
    facts_by_version: dict[str, list[Fact]] = {}
    for f in fs.facts.values():
        facts_by_version.setdefault(f.version_id, []).append(f)

    for v in fs.versions.values():
        sheet = f"文档 {v.id}"
        entry = entries.get(v.id)
        if entry is None or not (out_dir / entry["file"]).exists():
            errors.append(Issue(sheet, None, "缺少生成的文档"))
            continue
        path = out_dir / entry["file"]
        content = read_docx(path) if path.suffix == ".docx" else read_html(path)
        paras = norm("\n".join(content.paragraphs))
        cells = norm("\n".join(content.cells))
        text = content.all_text
        doc = fs.documents[v.document_id]
        plotted = {(d[0], float(d[1])) for fig in entry.get("figures", []) for d in fig["data"]}

        for f in facts_by_version.get(v.id, []):
            want = value_text(f)
            if f.carrier == "正文" and want not in paras:
                errors.append(Issue(sheet, None, f"{f.id}（{f.key}）的取值 {want} 不在正文中"))
            elif f.carrier == "表格" and want not in cells:
                errors.append(Issue(sheet, None, f"{f.id}（{f.key}）的取值 {want} 不在表格中"))
            elif f.carrier == "图表":
                if want in text or re.search(rf"(?<![\d.]){re.escape(str(f.value))}(?![\d.])", text):
                    errors.append(Issue(sheet, None, f"{f.id}（{f.key}）是图表类事实，取值不应出现在文字中"))
                if not content.images:
                    errors.append(Issue(sheet, None, f"{f.id}（{f.key}）是图表类事实，但文档中没有图片"))
                if (f.key, float(f.value)) not in plotted:
                    errors.append(Issue(sheet, None, f"{f.id}（{f.key}）的取值与图表数据不一致"))

        if doc.normative:
            if v.valid_from_style == "正文写明" and norm(cn_date(v.valid_from)) not in text:
                errors.append(Issue(sheet, None, f"生效日写法为正文写明，但文中没有 {cn_date(v.valid_from)}"))
            if v.valid_from_style in ("仅上传时间", "无"):
                found = FULL_DATE.findall(text)
                if found:
                    errors.append(Issue(sheet, None, f"生效日写法为{v.valid_from_style}，但文中出现日期 {found[0]}"))
        elif v.doc_date and norm(cn_date(v.doc_date)) not in text:
            errors.append(Issue(sheet, None, f"记录类文档缺少报告日期 {cn_date(v.doc_date)}"))

        own_keys = {f.key: f.value for f in facts_by_version.get(v.id, [])}
        for other in fs.facts.values():
            if (other.version_id != v.id and other.key in own_keys and other.value != own_keys[other.key]
                    and value_text(other) in text):
                notes.append(Issue(sheet, None, f"文中出现 {other.id} 的取值 {value_text(other)}（同一事实键的其他版本）"))
    return errors, notes
