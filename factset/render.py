"""按事实表生成合成文档（开发文档 14.2 第 3 步的模板实现）。

模板生成不调用模型，结果确定、零成本，事实一定落在指定载体上；
以后接入模型改写措辞时，改写结果同样要过 verify_docs 的检查。

生成规则：
- 正文类事实写成条款句子，表格类事实汇总成一张表，图表类事实画成柱状图，
  图表的取值只出现在图片里，正文、图注和替代文字都不写取值；HTML 把图片以 data URI 内嵌（文件自成一体，
  上传一个文件就带着图，与 Word 嵌入图片一致），同时在旁边留一份 PNG 便于查看；
- 生效日写法为“正文写明”的版本，末条写“自某年某月某日起施行”；
  “仅上传时间”和“无”的版本全文不出现任何完整日期；
- “无”的版本在清单中标记为批量导入，系统不得把上传时间当作时间依据（开发文档 6.2）；
- 新版本写明原版本同时废止；记录类文档写报告日期。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from html import escape
from pathlib import Path

from .model import Fact, FactSet, Version

LEGACY_BATCH = "legacy-import"
KINDS = ("管理办法", "实施细则", "办法", "规定", "细则", "制度", "标准", "规范", "通知")


def cn_date(d: date) -> str:
    return f"{d.year}年{d.month}月{d.day}日"


def fact_text(f: Fact) -> str:
    v = f"{f.value:,}" if isinstance(f.value, (int, float)) else str(f.value)
    return f"{v}{f.unit or ''}"


def kind_of(title: str) -> str:
    for k in KINDS:
        if title.endswith(k):
            return "办法" if k == "管理办法" else k
    return "文件"


@dataclass
class Block:
    kind: str                       # heading / para / table / figure
    text: str = ""
    level: int = 1
    rows: list[list[str]] = field(default_factory=list)
    image: str = ""                 # 图片文件名
    data: list[tuple[str, float, str]] = field(default_factory=list)  # 图表数据，只写进清单


def _pick(options: list[str], seed: str) -> str:
    h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]


class Renderer:
    def __init__(self, fs: FactSet):
        self.fs = fs
        self.facts_by_version: dict[str, list[Fact]] = {}
        for f in sorted(fs.facts.values(), key=lambda f: f.row):
            self.facts_by_version.setdefault(f.version_id, []).append(f)

    # ---------- 文档结构 ----------

    def blocks(self, v: Version) -> list[Block]:
        doc = self.fs.documents[v.document_id]
        facts = self.facts_by_version.get(v.id, [])
        text_facts = [f for f in facts if f.carrier == "正文"]
        table_facts = [f for f in facts if f.carrier == "表格"]
        chart_facts = [f for f in facts if f.carrier == "图表"]
        out = [Block("heading", doc.title, 1)]
        if v.version_no > 1:
            out.append(Block("para", f"（第{v.version_no}版）"))
        if not doc.normative:
            return out + self._record_body(v, text_facts, table_facts, chart_facts)

        k = kind_of(doc.title)
        n = 0

        def article(body: str) -> Block:
            nonlocal n
            n += 1
            return Block("para", f"第{_cn_num(n)}条　{body}")

        out.append(article(_pick([
            f"为规范相关工作，明确执行标准，根据公司有关规定，制定本{k}。",
            f"为加强内部管理，统一执行口径，结合公司实际，制定本{k}。",
        ], v.id + "purpose")))
        out.append(article(f"本{k}适用于公司各部门及全体员工。"))
        for f in text_facts:
            out.append(article(f"{f.key}为{fact_text(f)}。"))
        if table_facts:
            out.append(article("各项标准见下表。"))
            out.append(Block("table", rows=[["项目", "标准"]] + [[f.key, fact_text(f)] for f in table_facts]))
        if chart_facts:
            out.append(article("相关数据见图1。"))
            out.append(self._figure(v, chart_facts))
        out.append(article(f"本{k}由制定部门负责解释。"))
        out.append(article(self._closing(v, doc.title, k)))
        return out

    def _closing(self, v: Version, title: str, k: str) -> str:
        prev = self.fs.versions.get(v.supersedes) if v.supersedes else None
        abolish = f"原《{title}》（第{prev.version_no}版）同时废止。" if prev else ""
        if v.valid_from_style == "正文写明":
            return f"本{k}自{cn_date(v.valid_from)}起施行。{abolish}"
        # 仅上传时间、无：全文不出现完整日期
        return f"本{k}施行后，{abolish}" if abolish else f"本{k}未尽事宜，按公司其他有关规定执行。"

    def _record_body(self, v, text_facts, table_facts, chart_facts) -> list[Block]:
        out = [Block("para", f"报告日期：{cn_date(v.doc_date)}"),
               Block("heading", "一、概述", 2),
               Block("para", "本报告汇总报告期内的相关数据，供管理层参考。"),
               Block("heading", "二、主要数据", 2)]
        for f in text_facts:
            out.append(Block("para", f"报告期内，{f.key}为{fact_text(f)}。"))
        if table_facts:
            out.append(Block("table", rows=[["指标", "数值"]] + [[f.key, fact_text(f)] for f in table_facts]))
        if chart_facts:
            out.append(Block("para", "相关数据见图1。"))
            out.append(self._figure(v, chart_facts))
        out += [Block("heading", "三、说明", 2),
                Block("para", "以上数据来源于公司内部系统，未经审计。")]
        return out

    def _figure(self, v: Version, facts: list[Fact]) -> Block:
        caption = "图1　" + "、".join(f.key for f in facts)   # 图注只写事实键，不写取值
        data = [(f.key, float(f.value), f.unit or "") for f in facts]
        return Block("figure", caption, image=f"{v.id}-fig1.png", data=data)

    # ---------- 输出 ----------

    def render(self, out_dir: Path, fmt: str = "docx") -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"format": fmt, "versions": [], "events": []}
        for v in sorted(self.fs.versions.values(), key=lambda v: (v.approved_on, v.row)):
            doc = self.fs.documents[v.document_id]
            blocks = self.blocks(v)
            for b in blocks:
                if b.kind == "figure":
                    _draw_chart(b, out_dir / b.image)
            name = f"{v.id}.{fmt}"
            (_write_docx if fmt == "docx" else _write_html)(blocks, out_dir, out_dir / name)
            manifest["versions"].append({
                "version_id": v.id, "document_id": doc.id, "title": doc.title,
                "doc_type": doc.doc_type, "version_no": v.version_no, "file": name,
                "time_style": v.valid_from_style,
                "uploaded_on": v.uploaded_on.isoformat(),
                "approved_on": v.approved_on.isoformat(),
                "import_batch_id": LEGACY_BATCH if v.valid_from_style == "无" else None,
                "roles": sorted(doc.roles),
                "figures": [{"image": b.image, "data": b.data} for b in blocks if b.kind == "figure"],
            })
        for e in self.fs.events:
            manifest["events"].append({
                "event_id": e.id, "occurred_on": e.occurred_on.isoformat(), "type": e.type,
                "target": e.target,
                "new_valid_from": e.new_valid_from.isoformat() if e.new_valid_from else None,
                "new_grade": e.new_grade,
                "new_roles": sorted(e.new_roles) if e.new_roles else None,
            })
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


def _cn_num(n: int) -> str:
    digits = "零一二三四五六七八九"
    if n < 10:
        return digits[n]
    if n < 20:
        return "十" + (digits[n % 10] if n % 10 else "")
    return digits[n // 10] + "十" + (digits[n % 10] if n % 10 else "")


def chart_labels(data: list[tuple[str, float, str]]) -> tuple[list[str], str]:
    """柱子上的标注与纵轴标题。单位都相同时纵轴写单位、柱子只写数；单位不同时每根柱子写自己的单位、纵轴不写——
    否则不同单位的柱子共用一个写着第一项单位的坐标轴，图本身就是错的（54 会被读成 54 万元）。"""
    units = {u or "" for _, _, u in data}
    if len(units) == 1:
        return [f"{v:g}" for _, v, _ in data], units.pop()
    return [f"{v:g}{u or ''}" for _, v, u in data], ""


def _draw_chart(b: Block, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    families = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
        if name in families:
            plt.rcParams["font.family"] = name
            break
    labels = [d[0] for d in b.data]
    values = [d[1] for d in b.data]
    bar_texts, ylabel = chart_labels(b.data)
    fig, ax = plt.subplots(figsize=(max(4, 2.2 * len(values)), 3.2), dpi=150)
    bars = ax.bar(labels, values, color="#4c78a8", width=0.5)
    ax.bar_label(bars, labels=bar_texts, padding=3)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_docx(blocks: list[Block], out_dir: Path, path: Path) -> None:
    from docx import Document as Docx
    from docx.shared import Cm

    d = Docx()
    for b in blocks:
        if b.kind == "heading":
            d.add_heading(b.text, level=b.level)
        elif b.kind == "para":
            d.add_paragraph(b.text)
        elif b.kind == "table":
            t = d.add_table(rows=len(b.rows), cols=len(b.rows[0]))
            t.style = "Table Grid"
            for i, row in enumerate(b.rows):
                for j, cell in enumerate(row):
                    t.cell(i, j).text = cell
        elif b.kind == "figure":
            d.add_picture(str(out_dir / b.image), width=Cm(12))
            d.add_paragraph(b.text)
    d.save(path)


def _write_html(blocks: list[Block], out_dir: Path, path: Path) -> None:
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
             f"<title>{escape(blocks[0].text)}</title></head><body>"]
    for b in blocks:
        if b.kind == "heading":
            parts.append(f"<h{b.level}>{escape(b.text)}</h{b.level}>")
        elif b.kind == "para":
            parts.append(f"<p>{escape(b.text)}</p>")
        elif b.kind == "table":
            head, *body = b.rows
            parts.append("<table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in head) + "</tr></thead><tbody>")
            parts += ["<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in body]
            parts.append("</tbody></table>")
        elif b.kind == "figure":
            import base64
            uri = "data:image/png;base64," + base64.b64encode((out_dir / b.image).read_bytes()).decode()
            parts.append(f'<figure><img src="{uri}" alt="{escape(b.text)}">'
                         f"<figcaption>{escape(b.text)}</figcaption></figure>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
