"""入库吞吐（开发文档 6.7 AC-6-10）与机器规格对照表的测量（13.5 AC-13-6）：精简版单机每小时处理页数，按 6.1 各格式混合。

    EAER_ENV=eval python -m eaer.evalrun.throughput --admin-url postgresql://… --opensearch http://localhost:9200 \\
        --pages 700 [--pages-per-doc 5] [--formats html,docx,pdf,scan,pptx,xlsx,png] [--vision] [--out 结果.json]

生成一批合成制度文档，各格式页数相同：条款正文，每三页一张表，--vision 时每份一张多单位图表。按生产的流程入库：
接收（存原件、去重）→ 文本解析、分类与抽取 → 审核通过 → 写入索引（真实 Embedding 服务）→ 深度解析（表格；--vision
时图表交给看图模型）。后台只有一个执行者，与精简版的后台任务进程相同。页的口径：PDF 与扫描件按物理页，PPTX 按幻灯片，
XLSX 按工作表，图片每张一页，Word 与 HTML 按分页符（每页十条左右的条款）。

统计：每种格式各阶段的耗时与每小时页数，混合后的每小时页数（AC-6-10）；数据库、索引、原件存储的大小（折算到每千页），
本进程与模型服务进程的内存峰值；Docling 首次加载模型的耗时单列（只在进程启动时发生一次）；Embedding 条数、看图调用次数
与 token 数（估算费用用：乘以所用模型的单价）。Embedding 用 EAER_EMBED_* 配置的服务；不配置不能运行（测试替身的速度
没有意义）。文档由本工具按模板生成，不涉及任何评测数据集。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

FORMATS = ["html", "docx", "pdf", "scan", "pptx", "xlsx", "png"]
SUFFIX = {"html": "html", "docx": "docx", "pdf": "pdf", "scan": "pdf", "pptx": "pptx", "xlsx": "xlsx", "png": "png"}
TOPICS = ["差旅住宿", "会议组织", "办公用品采购", "车辆使用", "加班审批", "培训报销", "印章管理", "档案借阅",
          "资产盘点", "合同审批", "来访接待", "信息安全"]
LINES_PER_PAGE = 10


# ---------- 合成文档 ----------

def _cn(n: int) -> str:
    d = "零一二三四五六七八九"
    if n < 10:
        return d[n]
    if n < 100:
        t, o = divmod(n, 10)
        return ("" if t == 1 else d[t]) + "十" + (d[o] if o else "")
    h, rest = divmod(n, 100)
    return d[h] + "百" + ("" if rest == 0 else ("零" + d[rest] if rest < 10 else _cn(rest) if rest >= 20 else "一" + _cn(rest)))


@dataclass
class Page:
    lines: list[str]
    table: list[list[str]] | None = None


@dataclass
class Doc:
    title: str
    pages: list[Page]
    chart: list[tuple[str, float, str]] | None = None


def make_doc(i: int, n_pages: int, *, chart: bool) -> Doc:
    topic = TOPICS[i % len(TOPICS)]
    pages, art = [], 1
    for p in range(n_pages):
        lines = []
        for _ in range(LINES_PER_PAGE):
            if art == 1:
                lines.append(f"第一条　为规范{topic}工作（第{i}号），制定本办法。")
            else:
                lines.append(f"第{_cn(art)}条　{topic}事项第{art}项的标准为每次{(i * 37 + art * 11) % 900 + 100}元，"
                             f"由第{art % 5 + 1}级负责人审批，审批期限为{art % 7 + 1}个工作日。")
            art += 1
        table = None
        if p % 3 == 2:
            table = [["项目", "标准", "审批层级"]] + [[f"{topic}第{p}{r}类", f"{(i + p * 7 + r * 13) % 500 + 50}元",
                                                   f"第{r % 4 + 1}级"] for r in range(1, 5)]
        pages.append(Page(lines, table))
    pages[-1].lines[-1] = "本办法自2026年1月1日起施行。"
    data = [(f"{topic}总费用", (i * 13) % 700 + 100, "万元"), (f"{topic}预算执行率", (i * 7) % 60 + 40, "%"),
            (f"{topic}报销笔数", (i * 97) % 3000 + 500, "笔")] if chart else None
    return Doc(f"{topic}管理办法（第{i}号）", pages, data)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    families = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "PingFang SC",
                 "Hiragino Sans GB", "Arial Unicode MS"):
        if name in families:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["pdf.fonttype"] = 42
    return plt


def chart_png(doc: Doc) -> bytes:
    from factset.render import chart_labels

    plt = _plt()
    labels, values = [d[0] for d in doc.chart], [d[1] for d in doc.chart]
    texts, ylabel = chart_labels(doc.chart)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    bars = ax.bar(labels, values, color="#4c78a8", width=0.5)
    ax.bar_label(bars, labels=texts, padding=3)
    ax.set_ylabel(ylabel)
    ax.set_title(f"图1 {doc.title}执行情况")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _page_figure(doc: Doc, k: int, page: Page):
    plt = _plt()
    f = plt.figure(figsize=(8.27, 11.69))
    y = 0.94
    if k == 0:
        f.text(0.08, y, doc.title, fontsize=18)
        y -= 0.05
    for line in page.lines:
        f.text(0.08, y, line, fontsize=9.5)
        y -= 0.035
    if page.table:
        ax = f.add_axes([0.08, max(0.05, y - 0.22), 0.84, 0.2])
        ax.axis("off")
        ax.table(cellText=page.table, loc="upper left", cellLoc="left")
    return plt, f


def render(doc: Doc, fmt: str) -> bytes:
    """一份文档按格式渲染成字节；图表（--vision）放在第一页之后。"""
    if fmt == "html":
        import base64
        parts = [f"<h1>{doc.title}</h1>"]
        for k, p in enumerate(doc.pages):
            parts += [f"<p>{x}</p>" for x in p.lines]
            if p.table:
                parts.append("<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in p.table)
                             + "</table>")
            if k == 0 and doc.chart:
                parts.append(f'<figure><img src="data:image/png;base64,{base64.b64encode(chart_png(doc)).decode()}" '
                             f'alt="图1"><figcaption>图1</figcaption></figure>')
            parts.append('<hr class="page-break">')
        return ("<!doctype html><html><body>" + "".join(parts) + "</body></html>").encode("utf-8")
    if fmt == "docx":
        from docx import Document
        from docx.enum.text import WD_BREAK
        from docx.shared import Cm
        d = Document()
        d.add_heading(doc.title, 1)
        for k, p in enumerate(doc.pages):
            for x in p.lines:
                d.add_paragraph(x)
            if p.table:
                t = d.add_table(rows=len(p.table), cols=3)
                for r, row in enumerate(p.table):
                    for c, v in enumerate(row):
                        t.cell(r, c).text = v
            if k == 0 and doc.chart:
                d.add_picture(io.BytesIO(chart_png(doc)), width=Cm(12))
            if k < len(doc.pages) - 1:
                d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        buf = io.BytesIO()
        d.save(buf)
        return buf.getvalue()
    if fmt in ("pdf", "scan", "png"):
        from matplotlib.backends.backend_pdf import PdfPages
        pages = doc.pages[:1] if fmt == "png" else doc.pages
        images, buf = [], io.BytesIO()
        with PdfPages(buf) as pdf:
            for k, p in enumerate(pages):
                plt, f = _page_figure(doc, k, p)
                if fmt == "pdf":
                    pdf.savefig(f)
                else:
                    img = io.BytesIO()
                    f.savefig(img, format="png", dpi=150)
                    images.append(img.getvalue())
                plt.close(f)
        if fmt == "pdf":
            return buf.getvalue()
        if fmt == "png":
            return images[0]
        from PIL import Image
        frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        out = io.BytesIO()
        frames[0].save(out, format="PDF", save_all=True, append_images=frames[1:], resolution=150)
        return out.getvalue()
    if fmt == "pptx":
        from pptx import Presentation
        from pptx.util import Inches, Pt
        prs = Presentation()
        for k, p in enumerate(doc.pages):
            s = prs.slides.add_slide(prs.slide_layouts[5])
            s.shapes.title.text = doc.title if k == 0 else f"{doc.title}（{k + 1}）"
            box = s.shapes.add_textbox(Inches(0.4), Inches(1.3), Inches(9.2), Inches(3.4)).text_frame
            box.word_wrap = True
            for j, x in enumerate(p.lines):
                para = box.paragraphs[0] if j == 0 else box.add_paragraph()
                para.text, para.font.size = x, Pt(10)
            if p.table:
                t = s.shapes.add_table(len(p.table), 3, Inches(0.4), Inches(4.9), Inches(9.2), Inches(1.8)).table
                for r, row in enumerate(p.table):
                    for c, v in enumerate(row):
                        t.cell(r, c).text = v
        out = io.BytesIO()
        prs.save(out)
        return out.getvalue()
    if fmt == "xlsx":
        from openpyxl import Workbook
        wb = Workbook()
        wb.remove(wb.active)
        for k, p in enumerate(doc.pages):
            ws = wb.create_sheet(f"第{k + 1}页")
            ws.append(["条款"])
            for x in p.lines:
                ws.append([x])
            for row in p.table or []:
                ws.append(row)
        out = io.BytesIO()
        wb.save(out)
        return out.getvalue()
    raise ValueError(fmt)


# ---------- 测量 ----------

@dataclass
class FormatStats:
    fmt: str
    docs: int = 0
    pages: int = 0
    seconds: dict[str, float] = field(default_factory=dict)        # receive / parse_text / classify_extract / …
    failed: int = 0

    @property
    def total(self) -> float:
        return sum(self.seconds.values())

    def per_hour(self) -> float | None:
        return round(self.pages / self.total * 3600) if self.total else None


class Peak:
    """后台线程每秒采样本进程与模型服务进程的物理内存，记峰值。"""

    def __init__(self):
        from modelserver.memguard import footprint
        self.footprint, self.me, self.model, self._stop = footprint, 0, 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            from modelserver.watchdog import scan          # 需要 psutil（requirements-docling.txt）
        except ImportError:
            scan = None
        while not self._stop.is_set():
            self.me = max(self.me, self.footprint(os.getpid()))
            if scan is not None:
                self.model = max(self.model, sum(p.bytes for p in scan(["modelserver"])))
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(5)


def run(db_url: str, client, alias: str, store_dir: Path, embedder, *, pages: int, pages_per_doc: int,
        formats: list[str], figure_reader=None, log=lambda s: None) -> dict:
    import psycopg
    from psycopg.rows import dict_row

    from eaer.adapters.online import collect_calls, summarize
    from eaer.ingestion.indexer import Indexer
    from eaer.ingestion.pipeline import Ingestion, Worker
    from eaer.lifecycle.service import Lifecycle
    from eaer.retrieval.opensearch import OpenSearchIndex
    from eaer.review.service import Review
    from eaer.storage import LocalFileStore

    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    store = LocalFileStore(store_dir)
    ing = Ingestion(conn, store, figure_reader=figure_reader)
    lc, review = Lifecycle(conn), Review(conn, Lifecycle(conn), store)
    lc.ensure_tenant("t1")
    acl = lc.create_acl_policy("t1", ["R_STAFF"])
    worker = Worker(ing, Indexer(conn, OpenSearchIndex(client, alias=alias), embedder, "tp"))

    # Docling 首次加载模型：单独计时，不算进吞吐
    warm = {}
    for fmt in [f for f in formats if f in ("pdf", "scan", "pptx", "xlsx", "png")][:1]:
        t = time.monotonic()
        from eaer.ingestion.parsers import parser_for
        parser_for(f"w.{SUFFIX[fmt]}").parse(render(make_doc(10 ** 6, 1, chart=False), fmt), f"w.{SUFFIX[fmt]}")
        warm["docling_first_load_seconds"] = round(time.monotonic() - t, 1)

    stats: dict[str, FormatStats] = {}
    current: list[FormatStats] = []

    def timed(stage, fn):
        def run_(version_id):
            t = time.monotonic()
            try:
                return fn(version_id)
            finally:
                s = current[0].seconds
                s[stage] = s.get(stage, 0.0) + time.monotonic() - t
        return run_

    for stage, fn in list(worker.handlers.items()):
        worker.handlers[stage] = timed(stage, fn)

    per_format = max(1, pages // (pages_per_doc * len(formats)))
    gen_seconds = 0.0
    with Peak() as peak, collect_calls() as calls:
        for fi, fmt in enumerate(formats):
            st = stats[fmt] = FormatStats(fmt)
            current[:] = [st]
            t = time.monotonic()
            blobs = []
            for k in range(per_format * (pages_per_doc if fmt == "png" else 1)):      # 图片每张一页：张数补足，各格式页数相同
                n = 1 if fmt == "png" else pages_per_doc
                doc = make_doc(fi * 100000 + k, n, chart=figure_reader is not None)
                blobs.append((f"{fmt}-{k:05d}.{SUFFIX[fmt]}", render(doc, fmt), n))
            gen_seconds += time.monotonic() - t
            t = time.monotonic()
            vids = [ing.receive("t1", data, name, acl_policy_id=acl).version_id for name, data, _ in blobs]
            st.seconds["receive"] = time.monotonic() - t
            st.docs, st.pages = len(blobs), sum(n for *_, n in blobs)
            worker.run_until_settled()                                   # 文本解析、分类与抽取
            tasks = conn.execute("SELECT id FROM review_task WHERE status = 'open' AND version_id = ANY(%s)",
                                 (vids,)).fetchall()
            t = time.monotonic()
            res = review.decide("t1", [{"task_id": str(r["id"]), "decision": "approve",
                                        "corrections": {"doc_type": "normative"}} for r in tasks], "tp")
            st.seconds["approve"] = time.monotonic() - t
            worker.run_until_settled()                                   # 写入索引、深度解析、索引更新
            st.failed = conn.execute("SELECT count(*) AS n FROM ingest_job WHERE status = 'failed' AND version_id = ANY(%s)",
                                     (vids,)).fetchone()["n"] + sum(not x["ok"] for x in res)
            log(f"{fmt}：{st.docs} 份 {st.pages} 页，{st.total:.1f} 秒，每小时 {st.per_hour()} 页"
                + (f"，失败 {st.failed}" if st.failed else ""))
        cost = summarize(calls)

    client.indices.refresh(index=alias)
    index_bytes = sum(v["primaries"]["store"]["size_in_bytes"] for v in client.indices.stats(index=alias)["indices"].values())
    db_bytes = conn.execute("SELECT pg_database_size(current_database()) AS n").fetchone()["n"]
    evidence = conn.execute("SELECT modality, count(*) AS n FROM evidence GROUP BY modality").fetchall()
    conn.close()
    store_bytes = sum(p.stat().st_size for p in store_dir.rglob("*") if p.is_file())
    total_pages = sum(s.pages for s in stats.values())
    total_seconds = sum(s.total for s in stats.values())
    stages: dict[str, float] = {}
    for s in stats.values():
        for k, v in s.seconds.items():
            stages[k] = stages.get(k, 0.0) + v
    return {
        "pages": total_pages, "seconds": round(total_seconds, 1),
        "pages_per_hour": round(total_pages / total_seconds * 3600) if total_seconds else None,
        "by_format": {f: {"docs": s.docs, "pages": s.pages, "seconds": round(s.total, 1), "pages_per_hour": s.per_hour(),
                          "stages": {k: round(v, 1) for k, v in s.seconds.items()}, "failed": s.failed}
                      for f, s in stats.items()},
        "stages": {k: round(v, 1) for k, v in sorted(stages.items(), key=lambda x: -x[1])},
        "generation_seconds": round(gen_seconds, 1), **warm,
        "evidence": {r["modality"]: r["n"] for r in evidence},
        "bytes": {"database": db_bytes, "index": index_bytes, "files": store_bytes},
        "bytes_per_1000_pages": {k: round(v * 1000 / total_pages) for k, v in
                                 {"database": db_bytes, "index": index_bytes, "files": store_bytes}.items()} if total_pages else {},
        "peak_memory": {"this_process": peak.me, "model_servers": peak.model},
        "model_calls": cost.get("by_kind", {}), "embedder": embedder.name,
        "vision": figure_reader is not None,
    }


def report(r: dict) -> str:
    from modelserver.memguard import fmt as gb
    lines = [f"入库吞吐（AC-6-10）：{r['pages']} 页、{sum(f['docs'] for f in r['by_format'].values())} 份，各格式页数相同；"
             f"Embedding {r['embedder']}；看图 {'开' if r['vision'] else '关'}；后台一个执行者",
             f"混合：{r['seconds']} 秒，每小时 {r['pages_per_hour']} 页"
             + (f"（Docling 首次加载模型 {r['docling_first_load_seconds']} 秒，只在进程启动时一次，未计入）"
                if "docling_first_load_seconds" in r else ""),
             "", "| 格式 | 份数 | 页数 | 耗时（秒） | 每小时页数 | 主要耗时 | 失败 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for f, s in r["by_format"].items():
        top = "、".join(f"{k} {v}" for k, v in sorted(s["stages"].items(), key=lambda x: -x[1])[:3])
        lines.append(f"| {f} | {s['docs']} | {s['pages']} | {s['seconds']} | {s['pages_per_hour']} | {top} | {s['failed']} |")
    b, per = r["bytes"], r["bytes_per_1000_pages"]
    lines += ["", "各阶段合计（秒）：" + "、".join(f"{k} {v}" for k, v in r["stages"].items()),
              f"证据条数：{r['evidence']}",
              f"存储：数据库 {gb(b['database'])}、索引 {gb(b['index'])}、原件 {gb(b['files'])}；"
              f"每千页 数据库 {gb(per['database'])}、索引 {gb(per['index'])}、原件 {gb(per['files'])}",
              f"内存峰值：本进程 {gb(r['peak_memory']['this_process'])}，模型服务 {gb(r['peak_memory']['model_servers'])}",
              "模型调用：" + "；".join(f"{k} {v.get('calls')} 次、输入 {v.get('input_tokens')} token、输出 {v.get('output_tokens')} token"
                                    for k, v in r["model_calls"].items())]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from eaer.adapters.online import OnlineModelClient
    from eaer.evalrun.analyzers import scratch_databases
    from eaer.ingestion.indexer import fresh_index
    from eaer.retrieval.opensearch import connect

    p = argparse.ArgumentParser(prog="eaer.evalrun.throughput", description="入库吞吐与机器规格测量（AC-6-10、AC-13-6）")
    p.add_argument("--admin-url", required=True)
    p.add_argument("--opensearch", required=True)
    p.add_argument("--pages", type=int, default=700)
    p.add_argument("--pages-per-doc", type=int, default=5)
    p.add_argument("--formats", default=",".join(FORMATS))
    p.add_argument("--vision", action="store_true", help="每份文档带一张图表，深度解析时交给看图模型（需 EAER_GEN_VISION=1）")
    p.add_argument("--out", help="把结果（JSON）写入文件")
    args = p.parse_args(argv)
    if os.environ.get("EAER_ENV") != "eval":
        print("只在评测环境运行（EAER_ENV=eval）", file=sys.stderr)
        return 2
    model = OnlineModelClient.from_env()
    if model.emb is None:
        print("需要 Embedding 服务（EAER_EMBED_*）：测试替身的速度没有意义", file=sys.stderr)
        return 2
    reader = None
    if args.vision:
        from eaer.ingestion.figures import FigureReader
        reader = FigureReader(model)
    formats = [f for f in args.formats.split(",") if f]
    unknown = set(formats) - set(FORMATS)
    if unknown:
        print(f"不认识的格式：{sorted(unknown)}", file=sys.stderr)
        return 2
    client, alias = connect(args.opensearch), f"eaer-tp-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as tmp, scratch_databases(args.admin_url) as new_db, \
            fresh_index(client, alias, build_id="tp", dims=model.embedding_dims):
        r = run(new_db(), client, alias, Path(tmp) / "files", model.embedder(), pages=args.pages,
                pages_per_doc=args.pages_per_doc, formats=formats, figure_reader=reader,
                log=lambda s: print(s, file=sys.stderr, flush=True))
    print(report(r))
    if args.out:
        Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if not any(s["failed"] for s in r["by_format"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
