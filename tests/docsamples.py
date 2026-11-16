"""Docling 测试用的样例文件：文字版 PDF（两页）、扫描版 PDF（只有图像，需要 OCR）、PPTX（含表格）、XLSX。

全部在测试时生成，不依赖外部文件。中文字体按 factset.render 同样的顺序挑选。
"""
from __future__ import annotations

import io
from pathlib import Path

PARAS = ["第一条　为规范差旅报销，制定本办法。", "第二条　一线城市住宿上限为每晚650元。", "第三条　本办法自2026年1月1日起施行。"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    families = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "PingFang SC"):
        if name in families:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["pdf.fonttype"] = 42          # 嵌入 TrueType，PDF 中的文字可以提取
    return plt


def _first_page(fig) -> None:
    fig.text(0.1, 0.92, "差旅报销管理办法", fontsize=20)
    for i, p in enumerate(PARAS):
        fig.text(0.1, 0.86 - 0.04 * i, p, fontsize=12)


def make_all(out: Path) -> dict[str, Path]:
    from matplotlib.backends.backend_pdf import PdfPages

    plt = _plt()
    out.mkdir(parents=True, exist_ok=True)
    files = {k: out / v for k, v in {"pdf": "text.pdf", "scanned": "scanned.pdf", "pptx": "deck.pptx",
                                     "xlsx": "table.xlsx", "png": "scanned.png"}.items()}
    with PdfPages(files["pdf"]) as pdf:
        f = plt.figure(figsize=(8.27, 11.69))
        _first_page(f)
        pdf.savefig(f)
        plt.close(f)
        f = plt.figure(figsize=(8.27, 11.69))
        f.text(0.1, 0.9, "附则", fontsize=16)
        f.text(0.1, 0.85, "第四条　本办法由财务部负责解释。", fontsize=12)
        pdf.savefig(f)
        plt.close(f)

    f = plt.figure(figsize=(8.27, 11.69))
    _first_page(f)
    buf = io.BytesIO()
    f.savefig(buf, format="png", dpi=200)
    plt.close(f)
    buf.seek(0)
    img = plt.imread(buf)
    plt.imsave(files["png"], img)
    f = plt.figure(figsize=(8.27, 11.69))
    ax = f.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    f.savefig(files["scanned"])
    plt.close(f)

    from pptx import Presentation
    from pptx.util import Inches

    p = Presentation()
    s = p.slides.add_slide(p.slide_layouts[1])
    s.shapes.title.text = "差旅住宿标准"
    s.placeholders[1].text = "一线城市住宿上限为每晚650元"
    s = p.slides.add_slide(p.slide_layouts[5])
    s.shapes.title.text = "伙食补贴"
    t = s.shapes.add_table(3, 2, Inches(1), Inches(2), Inches(6), Inches(1.5)).table
    for r, row in enumerate([["项目", "标准"], ["伙食补贴", "80元/天"], ["交通补贴", "50元/天"]]):
        for c, v in enumerate(row):
            t.cell(r, c).text = v
    p.save(files["pptx"])

    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "住宿标准"
    for row in [["城市等级", "住宿上限（元/晚）"], ["一线城市", 650], ["二线城市", 450]]:
        ws.append(row)
    wb.save(files["xlsx"])
    return files
