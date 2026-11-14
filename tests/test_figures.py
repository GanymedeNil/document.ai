"""图片读取（开发文档 6.2 第 6 阶段、12.1、AC-14-6 的流程部分）。离线：脚本化的看图模型；真实模型的正确率见 CLAUDE.md。

- 解析器把图片本身带出来：Word 取嵌入的图片；HTML 只取内嵌的 data URI，外部地址不下载；按内容判断格式；
- 读取器只接受声明了看图能力的模型；图表每个数据点一条证据，没有标注数值的点不生成证据；非图表写一条描述；
- 深度解析：图片证据带 source=vision 与读取模型，能被检索到；已读过的图片不再读取；某张图读取失败时表格与
  其他图片照常写入并进索引，任务报错交给重试，重试只读失败的那张；没有配置读取器时跳过图片（与原来一致）；
- 后台任务进程按 EAER_GEN_VISION 决定是否读取图片。
"""
from __future__ import annotations

import base64
import io

import pytest

from eaer.adapters.model import Capabilities, CapabilityNotSupported, Completion, Usage
from eaer.ingestion.figures import FigureReader, Reading, evidence_texts
from eaer.ingestion.parsers import DocxParser, HtmlParser, sniff_image
from eaer.ingestion.pipeline import FigureReadError
from eaer.retrieval.service import Scope
from test_ingestion import env  # noqa: F401
from test_retrieval import ix  # noqa: F401


def png(text: str = "128") -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(2, 1.5), dpi=60)
    fig.text(0.5, 0.5, text, ha="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


class Vision:
    """脚本化的看图模型：按顺序返回预设的读取结果；元素为 dict 或要抛出的异常。"""

    def __init__(self, *readings, vision: bool = True):
        self.readings, self.vision, self.calls = list(readings), vision, []

    def capabilities(self):
        return Capabilities(True, 32000, self.vision, 0)

    def generate(self, messages, schema):
        imgs = [i for m in messages for i in m.images]
        self.calls.append((messages[-1].content, imgs))
        assert imgs and imgs[0].mime == "image/png"
        r = self.readings.pop(0)
        if isinstance(r, Exception):
            raise r
        return Completion("{}", r, Usage(400, 50), "vision-model")


CHART = {"kind": "chart", "title": "2026年各季度差旅费用", "unit": "万元",
         "points": [{"label": "第一季度", "value": "120"}, {"label": "第二季度", "value": "128"},
                    {"label": "第三季度", "value": ""}]}


def html_with(*images: bytes, table: bool = False) -> bytes:
    figs = "".join(f'<figure><img src="data:image/png;base64,{base64.b64encode(i).decode()}" alt="图{k + 1}">'
                   f"<figcaption>图{k + 1}</figcaption></figure>" for k, i in enumerate(images))
    tbl = ("<table><tr><th>项目</th><th>标准</th></tr><tr><td>一线城市住宿</td><td>650元</td></tr></table>"
           if table else "")
    return (f"<h1>2026年差旅费用分析报告</h1><p>报告日期：2026年7月10日</p>{tbl}{figs}").encode()


# ---------- 解析器 ----------

def test_parsers_carry_the_image():
    img = png()
    assert sniff_image(img) == "image/png" and sniff_image(b"GIF89a....") == "image/gif" and sniff_image(b"hello") is None
    blocks = HtmlParser().parse(html_with(img), "r.html").blocks
    (fig,) = [b for b in blocks if b.kind == "figure"]
    assert fig.image == img and fig.image_type == "image/png" and fig.text == "图1"
    external = HtmlParser().parse(b'<h1>t</h1><img src="https://example.com/c.png" alt="x">'
                                  b'<img src="data:image/png;base64,bm90IGFuIGltYWdl" alt="y">', "e.html").blocks
    assert [b.image for b in external if b.kind == "figure"] == [None, None]    # 不下载外部地址；不是图片的不收

    from docx import Document
    from docx.shared import Cm
    d = Document()
    d.add_heading("差旅费用分析报告", 1)
    d.add_picture(io.BytesIO(img), width=Cm(5))
    buf = io.BytesIO()
    d.save(buf)
    (fig,) = [b for b in DocxParser().parse(buf.getvalue(), "r.docx").blocks if b.kind == "figure"]
    assert fig.image == img and fig.image_type == "image/png"


# ---------- 读取器 ----------

def test_reader_requires_vision_and_formats_evidence():
    with pytest.raises(CapabilityNotSupported):
        FigureReader(Vision(vision=False))
    m = Vision(CHART)
    r = FigureReader(m).read(png(), "image/png", context="差旅费用分析报告 / 二、费用情况")
    assert "差旅费用分析报告 / 二、费用情况" in m.calls[0][0]
    assert r.points == [("", "第一季度", "120", ""), ("", "第二季度", "128", "")]   # 没标数值的点不要
    texts = evidence_texts(r, 2)
    assert [t for _, t in texts] == ["图2《2026年各季度差旅费用》｜第一季度：120万元", "图2《2026年各季度差旅费用》｜第二季度：128万元"]
    two = Reading("chart", "", "%", [("北京", "一线", "35%", "")])
    assert evidence_texts(two, 1, "图1 占比")[0][1] == "图1《图1 占比》｜北京｜一线：35%"            # 单位不重复
    # 一张图里单位不同：每个点自己的单位优先于坐标轴的单位；数值自带单位的不再加
    mixed = Reading("chart", "2026年第三季度通讯费用", "万元",
                    [("", "通讯总费用", "509", ""), ("", "预算执行率", "54", "%"), ("", "人均费用", "7,080元", ""),
                     ("", "报销笔数", "2,770", "笔")])
    assert [t for _, t in evidence_texts(mixed, 1)] == [
        "图1《2026年第三季度通讯费用》｜通讯总费用：509万元", "图1《2026年第三季度通讯费用》｜预算执行率：54%",
        "图1《2026年第三季度通讯费用》｜人均费用：7,080元", "图1《2026年第三季度通讯费用》｜报销笔数：2,770笔"]
    from eaer.ingestion.figures import SCHEMA, SYSTEM, with_unit
    assert "unit" in SCHEMA["properties"]["points"]["items"]["properties"] and "不要把坐标轴的单位硬套" in SYSTEM
    assert with_unit("54", "%") == "54%" and with_unit("54%", "万元") == "54%" and with_unit("-1.5", "℃") == "-1.5℃"
    photo = Reading("diagram", "审批流程", "", [], "部门经理审批后报财务复核", "m")
    assert evidence_texts(photo, 3) == [({"point": "description"}, "图3《审批流程》：部门经理审批后报财务复核")]
    assert evidence_texts(Reading("photo", "", "", []), 1) == []


# ---------- 深度解析 ----------

def publish(ix, data: bytes, name: str):
    r = ix.ing.receive("t1", data, name, acl_policy_id=ix.acl)
    ix.worker.run_until_idle()
    task = str(ix.conn.execute("SELECT id FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()["id"])
    assert ix.review.decide("t1", [{"task_id": task, "decision": "approve",
                                    "corrections": {"doc_type": "record"}}], "rv")[0]["ok"]
    return r


def figure_evidence(ix, vid):
    return ix.conn.execute("SELECT content, locator FROM evidence WHERE version_id = %s AND modality = 'figure' "
                           "ORDER BY content", (vid,)).fetchall()


def test_deep_parse_reads_figures_into_searchable_evidence(ix):
    m = Vision(CHART)
    ix.ing.figure_reader = FigureReader(m)
    r = publish(ix, html_with(png(), table=True), "r.html")
    ix.worker.run_until_idle()
    ev = figure_evidence(ix, r.version_id)
    assert [e["content"] for e in ev] == ["图1《2026年各季度差旅费用》｜第一季度：120万元",
                                         "图1《2026年各季度差旅费用》｜第二季度：128万元"]
    assert all(e["locator"]["source"] == "vision" and e["locator"]["reader"] == "vision-model"
               and e["locator"]["figure"] == 0 for e in ev)
    assert ix.conn.execute("SELECT count(*) AS n FROM evidence WHERE version_id = %s AND modality = 'table'",
                           (r.version_id,)).fetchone()["n"] == 1
    hits = ix.retriever.search(Scope("t1", frozenset({"R_STAFF"})), "第二季度差旅费用是多少", today=__import__("datetime").date(2026, 8, 1))
    assert any(h.modality == "figure" and "128万元" in h.content for h in hits)
    ix.ing.parse_deep(r.version_id)                                  # 再做一次深度解析：不再读取，不重复
    assert len(m.calls) == 1 and len(figure_evidence(ix, r.version_id)) == 2


def test_a_failed_figure_is_retried_alone_and_the_rest_is_kept(ix):
    second = {"kind": "chart", "title": "各城市住宿费用", "unit": "元", "points": [{"label": "北京", "value": "650"}]}
    m = Vision(TimeoutError("看图服务超时"), second, CHART)
    ix.ing.figure_reader = FigureReader(m)
    r = publish(ix, html_with(png("A"), png("B"), table=True), "r2.html")
    with pytest.raises(FigureReadError, match="图1"):
        ix.ing.parse_deep(r.version_id)
    ev = figure_evidence(ix, r.version_id)
    assert [e["locator"]["figure"] for e in ev] == [1]                          # 第二张照常写入
    assert ix.conn.execute("SELECT count(*) AS n FROM evidence WHERE version_id = %s AND modality = 'table'",
                           (r.version_id,)).fetchone()["n"] == 1
    assert ix.conn.execute("SELECT count(*) AS n FROM ingest_job WHERE version_id = %s AND stage = 'index_update' "
                           "AND status = 'queued'", (r.version_id,)).fetchone()["n"] == 1   # 已写入的照常进索引
    ix.ing.parse_deep(r.version_id)                                  # 重试：只读失败的第一张
    assert len(m.calls) == 3 and {e["locator"]["figure"] for e in figure_evidence(ix, r.version_id)} == {0, 1}


def test_without_a_reader_figures_are_skipped(ix):
    r = publish(ix, html_with(png()), "r3.html")
    ix.worker.run_until_idle()
    assert figure_evidence(ix, r.version_id) == []


def test_worker_reads_figures_only_when_vision_is_declared():
    from eaer.ingestion.__main__ import figure_reader_for
    base = {"EAER_GEN_BASE_URL": "http://127.0.0.1:1/v1", "EAER_GEN_API_KEY": "k", "EAER_GEN_MODEL": "m"}
    assert figure_reader_for(base) is None and figure_reader_for({}) is None
    assert isinstance(figure_reader_for({**base, "EAER_GEN_VISION": "1"}), FigureReader)
