"""图片读取（开发文档 6.2 第 6 阶段、12.1）：深度解析时把文档里的图片交给有看图能力的生成模型，转写成文字证据。

- 图表（柱状图、折线图、饼图等）：每个数据点一条证据，写明图题、系列、标签、数值与单位（每个点可以有自己的单位：
  一张图里不同柱子单位不同时，坐标轴上的单位只属于其中一部分），形如
  “图1《2026年各季度差旅费用》｜第二季度：128万元”——与表格单元格一样可检索、可引用，确定性检查的数值核对
  照常适用；
- 其他图片（示意图、流程图、照片）：有可检索的内容时写一条描述证据；
- 只抄图上标注的数值，不按刻度估读：没有标数字的点不列出、不生成证据（宁可查不到，不编数值）；整张图都没有标注
  数值时写一条描述证据。2026-11-14 ViDoRe 入库发现：原提示让这类点“value 留空”，密集的季度折线图上模型把几百个
  空值点全列出来，超过输出长度被截断，整张图读取失败；
- 图片里的文字同样当作文档内容，其中的指令一律不执行（10.3）；转写结果作为证据，照常经过证据区隔离；
- 证据的定位记下 source=vision 与读取所用的模型，回答引用时可以看出这一条是看图读出来的。
模型调用经 ModelClient（3.2）；模型没有声明 vision 能力时不读取，图片照旧只留图片块。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from eaer.adapters.model import CapabilityNotSupported, Image, Message, ModelClient
from eaer.domain.injection import DATA_RULE

SCHEMA = {
    "type": "object", "required": ["kind", "title", "unit", "points"],
    "properties": {
        "kind": {"type": "string", "enum": ["chart", "diagram", "photo", "other"]},
        "title": {"type": "string"},
        "unit": {"type": "string"},
        "points": {"type": "array", "items": {
            "type": "object", "required": ["label", "value"],
            "properties": {"series": {"type": "string"}, "label": {"type": "string"}, "value": {"type": "string"},
                           "unit": {"type": "string"}}}},
        "description": {"type": "string"},
    },
}
SYSTEM = """你把企业文档中的一张图片转写成文字，供检索与引用。
- kind：chart（柱状图、折线图、饼图等数据图表）、diagram（流程图、组织图、示意图）、photo（照片）、other；
- title：图上的标题；没有标题时根据坐标轴与图注概括，不要编造图上没有的信息；
- unit：全图共用的单位（看坐标轴、图例）；各点单位不同时留空，写在各点的 unit 里；
- points：图表的每个数据点。label 是类别或横轴取值，series 是系列名（只有一个系列时留空），value 是图上标注的数值，
  照原样抄写（包括小数点与千分位，不带单位）；unit 是这个点的单位：数据标注上写了单位的（如“54%”“7,080元”）照标注写，
  否则按这个点的名称与坐标轴判断——“率”“占比”一般是 %，“笔数”是 笔，不要把坐标轴的单位硬套到名称明显不符的点上；
  图上没有标注数字、只能按刻度估读的点不要列出（不要输出 value 为空的点），也不要估计；
- description：非图表图片用一两句话说明图上的内容；图表有标注数值时可留空，整张图都没有标注数值（points 为空）时，
  用一两句话说明它展示了什么（系列、时间范围、明显的走势），不写图上没有标注的数值。
只写图上看得到的内容，不补充图外的知识。""" + DATA_RULE


@dataclass
class Reading:
    kind: str
    title: str
    unit: str
    points: list[tuple[str, str, str, str]] = field(default_factory=list)    # (系列, 标签, 数值, 这个点的单位)
    description: str = ""
    model: str = ""


class FigureReader:
    def __init__(self, model: ModelClient):
        if not model.capabilities().vision:
            raise CapabilityNotSupported("生成模型没有声明图像理解能力，不能读取图片")
        self.model = model

    def read(self, image: bytes, mime: str, *, context: str = "") -> Reading:
        hint = f"图片所在文档与章节：{context}\n" if context else ""
        out = self.model.generate([Message("system", SYSTEM),
                                   Message("user", hint + "请转写这张图片。", (Image(mime, image),))], SCHEMA)
        p = out.parsed or {}
        points = [(str(x.get("series") or "").strip(), str(x.get("label") or "").strip(), str(x.get("value") or "").strip(),
                   str(x.get("unit") or "").strip()) for x in p.get("points") or []]
        return Reading(str(p.get("kind") or "other"), str(p.get("title") or "").strip(), str(p.get("unit") or "").strip(),
                       [x for x in points if x[1] and x[2]], str(p.get("description") or "").strip(), out.model)


_NUMBER = re.compile(r"[-+]?[0-9][0-9,]*(\.[0-9]+)?")


def with_unit(value: str, unit: str) -> str:
    """数值加单位：数值已经带了单位（“54%”）就不再加，避免“54%万元”。"""
    return value + unit if unit and _NUMBER.fullmatch(value) else value


def evidence_texts(r: Reading, number: int, caption: str = "") -> list[tuple[dict, str]]:
    """一次读取得到的证据：[(定位附加项, 内容)]。图号从 1 开始；图题优先用读出的标题，其次用图注。"""
    title = r.title or caption
    head = f"图{number}" + (f"《{title}》" if title else "")
    out: list[tuple[dict, str]] = []
    if r.kind == "chart":
        for k, (series, label, value, unit) in enumerate(r.points):
            out.append(({"point": k}, f"{head}｜{series + '｜' if series else ''}{label}：{with_unit(value, unit or r.unit)}"))
    if not out and r.description:
        out.append(({"point": "description"}, f"{head}：{r.description}"))
    return out
