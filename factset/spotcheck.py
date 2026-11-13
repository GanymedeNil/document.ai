"""人工抽检单（开发文档 14.2 第 6 步）：每个类别按编号哈希抽取至少 rate 的问答对（每类至少 2 条），
列出问题、提问条件、程序算出的标准答案与依据事实的原始记录，供人工逐条核对后冻结评测集版本。"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook

from .answer import Calculator
from .model import FactSet

HEADER = ["查询编号", "类型", "集合", "问题", "提问角色", "提问日", "生效时间点", "回溯时间点", "标准答案", "是否应拒答",
          "依据事实", "核对结果（对/错）", "备注"]


def spot_check(fs: FactSet, calc: Calculator, path: Path, rate: float = 0.05) -> int:
    by_type = defaultdict(list)
    for q in sorted(fs.queries, key=lambda q: hashlib.sha1(f"spot:{q.id}".encode()).hexdigest()):
        by_type[q.type].append(q)
    wb = Workbook()
    ws = wb.active
    ws.title = "抽检"
    ws.append(HEADER)
    n = 0
    for qtype in sorted(by_type):
        group = by_type[qtype]
        for q in group[:max(2, math.ceil(len(group) * rate))]:
            e = calc.compute(q)
            basis = "；".join(f"{f.id}（{f.version_id}，{f.key}={f.value}{f.unit or ''}）"
                             for f in (fs.facts[i] for i in e.basis))
            ws.append([q.id, q.type, q.split, q.question, q.role, q.asked_on, q.as_of_valid, q.as_of_system,
                       e.answer, "是" if e.refuse else "否", basis, None, e.reason or None])
            n += 1
    wb.save(path)
    return n
