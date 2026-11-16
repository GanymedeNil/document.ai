"""评测时取查询子集（开发文档 14.3）：指标只在测试集上报；试跑可以每类抽若干条。抽样按编号哈希，确定可复现。"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import replace

from .model import FactSet


def subset(fs: FactSet, *, split: str | None = None, per_type: int | None = None, salt: str = "eaer",
           types: set[str] | None = None) -> FactSet:
    """返回只含所选查询的事实表副本；文档、版本、事实、事件不变（按日期回放需要它们）。
    types 只留这几类查询；先按类过滤再按类抽样，所以同一类抽到的查询与不过滤时相同。"""
    qs = [q for q in fs.queries if (split is None or q.split == split) and (not types or q.type in types)]
    if per_type is not None:
        by_type = defaultdict(list)
        for q in sorted(qs, key=lambda q: hashlib.sha1(f"{salt}:{q.id}".encode()).hexdigest()):
            by_type[q.type].append(q)
        keep = {q.id for group in by_type.values() for q in group[:per_type]}
        qs = [q for q in qs if q.id in keep]
    return replace(fs, queries=qs)
