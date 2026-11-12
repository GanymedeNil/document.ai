"""事实表的数据模型。

每张表的一行对应一个记录；row 字段记录它在表格中的行号，用于报错定位。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# 来源等级，数值越大越可信
GRADE_RANK = {"explicit": 2, "inferred": 1, "unknown": 0}

# 版本表“生效日写法”到来源等级；记录类文档为“不适用”，没有等级
STYLE_TO_GRADE = {"正文写明": "explicit", "仅上传时间": "inferred", "无": "unknown", "不适用": None}

DOC_TYPES = {"规范类", "记录类"}
CARRIERS = {"正文", "表格", "图表"}
EVENT_TYPES = {"归档", "取消归档", "有效期更正", "权限变更"}
QUERY_TYPES = {"当前生效", "历史时点", "回溯", "版本差异", "冲突", "归档",
               "权限边界", "无答案", "图表", "验证器"}
MODES = {"Fast", "Deep", "Verified"}


def weakest(*grades: str | None) -> str | None:
    """返回几个来源等级中最弱的一个；忽略 None。"""
    present = [g for g in grades if g is not None]
    if not present:
        return None
    return min(present, key=lambda g: GRADE_RANK[g])


@dataclass(frozen=True)
class Role:
    row: int
    id: str
    description: str | None


@dataclass(frozen=True)
class Document:
    row: int
    id: str
    title: str
    doc_type: str
    language: str | None
    roles: frozenset[str]

    @property
    def normative(self) -> bool:
        return self.doc_type == "规范类"


@dataclass(frozen=True)
class Version:
    row: int
    id: str
    document_id: str
    version_no: int
    valid_from: date | None
    valid_from_style: str
    supersedes: str | None
    doc_date: date | None
    uploaded_on: date
    approved_on: date
    note: str | None

    @property
    def grade(self) -> str | None:
        return STYLE_TO_GRADE.get(self.valid_from_style)


@dataclass(frozen=True)
class Fact:
    row: int
    id: str
    version_id: str
    key: str
    value: int | float | str
    unit: str | None
    carrier: str
    note: str | None


@dataclass(frozen=True)
class Event:
    row: int
    id: str
    occurred_on: date
    type: str
    target: str  # 文档编号或版本编号
    description: str | None
    new_valid_from: date | None
    new_grade: str | None
    new_roles: frozenset[str] | None


@dataclass(frozen=True)
class Query:
    row: int
    id: str
    type: str
    question: str
    expected_answer: str
    should_refuse: bool
    role: str
    asked_on: date
    as_of_valid: date | None
    as_of_system: date | None
    basis: tuple[str, ...]
    expected_grade: str | None
    expected_mode: str | None
    target_key: str
    note: str | None


@dataclass
class FactSet:
    roles: dict[str, Role] = field(default_factory=dict)
    documents: dict[str, Document] = field(default_factory=dict)
    versions: dict[str, Version] = field(default_factory=dict)
    facts: dict[str, Fact] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)
    # 加载阶段发现的重复编号等问题，由校验统一报告
    load_issues: list["Issue"] = field(default_factory=list)


@dataclass(frozen=True)
class Issue:
    sheet: str
    row: int | None
    message: str

    def __str__(self) -> str:
        where = f"{self.sheet} 第 {self.row} 行" if self.row else self.sheet
        return f"[{where}] {self.message}"
