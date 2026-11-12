"""从表格导出的 xlsx 读取事实表。

按表头文字定位列，列的顺序可以调整；表中多出的列（例如带公式的“预期来源等级”）被忽略。
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from openpyxl import load_workbook

from .model import Document, Event, Fact, FactSet, Issue, Query, Role, Version

# 每张表：表头文字 -> 字段名。带 * 的列必须存在。
COLUMNS: dict[str, dict[str, str]] = {
    "角色": {"*角色编号": "id", "说明": "description"},
    "文档": {"*文档编号": "id", "*标题": "title", "*文档类型": "doc_type",
             "语言": "language", "*可访问角色": "roles"},
    "版本": {"*版本编号": "id", "*文档编号": "document_id", "*版本号": "version_no",
             "*生效日": "valid_from", "*生效日写法": "valid_from_style",
             "*取代的版本": "supersedes", "*文档日期": "doc_date", "*上传日": "uploaded_on",
             "*审核通过日": "approved_on", "备注": "note"},
    "事实": {"*事实编号": "id", "*版本编号": "version_id", "*事实键": "key",
             "*取值": "value", "单位": "unit", "*载体": "carrier", "备注": "note"},
    "事件": {"*事件编号": "id", "*发生日": "occurred_on", "*事件类型": "type",
             "*对象": "target", "变更内容": "description", "*新生效日": "new_valid_from",
             "*新来源等级": "new_grade", "*新可访问角色": "new_roles"},
    "查询": {"*查询编号": "id", "*查询类型": "type", "*问题": "question",
             "*预期答案": "expected_answer", "*是否应拒答": "should_refuse",
             "*提问角色": "role", "*提问日": "asked_on", "*生效时间点": "as_of_valid",
             "*回溯时间点": "as_of_system", "*依据事实": "basis",
             "*预期来源等级": "expected_grade", "预期模式": "expected_mode",
             "说明": "note", "*目标事实键": "target_key"},
}


class LoadError(Exception):
    """表或必需列缺失，无法继续。"""


def _text(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _date(v: Any) -> date | None:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"无法识别的日期：{v!r}")


def _number(v: Any) -> int | float | str:
    if isinstance(v, bool):
        raise ValueError(f"取值不能是布尔值：{v!r}")
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else v
    s = str(v).strip().replace(",", "")
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return str(v).strip()  # 文本型事实


def _int(v: Any) -> int:
    n = _number(v)
    if not isinstance(n, int):
        raise ValueError(f"应为整数：{v!r}")
    return n


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().upper()
    if s in ("TRUE", "是", "1"):
        return True
    if s in ("FALSE", "否", "0", ""):
        return False
    raise ValueError(f"应为勾选框：{v!r}")


def _list(v: Any) -> tuple[str, ...]:
    s = _text(v)
    if s is None:
        return ()
    return tuple(p.strip() for p in re.split(r"[;；,，]", s) if p.strip())


def _rows(ws, sheet: str) -> list[tuple[int, dict[str, Any]]]:
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise LoadError(f"表“{sheet}”是空的")
    header = [_text(h) for h in rows[0]]
    spec = COLUMNS[sheet]
    index: dict[str, int] = {}
    for title, fld in spec.items():
        name = title.lstrip("*")
        if name in header:
            index[fld] = header.index(name)
        elif title.startswith("*"):
            raise LoadError(f"表“{sheet}”缺少列“{name}”")
    out = []
    for n, raw in enumerate(rows[1:], start=2):
        if all(c is None or (isinstance(c, str) and not c.strip()) for c in raw):
            continue
        out.append((n, {fld: (raw[i] if i < len(raw) else None) for fld, i in index.items()}))
    return out


def load(path: str | Path) -> FactSet:
    wb = load_workbook(path, data_only=True, read_only=True)
    for sheet in COLUMNS:
        if sheet not in wb.sheetnames:
            raise LoadError(f"缺少表“{sheet}”")
    fs = FactSet()

    def build(sheet: str, n: int, ctor: Callable[[], Any], store: dict | list, key: str | None) -> None:
        try:
            rec = ctor()
        except ValueError as e:
            fs.load_issues.append(Issue(sheet, n, str(e)))
            return
        if isinstance(store, dict):
            if not key:
                fs.load_issues.append(Issue(sheet, n, "编号为空"))
            elif key in store:
                fs.load_issues.append(Issue(sheet, n, f"编号 {key} 重复（首次出现在第 {store[key].row} 行）"))
            else:
                store[key] = rec
        else:
            if any(r.id == key for r in store):
                fs.load_issues.append(Issue(sheet, n, f"编号 {key} 重复"))
            else:
                store.append(rec)

    for n, r in _rows(wb["角色"], "角色"):
        build("角色", n, lambda: Role(n, _text(r["id"]), _text(r.get("description"))),
              fs.roles, _text(r["id"]))
    for n, r in _rows(wb["文档"], "文档"):
        build("文档", n, lambda: Document(n, _text(r["id"]), _text(r["title"]), _text(r["doc_type"]),
                                         _text(r.get("language")), frozenset(_list(r["roles"]))),
              fs.documents, _text(r["id"]))
    for n, r in _rows(wb["版本"], "版本"):
        build("版本", n, lambda: Version(n, _text(r["id"]), _text(r["document_id"]), _int(r["version_no"]),
                                        _date(r["valid_from"]), _text(r["valid_from_style"]),
                                        _text(r["supersedes"]), _date(r["doc_date"]),
                                        _date(r["uploaded_on"]), _date(r["approved_on"]),
                                        _text(r.get("note"))),
              fs.versions, _text(r["id"]))
    for n, r in _rows(wb["事实"], "事实"):
        build("事实", n, lambda: Fact(n, _text(r["id"]), _text(r["version_id"]), _text(r["key"]),
                                     _number(r["value"]) if r["value"] is not None else None,
                                     _text(r.get("unit")), _text(r["carrier"]), _text(r.get("note"))),
              fs.facts, _text(r["id"]))
    for n, r in _rows(wb["事件"], "事件"):
        build("事件", n, lambda: Event(n, _text(r["id"]), _date(r["occurred_on"]), _text(r["type"]),
                                      _text(r["target"]), _text(r.get("description")),
                                      _date(r["new_valid_from"]), _text(r["new_grade"]),
                                      frozenset(_list(r["new_roles"])) or None),
              fs.events, _text(r["id"]))
    for n, r in _rows(wb["查询"], "查询"):
        build("查询", n, lambda: Query(n, _text(r["id"]), _text(r["type"]), _text(r["question"]),
                                      _text(r["expected_answer"]), _bool(r["should_refuse"]),
                                      _text(r["role"]), _date(r["asked_on"]), _date(r["as_of_valid"]),
                                      _date(r["as_of_system"]), _list(r["basis"]),
                                      _text(r["expected_grade"]), _text(r.get("expected_mode")),
                                      _text(r["target_key"]), _text(r.get("note"))),
              fs.queries, _text(r["id"]))
    wb.close()
    # 事件按发生日排序，同一天按表中顺序
    fs.events.sort(key=lambda e: (e.occurred_on or date.min, e.row))
    return fs
