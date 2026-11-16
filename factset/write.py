"""把事实表写成 xlsx（与在线模板同样的表名与表头），供生成器输出、load 读回。"""
from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from .load import COLUMNS
from .model import FactSet


def _roles(rs) -> str:
    return "；".join(sorted(rs)) if rs else None


def write(fs: FactSet, path: str | Path) -> None:
    rows = {
        "角色": [{"id": r.id, "description": r.description} for r in fs.roles.values()],
        "文档": [{"id": d.id, "title": d.title, "doc_type": d.doc_type, "language": d.language,
                  "roles": _roles(d.roles)} for d in fs.documents.values()],
        "版本": [{"id": v.id, "document_id": v.document_id, "version_no": v.version_no, "valid_from": v.valid_from,
                  "valid_from_style": v.valid_from_style, "supersedes": v.supersedes, "doc_date": v.doc_date,
                  "uploaded_on": v.uploaded_on, "approved_on": v.approved_on, "note": v.note}
                 for v in fs.versions.values()],
        "事实": [{"id": f.id, "version_id": f.version_id, "key": f.key, "value": f.value, "unit": f.unit,
                  "carrier": f.carrier, "note": f.note} for f in fs.facts.values()],
        "事件": [{"id": e.id, "occurred_on": e.occurred_on, "type": e.type, "target": e.target,
                  "description": e.description, "new_valid_from": e.new_valid_from, "new_grade": e.new_grade,
                  "new_roles": _roles(e.new_roles)} for e in fs.events],
        "查询": [{"id": q.id, "type": q.type, "question": q.question, "expected_answer": q.expected_answer,
                  "should_refuse": q.should_refuse, "role": q.role, "asked_on": q.asked_on,
                  "as_of_valid": q.as_of_valid, "as_of_system": q.as_of_system, "basis": "；".join(q.basis) or None,
                  "expected_grade": q.expected_grade, "expected_mode": q.expected_mode, "note": q.note,
                  "target_key": q.target_key, "split": q.split} for q in fs.queries],
    }
    wb = Workbook()
    wb.remove(wb.active)
    for sheet, spec in COLUMNS.items():
        ws = wb.create_sheet(sheet)
        ws.append([title.lstrip("*") for title in spec])
        for rec in rows[sheet]:
            ws.append([rec.get(field) for field in spec.values()])
    wb.save(path)
