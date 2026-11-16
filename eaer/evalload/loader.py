"""评测专用的历史日期装载入口（开发文档 14.2）。

把事实表中的文档、版本、事实和事件按历史日期写入数据库，回溯类查询才有意义。
它调用与生产相同的生命周期服务，只是换成按事件日期推进的时钟；
只在 EAER_ENV=eval 时可用，生产环境不开放。
"""
from __future__ import annotations

import hashlib
import os
import uuid
from datetime import date, datetime, time, timedelta, timezone

import psycopg

from eaer.lifecycle.service import Lifecycle
from factset.model import FactSet

ACTOR = "eval-loader"
MODALITY = {"正文": "text", "表格": "table", "图表": "figure"}
GRADE = {"正文写明": "explicit", "仅上传时间": "inferred", "无": "unknown", "不适用": "unknown"}
DOC_TYPE = {"规范类": "normative", "记录类": "record"}


class EvalLoader:
    def __init__(self, conn: psycopg.Connection, tenant_id: str = "eval"):
        if os.environ.get("EAER_ENV") != "eval":
            raise PermissionError("历史日期装载入口只在评测环境启用（EAER_ENV=eval）")
        self.conn = conn
        self.tenant_id = tenant_id
        self._now = datetime(1970, 1, 1, tzinfo=timezone.utc)
        self._seq = 0
        self.lc = Lifecycle(conn, clock=lambda: self._now)

    def _at(self, d: date) -> None:
        """把时钟拨到某日；同一天内的操作按顺序递增一微秒，保证时间严格递增。"""
        self._seq += 1
        self._now = datetime.combine(d, time.min, tzinfo=timezone.utc) + timedelta(microseconds=self._seq)

    def load(self, fs: FactSet, files: dict[str, str] | None = None) -> dict[str, uuid.UUID]:
        """装载事实表，返回事实表编号到数据库 id 的对照（文档与版本）。files 为版本编号到文档文件的对照。"""
        ids: dict[str, uuid.UUID] = {}
        first = min(v.uploaded_on for v in fs.versions.values())
        self._at(first)
        self.lc.ensure_tenant(self.tenant_id)
        for d in fs.documents.values():
            acl = self.lc.create_acl_policy(self.tenant_id, sorted(d.roles))
            ids[d.id] = self.lc.create_document(self.tenant_id, d.id, d.title, DOC_TYPE[d.doc_type], d.language)
            ids[f"acl:{d.id}"] = acl

        ops = []  # (日期, 顺序, 表中行号, 操作)
        for v in fs.versions.values():
            ops.append((v.uploaded_on, 0, v.row, "create", v))
            ops.append((v.approved_on, 1, v.row, "publish", v))
        for e in fs.events:
            ops.append((e.occurred_on, 2, e.row, "event", e))
        ops.sort(key=lambda o: o[:3])

        facts_by_version: dict[str, list] = {}
        for f in sorted(fs.facts.values(), key=lambda f: f.row):
            facts_by_version.setdefault(f.version_id, []).append(f)

        for day, _, _, kind, obj in ops:
            self._at(day)
            if kind == "create":
                v = obj
                vid = self.lc.add_version(
                    ids[v.document_id], version_no=v.version_no,
                    content_hash="sha256:" + hashlib.sha256(v.id.encode()).hexdigest(),
                    raw_object_uri=(files or {}).get(v.id, f"eval://{v.id}"),
                    acl_policy_id=ids[f"acl:{v.document_id}"], parser_version="eval",
                    doc_date=v.doc_date, import_batch_id="legacy-import" if v.valid_from_style == "无" else None)
                ids[v.id] = vid
                self.lc.set_draft_validity(vid, valid_from=v.valid_from, valid_from_grade=GRADE[v.valid_from_style])
                if v.supersedes:
                    self.lc.add_draft_relation("supersedes", vid, ids[v.supersedes], "explicit")
                for i, f in enumerate(facts_by_version.get(v.id, []), start=1):
                    self.lc.add_evidence(vid, MODALITY[f.carrier], {"page": 1, "seq": i},
                                         f"{f.key}：{f.value}{f.unit or ''}",
                                         "deep" if f.carrier == "图表" else "text")
            elif kind == "publish":
                self.lc.publish(ids[obj.id], ACTOR)
            else:
                self._event(fs, obj, ids)
        return ids

    def _event(self, fs: FactSet, e, ids) -> None:
        doc_id = e.target if e.target in fs.documents else fs.versions[e.target].document_id
        if e.type == "归档":
            self.lc.archive_document(ids[doc_id], ACTOR)
        elif e.type == "取消归档":
            self.lc.unarchive_document(ids[doc_id], ACTOR)
        elif e.type == "有效期更正":
            self.lc.correct_validity(ids[e.target], ACTOR, valid_from=e.new_valid_from,
                                     valid_from_grade=e.new_grade or ("explicit" if e.new_valid_from else None))
        elif e.type == "权限变更":
            self.lc.change_acl(ids[doc_id], sorted(e.new_roles), ACTOR)
        else:
            raise ValueError(f"未知事件类型 {e.type}")
