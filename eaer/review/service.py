"""审核（开发文档 6.3）：列出任务，逐份或批量提交决定。

批量提交时每份单独成事务，返回每份的结果；一份失败不影响其他份。
审核人补录的时间记为 explicit，并记录补录人。通过前必须已确认文档类型、抽取已完成。
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from eaer.lifecycle.service import InvalidState, Lifecycle
from eaer.storage import FileStore

GRADES = {"explicit", "inferred", "unknown"}
DOC_TYPES = {"normative", "record"}


class ReviewError(Exception):
    pass


class Review:
    def __init__(self, conn: psycopg.Connection, lifecycle: Lifecycle, store: FileStore):
        self.conn = conn
        self.lc = lifecycle
        self.store = store

    def list_tasks(self, tenant_id: str, status: str = "open", batch_id: uuid.UUID | None = None,
                   cursor: str | None = None, limit: int = 50) -> tuple[list[dict], str | None]:
        """按租户列出审核任务。已删除的任务只剩谁在何时删除：版本与文档的字段都为 null（6.3）。"""
        rows = self.conn.execute(
            "SELECT t.id AS task_id, t.batch_id, t.status, t.suggestions, t.corrections, t.reviewer, t.decided_at, "
            "  v.id AS version_id, v.version_no, v.uploaded_at, v.import_batch_id, "
            "  d.id AS document_id, d.title, d.doc_type, "
            "  (SELECT status FROM ingest_job j WHERE j.version_id = v.id AND j.stage = 'classify_extract' "
            "     ORDER BY created_at DESC LIMIT 1) AS extraction_status, "
            "  (SELECT row_to_json(x) FROM (SELECT valid_from, valid_from_grade, valid_to, valid_to_grade, "
            "     unresolved_text, locator FROM version_validity WHERE version_id = v.id AND recorded_at IS NULL) x) AS draft_validity "
            "FROM review_task t LEFT JOIN document_version v ON v.id = t.version_id "
            "  LEFT JOIN document d ON d.id = v.document_id "
            "WHERE t.tenant_id = %s AND t.status = %s AND (%s::uuid IS NULL OR t.batch_id = %s::uuid) "
            "AND (%s::text IS NULL OR t.id::text > %s::text) ORDER BY t.id::text LIMIT %s",
            (tenant_id, status, batch_id, batch_id, cursor, cursor, limit + 1)).fetchall()
        items = rows[:limit]
        nxt = str(items[-1]["task_id"]) if len(rows) > limit else None
        return items, nxt

    def decide(self, tenant_id: str, decisions: list[dict], reviewer: str) -> list[dict]:
        results = []
        for d in decisions:
            task_id = d.get("task_id")
            try:
                uri = None
                with self.conn.transaction():
                    uri = self._decide_one(tenant_id, d, reviewer)
                if uri:
                    self.store.delete(uri)   # 数据库删除提交后再删原件
                results.append({"task_id": task_id, "ok": True, "error": None})
            except (ReviewError, InvalidState) as e:
                results.append({"task_id": task_id, "ok": False, "error": str(e)})
            except (psycopg.errors.IntegrityError, psycopg.errors.RaiseException) as e:
                # 业务检查漏掉、被数据库不变量拒绝的情形：本份回滚并报错，其余份照常（6.3）
                msg = e.diag.message_primary or str(e).splitlines()[0]
                results.append({"task_id": task_id, "ok": False, "error": f"违反数据不变量：{msg}"})
        return results

    def _decide_one(self, tenant_id: str, d: dict, reviewer: str) -> str | None:
        try:
            tid = uuid.UUID(str(d.get("task_id")))
        except ValueError:
            raise ReviewError("task_id 格式错误")
        task = self.conn.execute(
            "SELECT t.*, v.document_id, d.doc_type FROM review_task t "
            "LEFT JOIN document_version v ON v.id = t.version_id LEFT JOIN document d ON d.id = v.document_id "
            "WHERE t.id = %s FOR UPDATE OF t", (tid,)).fetchone()
        # 按任务自己的租户判断：其他租户的任务（含已删除的）一律“不存在”
        if task is None or task["tenant_id"] != tenant_id:
            raise ReviewError("审核任务不存在")
        if task["status"] != "open":
            raise ReviewError(f"审核任务已处理（{task['status']}）")
        decision = d.get("decision")
        if decision == "delete":
            return self.lc.delete_pending(task["version_id"], reviewer)
        if decision != "approve":
            raise ReviewError("decision 只能是 approve 或 delete")
        job = self.conn.execute(
            "SELECT status FROM ingest_job WHERE version_id = %s AND stage = 'classify_extract' "
            "ORDER BY created_at DESC LIMIT 1", (task["version_id"],)).fetchone()
        if job is None or job["status"] != "done":
            raise ReviewError("抽取尚未完成，暂不能审核通过")
        corrections = d.get("corrections") or {}
        doc_type = self._apply(task, corrections, reviewer)
        if doc_type is None:
            raise ReviewError("文档类型未确认：请在 corrections.doc_type 中指定 normative 或 record")
        if corrections:
            self.conn.execute("UPDATE review_task SET corrections = %s WHERE id = %s", (Jsonb(corrections), tid))
        self.lc.publish(task["version_id"], reviewer)
        return None

    def _apply(self, task: dict, c: dict[str, Any], reviewer: str) -> str | None:
        vid, doc_type = task["version_id"], task["doc_type"]
        if "doc_type" in c:
            if c["doc_type"] not in DOC_TYPES:
                raise ReviewError("doc_type 只能是 normative 或 record")
            doc_type = c["doc_type"]
            self.conn.execute("UPDATE document SET doc_type = %s WHERE id = %s", (doc_type, task["document_id"]))
        if "valid_from" in c or "valid_to" in c:
            cur = self.conn.execute("SELECT * FROM version_validity WHERE version_id = %s AND recorded_at IS NULL",
                                    (vid,)).fetchone() or {}
            args = {k: cur.get(k) for k in ("valid_from", "valid_from_grade", "valid_to", "valid_to_grade",
                                             "valid_to_reason", "unresolved_text", "locator")}
            args["valid_from_grade"] = args["valid_from_grade"] or "unknown"
            args["valid_to_grade"] = args["valid_to_grade"] or "unknown"
            for side in ("valid_from", "valid_to"):
                if side not in c:
                    continue
                spec = c[side]
                if spec is None:
                    args[side], args[f"{side}_grade"] = None, "unknown"
                    continue
                value, grade = _date(spec.get("value")), spec.get("grade", "explicit")
                if grade not in GRADES:
                    raise ReviewError(f"{side}.grade 无效")
                args[side], args[f"{side}_grade"] = value, grade
                if side == "valid_to":
                    args["valid_to_reason"] = "stated" if value else None
            args["locator"] = {"corrected_by": reviewer}   # 补录：等级为 explicit，记录补录人
            args["unresolved_text"] = None if c.get("valid_from") else args["unresolved_text"]
            self.lc.set_draft_validity(vid, **args)
        if "relations" in c:
            self.lc.clear_draft_relations(vid)
            for r in c["relations"] or []:
                to = self.conn.execute(
                    "SELECT v.id FROM document_version v JOIN document d ON d.id = v.document_id "
                    "WHERE d.id = %s AND d.tenant_id = %s AND v.version_no = %s AND v.status <> 'pending_review'",
                    (r.get("to_document_id"), task["tenant_id"], r.get("to_version_no"))).fetchone()
                if to is None:
                    raise ReviewError(f"关系指向的版本不存在或未发布：{r}")
                self.lc.add_draft_relation(r.get("relation", "supersedes"), vid, to["id"], "explicit",
                                           {"corrected_by": reviewer})
        return doc_type


def _date(v) -> date | None:
    if v is None:
        return None
    try:
        return date.fromisoformat(str(v))
    except ValueError:
        raise ReviewError(f"日期格式应为 YYYY-MM-DD：{v}")
