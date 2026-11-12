"""文档生命周期服务（开发文档第 6 节）。

只有本模块改变版本状态。每个操作在一个事务内完成；时间由注入的时钟提供：
生产环境用当前时间，评测装载入口用历史时间，两者走同一套代码。
数据库触发器是第二道防线，本模块先做业务检查并给出可读的错误。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable

import psycopg
from psycopg.types.json import Jsonb

from eaer.domain.evidence import content_hash as hash_content
from eaer.domain.evidence import evidence_id

Clock = Callable[[], datetime]
MODALITIES = {"text", "table", "figure"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InvalidState(Exception):
    """操作与版本当前状态不符，对应接口的 409。"""


@dataclass(frozen=True)
class Validity:
    valid_from: date | None
    valid_from_grade: str
    valid_to: date | None
    valid_to_grade: str
    valid_to_reason: str | None
    unresolved_text: str | None
    locator: Any


class Lifecycle:
    def __init__(self, conn: psycopg.Connection, clock: Clock = utcnow):
        self.conn = conn
        self.clock = clock

    # ---------- 基础数据 ----------

    def ensure_tenant(self, tenant_id: str) -> None:
        self.conn.execute(
            "INSERT INTO tenant_config (tenant_id, updated_at) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (tenant_id, self.clock()))

    def create_acl_policy(self, tenant_id: str, principals: list[str], source: str = "reference") -> uuid.UUID:
        pid = uuid.uuid4()
        self.conn.execute(
            "INSERT INTO acl_policy (id, tenant_id, principals, source, updated_at) VALUES (%s, %s, %s, %s, %s)",
            (pid, tenant_id, sorted(set(principals)), source, self.clock()))
        return pid

    def create_document(self, tenant_id: str, source_ref: str, title: str,
                        doc_type: str | None = None, language: str | None = None) -> uuid.UUID:
        did = uuid.uuid4()
        self.conn.execute(
            "INSERT INTO document (id, tenant_id, source_ref, title, doc_type, language, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (did, tenant_id, source_ref, title, doc_type, language, self.clock()))
        return did

    # ---------- 入库与审核前 ----------

    def add_version(self, document_id: uuid.UUID, *, content_hash: str, raw_object_uri: str,
                    acl_policy_id: uuid.UUID, parser_version: str, version_no: int | None = None,
                    doc_date: date | None = None, import_batch_id: str | None = None,
                    source_created_at: datetime | None = None, batch_id: uuid.UUID | None = None) -> uuid.UUID:
        """新增一个待审核版本，并生成审核任务。"""
        now = self.clock()
        vid = uuid.uuid4()
        with self.conn.transaction():
            if version_no is None:
                row = self.conn.execute(
                    "SELECT coalesce(max(version_no), 0) + 1 AS n FROM document_version WHERE document_id = %s",
                    (document_id,)).fetchone()
                version_no = row["n"]
            self.conn.execute(
                "INSERT INTO document_version (id, document_id, version_no, status, content_hash, raw_object_uri, "
                "acl_policy_id, uploaded_at, source_created_at, import_batch_id, doc_date, parser_version) "
                "VALUES (%s, %s, %s, 'pending_review', %s, %s, %s, %s, %s, %s, %s, %s)",
                (vid, document_id, version_no, content_hash, raw_object_uri, acl_policy_id, now,
                 source_created_at, import_batch_id, doc_date, parser_version))
            self.conn.execute(
                "INSERT INTO review_task (id, version_id, batch_id, status) VALUES (%s, %s, %s, 'open')",
                (uuid.uuid4(), vid, batch_id))
        return vid

    def set_draft_validity(self, version_id: uuid.UUID, *, valid_from: date | None = None,
                           valid_from_grade: str = "unknown", valid_to: date | None = None,
                           valid_to_grade: str = "unknown", valid_to_reason: str | None = None,
                           unresolved_text: str | None = None, locator: Any = None) -> None:
        """写入或替换审核前的待确认有效期。"""
        with self.conn.transaction():
            self._require_status(version_id, {"pending_review"}, "只有待审核版本可以修改待确认值")
            self.conn.execute("DELETE FROM version_validity WHERE version_id = %s AND recorded_at IS NULL",
                              (version_id,))
            self.conn.execute(
                "INSERT INTO version_validity (id, version_id, valid_from, valid_from_grade, valid_to, "
                "valid_to_grade, valid_to_reason, unresolved_text, locator) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (uuid.uuid4(), version_id, valid_from, valid_from_grade, valid_to, valid_to_grade,
                 valid_to_reason, unresolved_text, _json(locator)))

    def add_draft_relation(self, relation: str, from_version_id: uuid.UUID, to_version_id: uuid.UUID,
                           grade: str = "explicit", locator: Any = None) -> uuid.UUID:
        rid = uuid.uuid4()
        with self.conn.transaction():
            self._require_status(from_version_id, {"pending_review"}, "关系只能加在待审核版本上")
            self.conn.execute(
                "INSERT INTO version_relation (id, relation, from_version_id, to_version_id, grade, locator) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (rid, relation, from_version_id, to_version_id, grade, _json(locator)))
        return rid

    def clear_draft_relations(self, version_id: uuid.UUID) -> None:
        """清除待审核版本上尚未记录的关系，供重新抽取时使用。"""
        with self.conn.transaction():
            self._require_status(version_id, {"pending_review"}, "只有待审核版本可以修改待确认值")
            self.conn.execute("DELETE FROM version_relation WHERE from_version_id = %s AND recorded_at IS NULL",
                              (version_id,))

    def add_evidence(self, version_id: uuid.UUID, modality: str, locator: dict, content: str,
                     parse_stage: str = "text") -> str:
        """写入一条证据；同一内容重复写入时沿用原 id（开发文档 4.3）。"""
        if modality not in MODALITIES:
            raise ValueError(f"未知模态 {modality}")
        h = hash_content(content)
        eid = evidence_id(str(version_id), modality, locator, h)
        self.conn.execute(
            "INSERT INTO evidence (id, version_id, modality, locator, content, content_hash, parse_stage, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
            (eid, version_id, modality, Jsonb(locator), content, h, parse_stage, self.clock()))
        return eid

    # ---------- 发布 ----------

    def publish(self, version_id: uuid.UUID, actor: str) -> None:
        """审核通过并发布（开发文档 6.4），一个事务内完成。"""
        now = self.clock()
        with self.conn.transaction():
            self._require_status(version_id, {"pending_review"}, "只有待审核版本可以发布", lock=True)
            self.conn.execute(
                "UPDATE document_version SET status = 'searchable', approved_at = %s WHERE id = %s",
                (now, version_id))
            cur = self.conn.execute(
                "UPDATE version_validity SET recorded_at = %s WHERE version_id = %s AND recorded_at IS NULL",
                (now, version_id))
            if cur.rowcount == 0:  # 没有抽取到任何时间：记为 unknown
                self.conn.execute(
                    "INSERT INTO version_validity (id, version_id, valid_from_grade, valid_to_grade, recorded_at) "
                    "VALUES (%s, %s, 'unknown', 'unknown', %s)", (uuid.uuid4(), version_id, now))
            self.conn.execute(
                "UPDATE version_relation SET recorded_at = %s WHERE from_version_id = %s AND recorded_at IS NULL",
                (now, version_id))
            self.conn.execute(
                "UPDATE review_task SET status = 'approved', reviewer = %s, decided_at = %s "
                "WHERE version_id = %s AND status = 'open'", (actor, now, version_id))
            self._close_predecessors(version_id, now)
            self._enqueue(version_id, "publish_index", now)
            self._enqueue(version_id, "parse_deep", now)        # 第 6 阶段不阻塞发布（开发文档 6.4）
            self._event(version_id, "published", actor, None, now)

    # ---------- 发布之后 ----------

    def archive_document(self, document_id: uuid.UUID, actor: str) -> list[uuid.UUID]:
        """归档以文档为单位，作用于全部可搜索版本（开发文档 6.6）。"""
        return self._set_archived(document_id, actor, archive=True)

    def unarchive_document(self, document_id: uuid.UUID, actor: str) -> list[uuid.UUID]:
        return self._set_archived(document_id, actor, archive=False)

    def correct_validity(self, version_id: uuid.UUID, actor: str, *, valid_from: date | None = None,
                         valid_from_grade: str | None = None) -> None:
        """有效期更正：新增一行并替换旧行；它若取代了别的版本，同步调整那个版本的作废日。"""
        now = self.clock()
        with self.conn.transaction():
            # 6.5：有效期更正只从“可搜索”出发；已归档的版本先取消归档再更正
            self._require_status(version_id, {"searchable"}, "只有可搜索的版本可以更正有效期，已归档的请先取消归档",
                                 lock=True)
            old = self._current_validity(version_id)
            new = Validity(valid_from if valid_from is not None else old.valid_from,
                           valid_from_grade or ("explicit" if valid_from is not None else old.valid_from_grade),
                           old.valid_to, old.valid_to_grade, old.valid_to_reason, None, {"corrected_by": actor})
            self._record(version_id, new, now)
            self._event(version_id, "validity_corrected", actor, {
                "old": {"valid_from": _iso(old.valid_from), "grade": old.valid_from_grade},
                "new": {"valid_from": _iso(new.valid_from), "grade": new.valid_from_grade}}, now)
            self._enqueue(version_id, "index_update", now)
            self._close_predecessors(version_id, now)

    def change_acl(self, document_id: uuid.UUID, principals: list[str], actor: str) -> uuid.UUID:
        """权限变更：新建策略行并让文档全部版本指向它，旧策略保留以备审计；先改数据库，再排入索引更新。

        6.5 表中只列了“可搜索 → 权限变更 → 可搜索”。权限是文档属性、不改变状态，所以待审核与已归档的版本
        同样跟随：否则撤权会在审核通过或取消归档时丢失（AC-10-3）。用户已确认，开发文档 6.5 表后已写明。
        """
        now = self.clock()
        with self.conn.transaction():
            doc = self.conn.execute("SELECT tenant_id FROM document WHERE id = %s", (document_id,)).fetchone()
            if doc is None:
                raise InvalidState(f"文档 {document_id} 不存在")
            rows = self.conn.execute(
                "SELECT v.id, v.status, p.principals FROM document_version v JOIN acl_policy p ON p.id = v.acl_policy_id "
                "WHERE v.document_id = %s FOR UPDATE OF v", (document_id,)).fetchall()
            pid = self.create_acl_policy(doc["tenant_id"], principals)
            for r in rows:
                self.conn.execute("UPDATE document_version SET acl_policy_id = %s WHERE id = %s", (pid, r["id"]))
                if r["status"] != "pending_review":
                    self._event(r["id"], "acl_changed", actor,
                                {"old": r["principals"], "new": sorted(set(principals))}, now)
                    self._enqueue(r["id"], "index_update", now)
            return pid

    def delete_pending(self, version_id: uuid.UUID, actor: str) -> str:
        """删除待审核版本：证据、待确认值、任务一并清除；审核记录保留谁在何时删除，不保留内容。

        返回原件位置，由调用方删除文件存储中的原件。
        """
        now = self.clock()
        with self.conn.transaction():
            v = self._require_status(version_id, {"pending_review"}, "只有待审核版本可以删除", lock=True)
            self.conn.execute(
                "UPDATE review_task SET status = 'deleted', version_id = NULL, corrections = NULL, suggestions = NULL, "
                "reviewer = %s, decided_at = %s WHERE version_id = %s", (actor, now, version_id))
            self.conn.execute("DELETE FROM document_version WHERE id = %s", (version_id,))
            left = self.conn.execute("SELECT count(*) AS n FROM document_version WHERE document_id = %s",
                                     (v["document_id"],)).fetchone()["n"]
            if left == 0:
                self.conn.execute("DELETE FROM document WHERE id = %s", (v["document_id"],))
            return v["raw_object_uri"]

    # ---------- 内部 ----------

    def _require_status(self, version_id, allowed: set[str], message: str, lock: bool = False) -> dict:
        sql = "SELECT * FROM document_version WHERE id = %s" + (" FOR UPDATE" if lock else "")
        row = self.conn.execute(sql, (version_id,)).fetchone()
        if row is None:
            raise InvalidState(f"版本 {version_id} 不存在")
        if row["status"] not in allowed:
            raise InvalidState(f"{message}（当前状态 {row['status']}）")
        return row

    def _set_archived(self, document_id, actor: str, archive: bool) -> list[uuid.UUID]:
        now = self.clock()
        src, dst = ("searchable", "archived") if archive else ("archived", "searchable")
        with self.conn.transaction():
            rows = self.conn.execute(
                "SELECT id FROM document_version WHERE document_id = %s AND status = %s FOR UPDATE",
                (document_id, src)).fetchall()
            if not rows:
                raise InvalidState(f"文档 {document_id} 没有可{'归档' if archive else '取消归档'}的版本")
            ids = [r["id"] for r in rows]
            self.conn.execute(
                "UPDATE document_version SET status = %s, archived_at = %s WHERE id = ANY(%s)",
                (dst, now if archive else None, ids))
            for vid in ids:
                self._event(vid, "archived" if archive else "unarchived", actor, None, now)
                self._enqueue(vid, "index_update", now)
            return ids

    def _current_validity(self, version_id) -> Validity:
        r = self.conn.execute(
            "SELECT * FROM version_validity WHERE version_id = %s AND recorded_at IS NOT NULL AND replaced_at IS NULL",
            (version_id,)).fetchone()
        if r is None:
            raise InvalidState(f"版本 {version_id} 没有当前有效期")
        return Validity(r["valid_from"], r["valid_from_grade"], r["valid_to"], r["valid_to_grade"],
                        r["valid_to_reason"], r["unresolved_text"], r["locator"])

    def _record(self, version_id, v: Validity, now: datetime) -> None:
        """有效期只追加：旧的当前行写入 replaced_at，再插入新行。"""
        self.conn.execute(
            "UPDATE version_validity SET replaced_at = %s "
            "WHERE version_id = %s AND recorded_at IS NOT NULL AND replaced_at IS NULL", (now, version_id))
        self.conn.execute(
            "INSERT INTO version_validity (id, version_id, valid_from, valid_from_grade, valid_to, valid_to_grade, "
            "valid_to_reason, unresolved_text, locator, recorded_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (uuid.uuid4(), version_id, v.valid_from, v.valid_from_grade, v.valid_to, v.valid_to_grade,
             v.valid_to_reason, v.unresolved_text, _json(v.locator), now))

    def _close_predecessors(self, version_id, now: datetime) -> None:
        """按开发文档 7.3 推导被取代版本的作废日。"""
        succ = self.conn.execute(
            "SELECT document_id, approved_at FROM document_version WHERE id = %s", (version_id,)).fetchone()
        rels = self.conn.execute(
            "SELECT r.to_version_id, v.document_id, v.status FROM version_relation r "
            "JOIN document_version v ON v.id = r.to_version_id "
            "WHERE r.from_version_id = %s AND r.relation = 'supersedes' "
            "AND r.recorded_at IS NOT NULL AND r.retracted_at IS NULL", (version_id,)).fetchall()
        if not rels:
            return
        cur = self._current_validity(version_id)
        if cur.valid_from is not None and cur.valid_from_grade == "explicit":
            end, grade = cur.valid_from, "explicit"
        else:
            end, grade = succ["approved_at"].astimezone(timezone.utc).date(), "inferred"
        for r in rels:
            if r["status"] == "pending_review":
                continue
            pred = self._current_validity(r["to_version_id"])
            if pred.valid_from is not None and end <= pred.valid_from:
                # 7.3 未覆盖的情形：推导出的作废日不晚于旧版本生效日，旧版本的有效区间会变成空的。
                # 数据库约束 interval_ordered 会拒绝；这里先给出可读的错误，由审核人更正生效日或取代关系
                raise InvalidState(
                    f"新版本的{'生效日' if grade == 'explicit' else '审核通过日（推断的生效日）'} {end} "
                    f"不晚于被取代版本的生效日 {pred.valid_from}，无法推导作废日；请更正新版本的生效日或取代关系")
            reason = "new_version" if r["document_id"] == succ["document_id"] else "superseded"
            if (pred.valid_to, pred.valid_to_grade, pred.valid_to_reason) == (end, grade, reason):
                continue
            self._record(r["to_version_id"], Validity(pred.valid_from, pred.valid_from_grade, end, grade, reason,
                                                      pred.unresolved_text, pred.locator), now)
            self._enqueue(r["to_version_id"], "index_update", now)

    def _enqueue(self, version_id, stage: str, now: datetime) -> None:
        self.conn.execute(
            "INSERT INTO ingest_job (id, version_id, stage, status, created_at, updated_at) "
            "VALUES (%s, %s, %s, 'queued', %s, %s)", (uuid.uuid4(), version_id, stage, now, now))

    def _event(self, version_id, event: str, actor: str, detail: Any, now: datetime) -> None:
        self.conn.execute(
            "INSERT INTO lifecycle_event (id, version_id, event, actor, detail, occurred_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)", (uuid.uuid4(), version_id, event, actor, _json(detail), now))


def _json(v: Any):
    return None if v is None else Jsonb(v)


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d else None
