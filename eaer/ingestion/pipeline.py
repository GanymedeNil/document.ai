"""入库流水线第 1–3 阶段（开发文档 6.2）。

接收在请求内同步完成；文本解析、分类与抽取作为 ingest_job 由后台进程执行：
用 SELECT … FOR UPDATE SKIP LOCKED 领取任务，失败重试，进程中断后租约到期即可被重新领取。
每个阶段都可重复执行：证据 id 稳定，重复写入不会产生重复证据；待确认值整体替换。
"""
from __future__ import annotations

import hashlib
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import psycopg
from psycopg.types.json import Jsonb

from eaer.domain.injection import suspicious
from eaer.domain.jobs import DEFAULT_PRIORITY, PRIORITY
from eaer.ingestion.extract import classify, extract
from eaer.ingestion.parsers import DEFAULT_PARSERS, Parser, parser_for
from eaer.lifecycle.service import Lifecycle, utcnow
from eaer.observability import set_attrs, span
from eaer.storage import FileStore

LEGACY_GRADE = "unknown"   # 批量导入的上传时间不作为时间依据（6.2）


@dataclass(frozen=True)
class Received:
    document_id: uuid.UUID
    version_id: uuid.UUID
    duplicate: bool


class FigureReadError(Exception):
    """有图片没能读取；已写入的证据保留，任务交给重试。"""


class Ingestion:
    def __init__(self, conn: psycopg.Connection, store: FileStore,
                 clock: Callable[[], datetime] = utcnow, parsers: list[Parser] | None = None, figure_reader=None):
        self.conn = conn
        self.store = store
        self.clock = clock
        self.parsers = parsers or DEFAULT_PARSERS
        self.figure_reader = figure_reader      # eaer.ingestion.figures.FigureReader；None 时不读取图片
        self.lc = Lifecycle(conn, clock)

    # ---------- 第 1 阶段：接收 ----------

    def receive(self, tenant_id: str, data: bytes, filename: str, *, acl_policy_id: uuid.UUID,
                source_ref: str | None = None, title: str | None = None,
                source_created_at: datetime | None = None, import_batch_id: str | None = None,
                batch_id: uuid.UUID | None = None) -> Received:
        parser_for(filename, self.parsers)          # 不支持的格式在接收时就拒绝
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        with self.conn.transaction():
            dup = self.conn.execute(
                "SELECT v.id, v.document_id FROM document_version v JOIN document d ON d.id = v.document_id "
                "WHERE d.tenant_id = %s AND v.content_hash = %s ORDER BY v.uploaded_at LIMIT 1",
                (tenant_id, digest)).fetchone()
            if dup:
                return Received(dup["document_id"], dup["id"], True)
            ref = source_ref or Path(filename).stem
            doc = self.conn.execute("SELECT id FROM document WHERE tenant_id = %s AND source_ref = %s FOR UPDATE",
                                    (tenant_id, ref)).fetchone()
            # 没给标题时先以 source_ref 占位，文本解析后换成文档中的实际标题
            doc_id = doc["id"] if doc else self.lc.create_document(tenant_id, ref, title or ref)
            uri = self.store.put(tenant_id, data, filename)
            vid = self.lc.add_version(doc_id, content_hash=digest, raw_object_uri=uri,
                                      acl_policy_id=acl_policy_id, parser_version="pending",
                                      source_created_at=source_created_at, import_batch_id=import_batch_id,
                                      batch_id=batch_id)
            self._enqueue(vid, "parse_text")
        return Received(doc_id, vid, False)

    # ---------- 第 2 阶段：文本解析 ----------

    def parse_text(self, version_id: uuid.UUID) -> None:
        v = self.conn.execute("SELECT * FROM document_version WHERE id = %s", (version_id,)).fetchone()
        if v is None or v["status"] != "pending_review":
            return  # 已删除或已发布：没有要做的
        filename = v["raw_object_uri"].rsplit("/", 1)[-1]
        parser = parser_for(filename, self.parsers)
        parsed = parser.parse(self.store.get(v["raw_object_uri"]), filename)
        with self.conn.transaction():
            for b in parsed.paragraphs():
                self.lc.add_evidence(version_id, "text", _locator(b, para=b.order), b.text)
            # parser_version 只在待审核时可改；记录实际使用的解析器
            self.conn.execute("UPDATE document_version SET parser_version = %s WHERE id = %s",
                              (parser.name, version_id))
            if parsed.title:
                self.conn.execute(
                    "UPDATE document SET title = %s WHERE id = %s AND title = source_ref", (parsed.title, v["document_id"]))
            self._enqueue(version_id, "classify_extract")

    # ---------- 第 3 阶段：分类与抽取 ----------

    def classify_extract(self, version_id: uuid.UUID) -> None:
        """第 3 阶段：分类与抽取，结果写入审核建议。另把疑似给模型的指令文字（10.3）列在建议的 injection 中，
        提醒审核人：这类文字不会被执行，但审核人可以据此退回或清理文档。"""
        v = self.conn.execute(
            "SELECT v.*, d.title, d.doc_type, d.tenant_id FROM document_version v JOIN document d ON d.id = v.document_id "
            "WHERE v.id = %s", (version_id,)).fetchone()
        if v is None or v["status"] != "pending_review":
            return
        paras = [(r["locator"], r["content"]) for r in self.conn.execute(
            "SELECT locator, content FROM evidence WHERE version_id = %s AND modality = 'text' "
            "ORDER BY (locator->>'para')::int", (version_id,))]
        doc_type, rule = (v["doc_type"], "沿用文档已确认的类型") if v["doc_type"] else classify(v["title"])
        ex = extract(paras)
        relations, unmatched = self._match_relations(v, ex)

        with self.conn.transaction():
            if doc_type and not v["doc_type"]:
                self.conn.execute("UPDATE document SET doc_type = %s WHERE id = %s AND doc_type IS NULL",
                                  (doc_type, v["document_id"]))
            if doc_type == "record":
                self.lc.set_draft_validity(version_id, valid_from_grade="unknown")
                if ex.doc_date:
                    self.conn.execute("UPDATE document_version SET doc_date = %s WHERE id = %s",
                                      (ex.doc_date.value, version_id))
            else:
                self.lc.set_draft_validity(version_id, **self._validity(v, ex))
            self.lc.clear_draft_relations(version_id)
            for to_version, grade, locator in relations:
                self.lc.add_draft_relation("supersedes", version_id, to_version, grade, locator)
            self.conn.execute(
                "UPDATE review_task SET suggestions = %s WHERE version_id = %s AND status = 'open'",
                (Jsonb({"doc_type": doc_type, "doc_type_rule": rule, "extraction": ex.to_json(),
                        "unmatched_relations": unmatched, "injection": self._injection(v["title"], paras)}), version_id))

    @staticmethod
    def _injection(title: str, paras: list[tuple[dict, str]]) -> list[dict]:
        out = [{"field": "title", "kinds": k, "text": title[:80]} for k in [suspicious(title)] if k]
        return out + [{"field": "paragraph", "locator": loc, "kinds": k, "text": c[:80]}
                      for loc, c in paras for k in [suspicious(c)] if k]

    # ---------- 第 6 阶段：深度解析（表格与图片） ----------

    def parse_deep(self, version_id: uuid.UUID) -> None:
        """发布后在后台解析表格与图片（6.2 第 6 阶段，不阻塞发布）。

        表格：每个数据单元格一条证据，内容带行名和列名；只有第一格有内容的行（单列的清单、电子表格里一列写的条款）
        整格一条证据。图片：配置了有看图能力的模型（figure_reader）时交给它
        转写，图表每个数据点一条证据（figures.py）；没有配置时跳过。表格先写入；图片逐张读取、逐张提交，
        已有证据的图片不再读取，所以任务失败重试时不会重复读取、也不会产生重复证据。有图片读取失败时，
        已写入的照常进索引，任务报错交给重试。
        """
        v = self.conn.execute("SELECT * FROM document_version WHERE id = %s", (version_id,)).fetchone()
        if v is None or v["status"] == "pending_review":
            return
        filename = v["raw_object_uri"].rsplit("/", 1)[-1]
        parsed = parser_for(filename, self.parsers).parse(self.store.get(v["raw_object_uri"]), filename)
        tables = [b for b in parsed.blocks if b.kind == "table"]
        with self.conn.transaction():
            for t_index, t in enumerate(tables):
                if len(t.rows) < 2:
                    continue
                header = t.rows[0]
                single = all(not any(x for x in row[1:]) for row in t.rows)
                for r, row in enumerate(t.rows):
                    if r == 0 and not single:
                        continue
                    if not any(x for x in row[1:]):
                        # 只有第一格有内容的行（电子表格里一列写的条款、清单，或单列的表）：整格就是一条证据，
                        # 不能因为没有“列”就丢掉（Docling 把每个工作表解析成一张表）
                        if row and row[0]:
                            self.lc.add_evidence(version_id, "table", _locator(t, table=t_index, row=r, column=0),
                                                 row[0], "deep")
                        continue
                    for c in range(1, len(row)):
                        if not row[c]:
                            continue
                        col = header[c] if c < len(header) and header[c] else f"第{c + 1}列"
                        self.lc.add_evidence(version_id, "table", _locator(t, table=t_index, row=r, column=c),
                                             f"{row[0]}｜{col}：{row[c]}", "deep")
        failed = self._read_figures(version_id, parsed)
        self._enqueue(version_id, "index_update", repeatable=True)
        if failed:
            raise FigureReadError("；".join(failed))

    def _read_figures(self, version_id: uuid.UUID, parsed) -> list[str]:
        if self.figure_reader is None:
            return []
        from eaer.ingestion.figures import evidence_texts

        figures = [b for b in parsed.blocks if b.kind == "figure"]
        if not any(b.image for b in figures):
            return []
        done = {r["n"] for r in self.conn.execute(
            "SELECT DISTINCT (locator->>'figure')::int AS n FROM evidence WHERE version_id = %s AND modality = 'figure'",
            (version_id,))}
        title = self.conn.execute("SELECT d.title FROM document_version v JOIN document d ON d.id = v.document_id "
                                  "WHERE v.id = %s", (version_id,)).fetchone()["title"]
        failed = []
        for i, b in enumerate(figures):
            if b.image is None or i in done:
                continue
            try:
                r = self.figure_reader.read(b.image, b.image_type, context=" / ".join([title, *b.section]))
            except Exception as e:  # noqa: BLE001 — 一张图读不出来不影响其他图与表格
                failed.append(f"图{i + 1}：{type(e).__name__}: {e}")
                continue
            with self.conn.transaction():
                for extra, text in evidence_texts(r, i + 1, b.text):
                    self.lc.add_evidence(version_id, "figure",
                                         _locator(b, figure=i, **extra, source="vision", reader=r.model), text, "deep")
        return failed

    def _validity(self, v: dict, ex) -> dict:
        out: dict = {}
        if ex.effective:
            out.update(valid_from=ex.effective.value, valid_from_grade="explicit",
                       locator={"para": ex.effective.locator.get("para"), "sentence": ex.effective.sentence})
        elif ex.relative:
            out.update(valid_from_grade="unknown", unresolved_text=ex.relative.sentence)
        else:
            out.update(valid_from_grade=LEGACY_GRADE if v["import_batch_id"] else "inferred")
        if ex.expires:
            out.update(valid_to=ex.expires.value, valid_to_grade="explicit", valid_to_reason="stated")
        return out

    def _match_relations(self, v: dict, ex) -> tuple[list, list]:
        """废止关系：按标题匹配已发布的文档；同一文档的新版本总是取代上一个已发布版本。"""
        relations, unmatched, seen = [], [], set()
        for a in ex.abolishes:
            title, no = a.value
            target = self.conn.execute(
                "SELECT v.id FROM document_version v JOIN document d ON d.id = v.document_id "
                "WHERE d.tenant_id = %s AND d.title = %s AND v.status <> 'pending_review' AND v.id <> %s "
                "AND (%s::int IS NULL OR v.version_no = %s::int) ORDER BY v.version_no DESC LIMIT 1",
                (v["tenant_id"], title, v["id"], no, no)).fetchone()
            if target and target["id"] not in seen:
                relations.append((target["id"], "explicit", {"para": a.locator.get("para"), "sentence": a.sentence}))
                seen.add(target["id"])
            elif not target:
                unmatched.append({"title": title, "version_no": no, "sentence": a.sentence})
        prev = self.conn.execute(
            "SELECT id FROM document_version WHERE document_id = %s AND version_no < %s "
            "AND status <> 'pending_review' ORDER BY version_no DESC LIMIT 1",
            (v["document_id"], v["version_no"])).fetchone()
        if prev and prev["id"] not in seen:
            relations.append((prev["id"], "inferred", {"rule": "同一文档的新版本"}))
        return relations, unmatched

    def _enqueue(self, version_id: uuid.UUID, stage: str, repeatable: bool = False) -> None:
        """流水线阶段每个版本只做一次；索引更新可以反复做，只避免重复排队。"""
        states = ("queued",) if repeatable else ("queued", "running", "done")
        exists = self.conn.execute(
            "SELECT 1 FROM ingest_job WHERE version_id = %s AND stage = %s AND status = ANY(%s)",
            (version_id, stage, list(states))).fetchone()
        if not exists:
            now = self.clock()
            self.conn.execute(
                "INSERT INTO ingest_job (id, version_id, stage, status, priority, created_at, updated_at) "
                "VALUES (%s, %s, %s, 'queued', %s, %s, %s)",
                (uuid.uuid4(), version_id, stage, PRIORITY.get(stage, DEFAULT_PRIORITY), now, now))


def _locator(b, **extra) -> dict:
    """证据定位：章节路径加位置；有页码的格式（PDF、PPTX、XLSX）带上页码，便于核对（AC-6-1）。
    没有页码时不写这个键，原有 Word、HTML 证据的 id 保持不变。"""
    loc = {"section": list(b.section), **extra}
    if b.page is not None:
        loc["page"] = b.page
    return loc


class Worker:
    """后台任务进程：领取本进程能处理的阶段的任务并执行。"""

    def __init__(self, ingestion: Ingestion, indexer=None, lease: timedelta = timedelta(minutes=5),
                 max_attempts: int = 3):
        self.ing = ingestion
        self.conn = ingestion.conn
        self.lease = lease
        self.max_attempts = max_attempts
        self.handlers = {"parse_text": ingestion.parse_text, "classify_extract": ingestion.classify_extract,
                         "parse_deep": ingestion.parse_deep}
        if indexer is not None:
            self.handlers |= {"publish_index": indexer.sync_version, "index_update": indexer.sync_version}

    def claim(self) -> dict | None:
        now = self.ing.clock()
        # 最后一次尝试时进程中断、租约已到期的任务：不再重试，标为失败
        self.conn.execute(
            "UPDATE ingest_job SET status = 'failed', locked_until = NULL, updated_at = %s, "
            "last_error = coalesce(last_error, '') || '租约到期且已达最大重试次数' "
            "WHERE stage = ANY(%s) AND status = 'running' AND locked_until < %s AND attempts >= %s",
            (now, list(self.handlers), now, self.max_attempts))
        return self.conn.execute(
            "UPDATE ingest_job SET status = 'running', locked_until = %s, attempts = attempts + 1, updated_at = %s "
            "WHERE id = (SELECT id FROM ingest_job WHERE stage = ANY(%s) AND attempts < %s AND "
            "  (status = 'queued' OR (status = 'running' AND locked_until < %s)) "
            "  ORDER BY priority, created_at FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *",
            (now + self.lease, now, list(self.handlers), self.max_attempts, now)).fetchone()

    def run_once(self) -> bool:
        job = self.claim()
        if job is None:
            return False
        with span(f"eaer.ingest.{job['stage']}", **{"eaer.version_id": str(job["version_id"]),
                                                    "eaer.attempt": job["attempts"]}) as sp:
            try:
                self.handlers[job["stage"]](job["version_id"])
            except Exception as e:  # noqa: BLE001 — 任何错误都记入任务，交给重试
                status = "failed" if job["attempts"] >= self.max_attempts else "queued"
                self.conn.execute(
                    "UPDATE ingest_job SET status = %s, last_error = %s, locked_until = NULL, updated_at = %s WHERE id = %s",
                    (status, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}", self.ing.clock(), job["id"]))
                set_attrs(sp, {"eaer.job_status": status, "error.type": type(e).__name__})
                return True
            self.conn.execute("UPDATE ingest_job SET status = 'done', locked_until = NULL, updated_at = %s WHERE id = %s",
                              (self.ing.clock(), job["id"]))
            set_attrs(sp, {"eaer.job_status": "done"})
        return True

    def run_until_idle(self, limit: int = 100_000) -> int:
        n = 0
        while n < limit and self.run_once():
            n += 1
        return n
