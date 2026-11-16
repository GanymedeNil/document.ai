"""入库流水线第 1–3 阶段（开发文档 6.2）。

接收在请求内同步完成；文本解析、分类与抽取作为 ingest_job 由后台进程执行：
用 SELECT … FOR UPDATE SKIP LOCKED 领取任务，失败后退避重试（Worker.backoff），进程中断后租约到期即可被重新领取。
每个阶段都可重复执行：证据 id 稳定，重复写入不会产生重复证据；待确认值整体替换。
"""
from __future__ import annotations

import hashlib
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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
                batch_id: uuid.UUID | None = None, doc_date: date | None = None,
                publisher: str | None = None) -> Received:
        """第 1 阶段：接收。doc_date、publisher 是数据源随文档给出的文档日期（发布日、报告日，不是上传或文件创建时间）
        与出处（发布方、来源名称）：写入版本，证据标注据此给出“文档日期”“出处”。数据源给出的文档日期与正文写明的同等
        对待（来源等级 explicit），优先于从正文抽取的；没有时由第 3 阶段从正文抽取。"""
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
                                      batch_id=batch_id, doc_date=doc_date, publisher=publisher)
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
            self._save_parsed(version_id, parsed)          # 留给深度解析，不再解析第二遍（迁移 0011）
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
                if ex.doc_date and v["doc_date"] is None:          # 数据源给了文档日期时以它为准
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

        表格：每个数据单元格一条证据，内容带行名和列名；有表题时前面加“表N《表题》｜”（与图表的“图N《标题》”一致，
        单元格要靠表题才检索得到、才知道单位），没有表题的照旧。多层表头（解析器给出的表头行数大于 1）时，列名是
        同一列各行表头连起来（“EU movers / 1 000s / 2021”），表头的各行都不当作数据：否则第二行表头（单位、年份）
        会变成只有表题、没有数值的单元格占满检索结果，数据单元格也看不出是哪一年、什么单位。
        只有第一格有内容的行（单列的清单、电子表格里一列写的条款）
        整格一条证据。图片：配置了有看图能力的模型（figure_reader）时交给它
        转写，图表每个数据点一条证据（figures.py）；没有配置时跳过。表格先写入；图片逐张读取、逐张提交，
        已有证据的图片不再读取，所以任务失败重试时不会重复读取、也不会产生重复证据。有图片读取失败时，
        已写入的照常进索引，任务报错交给重试。
        """
        v = self.conn.execute("SELECT * FROM document_version WHERE id = %s", (version_id,)).fetchone()
        if v is None or v["status"] == "pending_review":
            return
        filename = v["raw_object_uri"].rsplit("/", 1)[-1]
        parser = parser_for(filename, self.parsers)
        parsed = self._load_parsed(version_id, parser.name) or parser.parse(self.store.get(v["raw_object_uri"]), filename)
        tables = [b for b in parsed.blocks if b.kind == "table"]
        with self.conn.transaction():
            for t_index, t in enumerate(tables):
                if len(t.rows) < 2:
                    continue
                n_head = t.header_rows if 0 < t.header_rows < len(t.rows) else 1
                header = column_names(t.rows[:n_head])
                head = f"表{t_index + 1}《{t.text}》｜" if t.text else ""
                single = all(not any(x for x in row[1:]) for row in t.rows)
                for r, row in enumerate(t.rows):
                    if r < n_head and not single:
                        continue
                    if not any(x for x in row[1:]):
                        # 只有第一格有内容的行（电子表格里一列写的条款、清单，或单列的表）：整格就是一条证据，
                        # 不能因为没有“列”就丢掉（Docling 把每个工作表解析成一张表）
                        if row and row[0]:
                            self.lc.add_evidence(version_id, "table", _locator(t, table=t_index, row=r, column=0),
                                                 head + row[0], "deep")
                        continue
                    for c in range(1, len(row)):
                        if not row[c]:
                            continue
                        col = header[c] if c < len(header) and header[c] else f"第{c + 1}列"
                        self.lc.add_evidence(version_id, "table", _locator(t, table=t_index, row=r, column=c),
                                             f"{head}{row[0]}｜{col}：{row[c]}", "deep")
        failed = self._read_figures(version_id, parsed)
        self._enqueue(version_id, "index_update", repeatable=True)
        if failed:
            raise FigureReadError("；".join(failed))      # 暂存留着：重试时不必再解析
        self.conn.execute("DELETE FROM parse_cache WHERE version_id = %s", (version_id,))

    def _save_parsed(self, version_id: uuid.UUID, parsed) -> None:
        blocks = [{"kind": b.kind, "text": b.text, "level": b.level, "rows": b.rows, "section": list(b.section),
                   "order": b.order, "page": b.page, "image_type": b.image_type if b.image else None,
                   "header_rows": b.header_rows}
                  for b in parsed.blocks]
        self.conn.execute(
            "INSERT INTO parse_cache (version_id, parser, title, blocks, created_at) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (version_id) DO UPDATE SET parser = EXCLUDED.parser, title = EXCLUDED.title, "
            "blocks = EXCLUDED.blocks, created_at = EXCLUDED.created_at",
            (version_id, parsed.parser, parsed.title, Jsonb(blocks), self.clock()))
        self.conn.execute("DELETE FROM parse_cache_image WHERE version_id = %s", (version_id,))
        for b in parsed.blocks:
            if b.image and b.image_type:
                self.conn.execute("INSERT INTO parse_cache_image (version_id, block_order, mime, data) "
                                  "VALUES (%s, %s, %s, %s)", (version_id, b.order, b.image_type, b.image))

    def _load_parsed(self, version_id: uuid.UUID, parser_name: str):
        """文本解析时暂存的结果；没有、或解析器版本不同（升级后）时为 None，交给重新解析。"""
        from eaer.ingestion.parsers import Block, ParsedDocument

        row = self.conn.execute("SELECT parser, title, blocks FROM parse_cache WHERE version_id = %s",
                                (version_id,)).fetchone()
        if row is None or row["parser"] != parser_name:
            return None
        images = {r["block_order"]: (r["mime"], bytes(r["data"])) for r in self.conn.execute(
            "SELECT block_order, mime, data FROM parse_cache_image WHERE version_id = %s", (version_id,))}
        blocks = []
        for d in row["blocks"]:
            mime, data = images.get(d["order"], (None, None))
            blocks.append(Block(d["kind"], d["text"], d["level"], d["rows"], tuple(d["section"]), d["order"], d["page"],
                                image=data, image_type=mime, header_rows=d.get("header_rows", 1)))
        return ParsedDocument(row["title"], blocks, row["parser"])

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


def column_names(header: list[list[str]]) -> list[str]:
    """各列的列名：多层表头时把同一列各行的表头用“ / ”连起来（“EU movers / 1 000s / 2021”），跨行合并的表头
    在各行重复出现，只取一次。只有一行表头时就是那一行。"""
    names = []
    for c in range(max((len(r) for r in header), default=0)):
        parts: list[str] = []
        for row in header:
            x = row[c] if c < len(row) else ""
            if x and (not parts or parts[-1] != x):
                parts.append(x)
        names.append(" / ".join(parts))
    return names


def _locator(b, **extra) -> dict:
    """证据定位：章节路径加位置；有页码的格式（PDF、PPTX、XLSX）带上页码，便于核对（AC-6-1）。
    没有页码时不写这个键，原有 Word、HTML 证据的 id 保持不变。"""
    loc = {"section": list(b.section), **extra}
    if b.page is not None:
        loc["page"] = b.page
    return loc


BACKOFF = (timedelta(seconds=30), timedelta(minutes=2), timedelta(minutes=10))
"""任务失败后再次领取前的等待：第 1 次失败后 30 秒，第 2 次 2 分钟，第 3 次 10 分钟（迁移 0013）。模型客户端在一次尝试
里已有秒级的重试（eaer/adapters/online.py），这里应对的是更长的过载：本地 Embedding 服务在内存预算下连续几十秒返回
503 时，立即重排会在几秒内用完全部尝试次数。不加随机抖动：任务由后台线程逐个执行，到期的任务再多，同时发出的请求
也不超过线程数。"""


class Worker:
    """后台任务进程：领取本进程能处理的阶段的任务并执行。

    任务出错后按 backoff 退避再排队（not_before），领取时跳过还没到时间的任务，其他任务照常按优先级领取；
    尝试次数默认是退避次数加 1（4 次，最后一次失败约在首次失败 12.5 分钟后），用完标为 failed。
    索引任务永久失败后由对账补写（eaer/ingestion/repair.py）。
    """

    def __init__(self, ingestion: Ingestion, indexer=None, lease: timedelta = timedelta(minutes=5),
                 max_attempts: int | None = None, backoff: tuple[timedelta, ...] = BACKOFF):
        self.ing = ingestion
        self.conn = ingestion.conn
        self.lease = lease
        self.backoff = tuple(backoff)
        self.max_attempts = len(self.backoff) + 1 if max_attempts is None else max_attempts
        self.indexer = indexer
        self.handlers = {"parse_text": ingestion.parse_text, "classify_extract": ingestion.classify_extract,
                         "parse_deep": ingestion.parse_deep}
        if indexer is not None:
            self.handlers |= {"publish_index": indexer.sync_version, "index_update": indexer.sync_version}

    def delay_after(self, attempts: int) -> timedelta:
        """第 attempts 次尝试失败后的等待；退避表用完后沿用最后一档。"""
        if not self.backoff:
            return timedelta(0)
        return self.backoff[min(attempts, len(self.backoff)) - 1]

    def claim(self) -> dict | None:
        now = self.ing.clock()
        # 最后一次尝试时进程中断、租约已到期的任务：不再重试，标为失败
        self.conn.execute(
            "UPDATE ingest_job SET status = 'failed', locked_until = NULL, updated_at = %s, "
            "last_error = coalesce(last_error, '') || '租约到期且已达最大重试次数' "
            "WHERE stage = ANY(%s) AND status = 'running' AND locked_until < %s AND attempts >= %s",
            (now, list(self.handlers), now, self.max_attempts))
        # 退避中的任务（not_before 未到）不领取，也不占位：子查询直接越过它，取下一个可领取的
        return self.conn.execute(
            "UPDATE ingest_job SET status = 'running', locked_until = %s, attempts = attempts + 1, updated_at = %s, "
            "  not_before = NULL "
            "WHERE id = (SELECT id FROM ingest_job WHERE stage = ANY(%s) AND attempts < %s AND "
            "  ((status = 'queued' AND (not_before IS NULL OR not_before <= %s)) "
            "   OR (status = 'running' AND locked_until < %s)) "
            "  ORDER BY priority, created_at FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *",
            (now + self.lease, now, list(self.handlers), self.max_attempts, now, now)).fetchone()

    def run_once(self) -> bool:
        job = self.claim()
        if job is None:
            return False
        with span(f"eaer.ingest.{job['stage']}", **{"eaer.version_id": str(job["version_id"]),
                                                    "eaer.attempt": job["attempts"]}) as sp:
            try:
                self.handlers[job["stage"]](job["version_id"])
            except Exception as e:  # noqa: BLE001 — 任何错误都记入任务，交给重试
                now = self.ing.clock()
                final = job["attempts"] >= self.max_attempts
                status, not_before = ("failed", None) if final else ("queued", now + self.delay_after(job["attempts"]))
                self.conn.execute(
                    "UPDATE ingest_job SET status = %s, not_before = %s, last_error = %s, locked_until = NULL, "
                    "updated_at = %s WHERE id = %s",
                    (status, not_before, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}", now, job["id"]))
                set_attrs(sp, {"eaer.job_status": status, "error.type": type(e).__name__,
                               **({"eaer.retry_at": not_before.isoformat()} if not_before else {})})
                return True
            self.conn.execute("UPDATE ingest_job SET status = 'done', locked_until = NULL, updated_at = %s WHERE id = %s",
                              (self.ing.clock(), job["id"]))
            set_attrs(sp, {"eaer.job_status": "done"})
        return True

    def run_until_idle(self, limit: int = 100_000) -> int:
        """执行到没有可领取的任务为止；退避中的任务留在队列里，由之后的轮询领取（后台任务进程）。"""
        n = 0
        while n < limit and self.run_once():
            n += 1
        return n

    def next_retry_at(self) -> datetime | None:
        """本进程能处理的阶段里，退避中的任务最早什么时候到期；没有则为 None。"""
        return self.conn.execute(
            "SELECT min(not_before) AS t FROM ingest_job WHERE stage = ANY(%s) AND status = 'queued' "
            "AND attempts < %s AND not_before > %s",
            (list(self.handlers), self.max_attempts, self.ing.clock())).fetchone()["t"]

    def run_until_settled(self, sleep: Callable[[float], None] = time.sleep) -> int:
        """执行到本进程能处理的任务都结束（完成或失败）为止：退避中的任务等到期再做。供评测运行器与一次性的工具用——
        它们入库后紧接着就要检索，用 run_until_idle 会把正在退避的任务留下、结束时随评测库一起删掉。
        别的进程正在执行的任务不等。"""
        n = 0
        while True:
            n += self.run_until_idle()
            due = self.next_retry_at()
            if due is None:
                return n
            sleep(max((due - self.ing.clock()).total_seconds(), 0.0) + 0.05)
