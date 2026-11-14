"""检索评测（开发文档 14、AC-5-4）。

按日期把合成语料走一遍真实的入库、审核、生命周期和索引代码，每个查询在它的提问日提出，
这样归档前后、补录前后、新版本审核前后的状态都是提问当时的状态。

计分：
- 命中：每条依据事实都至少有一条证据出现在前 top_m 条中（严格，多条依据缺一条即未命中）；
- 拒答类：前 top_m 条中不得出现目标事实所在版本、含该取值的证据（只看目标事实自己的版本：别的规定碰巧有同样的数字
  不算泄露；无权或已归档的版本在数据库核对时已剔除，若出现会同时计为越权）；
- 名次：命中查询的名次取各条依据事实最早出现位置中最靠后的一个，用来计算 MRR（未命中记 0），
  召回率相同时用它比较排序质量；
- 越权：任一返回证据在提问时不可搜索或提问角色无权查看，即计一次，应为 0；
- 图表类事实需要视觉模型解析：没有配置图片读取器时单独列出，不计入召回分母；配置了（生成服务声明看图能力）则照常计分。
时间参数直接取事实表中的生效时间点与回溯时间点，相当于时间识别完全正确；时间识别的准确率另行评测（AC-7-6）。
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import psycopg

from eaer.ingestion.indexer import Indexer
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.lifecycle.service import Lifecycle
from eaer.retrieval.embed import Embedder
from eaer.retrieval.index import SearchIndex
from eaer.retrieval.reconcile import reconcile
from eaer.retrieval.service import Retriever, Scope
from eaer.review.service import Review
from eaer.storage import FileStore
from factset.model import FactSet

DOC_TYPE = {"规范类": "normative", "记录类": "record"}
ACTOR = "eval-runner"


def norm(s: str) -> str:
    s = re.sub(r"\s+", "", s.translate(str.maketrans("０１２３４５６７８９，", "0123456789,")))
    return re.sub(r"(?<=\d),(?=\d{3})", "", s)


@dataclass
class QueryResult:
    query_id: str
    type: str
    outcome: str                  # hit / miss / needs_vision / refused / refusal_leak
    found: int = 0
    needed: int = 0
    leaks: int = 0
    top: list[str] = field(default_factory=list)
    rank: int | None = None       # 命中时全部依据都已出现的名次（从 1 开始）


@dataclass
class Report:
    results: list[QueryResult]
    reconcile_diffs: int
    embedder: str
    top_m: int

    @property
    def recall(self) -> float:
        scored = [r for r in self.results if r.outcome in ("hit", "miss")]
        return sum(r.outcome == "hit" for r in scored) / len(scored) if scored else float("nan")

    @property
    def mrr(self) -> float:
        scored = [r for r in self.results if r.outcome in ("hit", "miss")]
        return sum(1 / r.rank for r in scored if r.outcome == "hit" and r.rank) / len(scored) if scored else float("nan")

    @property
    def leaks(self) -> int:
        return sum(r.leaks for r in self.results)

    def by_type(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in self.results:
            out[r.type][r.outcome] += 1
        return {k: dict(v) for k, v in out.items()}

    def text(self) -> str:
        lines = [f"检索评测：{len(self.results)} 个查询，向量化 {self.embedder}，前 {self.top_m} 条",
                 f"召回率（不含需视觉模型的查询）：{self.recall:.1%}；MRR {self.mrr:.3f}",
                 f"拒答类查询：{sum(r.outcome == 'refused' for r in self.results)} 个正确，"
                 f"{sum(r.outcome == 'refusal_leak' for r in self.results)} 个泄露取值",
                 f"越权返回：{self.leaks} 次；索引对账差异：{self.reconcile_diffs} 个版本"]
        for t, c in sorted(self.by_type().items()):
            lines.append(f"  {t}：" + "，".join(f"{k} {v}" for k, v in sorted(c.items())))
        for r in self.results:
            if r.outcome in ("miss", "refusal_leak", "needs_vision") or r.leaks:
                lines.append(f"  · {r.query_id}（{r.type}）{r.outcome}，依据 {r.found}/{r.needed}，越权 {r.leaks}")
        return "\n".join(lines)


class _Clock:
    def __init__(self):
        self.now = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def set(self, d: date) -> None:
        self.now = max(self.now, datetime.combine(d, time.min, tzinfo=timezone.utc))

    def __call__(self) -> datetime:
        self.now += timedelta(microseconds=1)
        return self.now


def run(conn: psycopg.Connection, fs: FactSet, corpus: Path, manifest: dict, *, store: FileStore,
        index: SearchIndex, embedder: Embedder, tenant_id: str = "eval", top_m: int = 10,
        build_id: str = "eval", ask=None, reranker=None, figure_reader=None) -> Report:
    """ask：在提问日回答每个查询的函数，签名同 _ask；默认按检索结果计分，Fast 试跑传入自己的回答函数。
    figure_reader：深度解析时读取图片的读取器（eaer/ingestion/figures.py）；给了它，图表类事实照常计分。"""
    if os.environ.get("EAER_ENV") != "eval":
        raise PermissionError("评测运行器只在评测环境启用（EAER_ENV=eval）")
    clock = _Clock()
    ing = Ingestion(conn, store, clock, figure_reader=figure_reader)
    lc = Lifecycle(conn, clock)
    review = Review(conn, lc, store)
    worker = Worker(ing, Indexer(conn, index, embedder, build_id))
    retriever = Retriever(conn, index, embedder, reranker, top_m=top_m)
    entry = {m["version_id"]: m for m in manifest["versions"]}

    clock.set(min(v.uploaded_on for v in fs.versions.values()))
    lc.ensure_tenant(tenant_id)
    acl = {d.id: lc.create_acl_policy(tenant_id, sorted(d.roles)) for d in fs.documents.values()}
    ids: dict[str, object] = {}
    doc_ids: dict[str, object] = {}

    ops = [(v.uploaded_on, 0, v.row, "create", v) for v in fs.versions.values()]
    ops += [(v.approved_on, 1, v.row, "publish", v) for v in fs.versions.values()]
    ops += [(e.occurred_on, 2, e.row, "event", e) for e in fs.events]
    ops += [(q.asked_on, 3, q.row, "query", q) for q in fs.queries]
    results: list[QueryResult] = []
    for day, _, _, kind, obj in sorted(ops, key=lambda o: o[:3]):
        clock.set(day)
        if kind == "create":
            m = entry[obj.id]
            r = ing.receive(tenant_id, (corpus / m["file"]).read_bytes(), m["file"], acl_policy_id=acl[obj.document_id],
                            source_ref=obj.document_id, title=fs.documents[obj.document_id].title,
                            import_batch_id=m["import_batch_id"])
            ids[obj.id], doc_ids[obj.document_id] = r.version_id, r.document_id
        elif kind == "publish":
            task = conn.execute("SELECT id FROM review_task WHERE version_id = %s", (ids[obj.id],)).fetchone()["id"]
            res = review.decide(tenant_id, [{"task_id": str(task), "decision": "approve", "corrections": {
                "doc_type": DOC_TYPE[fs.documents[obj.document_id].doc_type]}}], ACTOR)
            if not res[0]["ok"]:
                raise RuntimeError(f"{obj.id} 审核失败：{res[0]['error']}")
        elif kind == "event":
            _apply_event(lc, fs, obj, ids, doc_ids)
        else:
            results.append(ask(conn, fs, obj, retriever, ids, tenant_id) if ask else
                           _ask(conn, fs, obj, retriever, ids, tenant_id, vision=figure_reader is not None))
        worker.run_until_idle()
    diffs = reconcile(conn, index, tenant_id)
    return Report(results, sum(len(v) for v in diffs.values()), embedder.name, top_m)


def _apply_event(lc: Lifecycle, fs: FactSet, e, ids, doc_ids) -> None:
    doc = e.target if e.target in fs.documents else fs.versions[e.target].document_id
    if e.type == "归档":
        lc.archive_document(doc_ids[doc], ACTOR)
    elif e.type == "取消归档":
        lc.unarchive_document(doc_ids[doc], ACTOR)
    elif e.type == "有效期更正":
        lc.correct_validity(ids[e.target], ACTOR, valid_from=e.new_valid_from,
                            valid_from_grade=e.new_grade or ("explicit" if e.new_valid_from else None))
    elif e.type == "权限变更":
        lc.change_acl(doc_ids[doc], sorted(e.new_roles), ACTOR)


def _ask(conn, fs: FactSet, q, retriever: Retriever, ids, tenant_id: str, *, vision: bool = False) -> QueryResult:
    hits = retriever.search(Scope(tenant_id, frozenset({q.role})), q.question, today=q.asked_on,
                            as_of_valid=q.as_of_valid, as_of_system=q.as_of_system)
    top = [h.evidence_id for h in hits]
    texts = [norm(h.content) for h in hits]
    leaks = _leaks(conn, hits, q.role)
    if q.should_refuse:
        own = {str(ids[f.version_id]): norm(f"{f.value}{f.unit or ''}") for f in fs.facts.values()
               if f.key == q.target_key and f.version_id in ids}
        leaked = any(h.version_id in own and own[h.version_id] in t for h, t in zip(hits, texts))
        return QueryResult(q.id, q.type, "refusal_leak" if leaked else "refused", leaks=leaks, top=top)
    facts = [fs.facts[i] for i in q.basis]
    if not vision and any(f.carrier == "图表" for f in facts):
        return QueryResult(q.id, q.type, "needs_vision", needed=len(facts), leaks=leaks, top=top)
    found, rank = 0, 0
    for f in facts:
        want = norm(f"{f.value}{f.unit or ''}")
        version = str(ids[f.version_id])
        pos = next((i for i, (h, t) in enumerate(zip(hits, texts), start=1)
                    if h.version_id == version and want in t), None)
        if pos is not None:
            found, rank = found + 1, max(rank, pos)
    hit = found == len(facts)
    return QueryResult(q.id, q.type, "hit" if hit else "miss", found, len(facts), leaks, top, rank if hit else None)


def _leaks(conn, hits, role: str) -> int:
    if not hits:
        return 0
    rows = conn.execute(
        "SELECT v.id, v.status, p.principals FROM document_version v JOIN acl_policy p ON p.id = v.acl_policy_id "
        "WHERE v.id = ANY(%s::uuid[])", ([h.version_id for h in hits],)).fetchall()
    state = {str(r["id"]): r for r in rows}
    return sum(1 for h in hits if h.version_id not in state or state[h.version_id]["status"] != "searchable"
               or role not in state[h.version_id]["principals"])
