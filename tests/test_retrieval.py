"""检索（开发文档第 5 节；AC-5-1、5-2、5-3 的切换逻辑、5-5、6-7；检索评测流程）。"""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from eaer.evalrun.retrieval import run as run_eval
from eaer.ingestion.pipeline import Worker
from eaer.retrieval.embed import HashingEmbedder
from eaer.retrieval.index import Filters, IndexDoc, MemoryIndex, MissingScope, rrf
from eaer.ingestion.indexer import Indexer
from eaer.retrieval.reconcile import reconcile
from eaer.retrieval.opensearch import OpenSearchIndex, mapping, text_query, vector_query
from eaer.retrieval.service import Retriever, Scope
from factset import load
from factset.render import Renderer
from fixture import sample, write
from test_ingestion import env, html  # noqa: F401  复用夹具

UTC = timezone.utc


# ---------- 查询语句求值器：证明 to_dsl() 与 matches() 等价 ----------

def _get(src, field):
    return src.get(field)


def evaluate(q: dict, src: dict) -> bool:
    (kind, body), = q.items()
    if kind == "term":
        (f, v), = body.items()
        val = _get(src, f)
        return v in val if isinstance(val, list) else val == v
    if kind == "terms":
        (f, vs), = body.items()
        val = _get(src, f)
        vals = val if isinstance(val, list) else [val]
        return any(x in vals for x in vs)
    if kind == "exists":
        val = _get(src, body["field"])
        return val is not None and val != []
    if kind == "range":
        (f, cond), = body.items()
        val = _get(src, f)
        if val is None:
            return False
        ops = {"lte": lambda a, b: a <= b, "lt": lambda a, b: a < b, "gte": lambda a, b: a >= b, "gt": lambda a, b: a > b}
        return all(ops[op](val, ref) for op, ref in cond.items())
    if kind == "bool":
        ok = all(evaluate(x, src) for x in body.get("filter", []) + body.get("must", []))
        ok = ok and not any(evaluate(x, src) for x in body.get("must_not", []))
        if "should" in body:
            n = sum(evaluate(x, src) for x in body["should"])
            ok = ok and n >= body.get("minimum_should_match", 1)
        return ok
    raise AssertionError(f"求值器不支持 {kind}")


def random_doc(rng: random.Random) -> IndexDoc:
    d = lambda: rng.choice([None, date(2025, 1, 1) + timedelta(days=rng.randrange(0, 730))])  # noqa: E731
    vf, vt = d(), d()
    if vf and vt and vt <= vf:
        vt = None
    return IndexDoc(
        evidence_id=f"ev_{rng.getrandbits(64):016x}", version_id="v", document_id="d",
        tenant_id=rng.choice(["t1", "t2"]), principals=tuple(sorted(rng.sample(["A", "B", "C"], rng.randint(1, 2)))),
        doc_type=rng.choice(["normative", "record", None]), modality="text", text="x", embedding=(0.0,),
        approved_at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=rng.randrange(0, 730 * 24)),
        valid_from=vf, valid_to=vt, valid_grade=None, doc_date=d(), index_build_id="b")


def random_filters(rng: random.Random) -> Filters:
    on = date(2025, 1, 1) + timedelta(days=rng.randrange(0, 730))
    t = rng.choice([None, datetime(2025, 1, 1, 23, 59, 59, tzinfo=UTC) + timedelta(days=rng.randrange(0, 730))])
    return Filters(rng.choice(["t1", "t2"]), frozenset(rng.sample(["A", "B", "C"], rng.randint(1, 2))), on, t)


def test_dsl_and_python_filters_agree():
    rng = random.Random(20261111)
    checked = agree_true = 0
    for _ in range(300):
        f = random_filters(rng)
        dsl = f.to_dsl()
        for _ in range(40):
            doc = random_doc(rng)
            got, want = evaluate(dsl, doc.to_source()), f.matches(doc)
            assert got == want, (f, doc)
            checked += 1
            agree_true += want
    assert checked == 12000 and 100 < agree_true < 11900   # 两种结果都有足够多的样本


# ---------- AC-5-1：授权范围必填，且每个检索请求都带过滤 ----------

class FakeIndices:
    def __init__(self, aliases=None):
        self.calls, self.aliases = [], aliases or {}

    def create(self, **kw):
        self.calls.append(("create", kw))

    def exists_alias(self, name):
        return bool(self.aliases)

    def get_alias(self, name):
        return self.aliases

    def update_aliases(self, body):
        self.calls.append(("update_aliases", body))


class FakeClient:
    def __init__(self, aliases=None, bulk_errors=False):
        self.indices = FakeIndices(aliases)
        self.searches, self.bulks, self.deletes = [], [], []
        self.bulk_errors = bulk_errors

    def search(self, index, body):
        self.searches.append(body)
        return {"hits": {"hits": []}, "aggregations": {"v": {"buckets": []}}}

    def bulk(self, body, refresh):
        self.bulks.append(body)
        if self.bulk_errors:
            return {"errors": True, "items": [{"index": {"error": {"type": "mapper_parsing_exception"}}}]}
        return {"errors": False, "items": []}

    def delete_by_query(self, **kw):
        self.deletes.append(kw)


def walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def test_scope_is_mandatory():
    with pytest.raises(MissingScope):
        Filters("t1", frozenset(), date(2026, 1, 1))
    with pytest.raises(MissingScope):
        Filters("", frozenset({"A"}), date(2026, 1, 1))
    r = Retriever(conn=None, index=MemoryIndex(), embedder=HashingEmbedder())
    with pytest.raises(MissingScope):
        r.search(None, "住宿上限", today=date(2026, 1, 1))
    with pytest.raises(MissingScope):
        r.search(Scope("t1", frozenset()), "住宿上限", today=date(2026, 1, 1))


@pytest.mark.parametrize("system_t", [None, datetime(2026, 7, 2, 23, 59, tzinfo=UTC)])
def test_every_opensearch_search_carries_scope_filters(system_t):
    client = FakeClient()
    ix = OpenSearchIndex(client)
    f = Filters("t1", frozenset({"R_STAFF"}), date(2026, 7, 2), system_t)
    ix.search_text("住宿上限", f, 50)
    ix.search_vector([0.1] * 4, f, 50)
    assert len(client.searches) == 2
    for body in client.searches:
        nodes = list(walk(body))
        assert {"term": {"tenant_id": "t1"}} in nodes
        assert {"terms": {"principals": ["R_STAFF"]}} in nodes
    assert client.searches[1]["query"]["knn"]["embedding"]["filter"] == f.to_dsl()


def test_mapping_has_the_fields_of_section_5_1():
    props = mapping(256)["mappings"]["properties"]
    assert set(props) == {"evidence_id", "version_id", "document_id", "tenant_id", "principals", "doc_type",
                          "modality", "text", "embedding", "approved_at", "valid_from", "valid_to",
                          "valid_grade", "doc_date", "index_build_id"}
    assert props["embedding"]["dimension"] == 256 and props["text"]["analyzer"] == "smartcn"


def test_opensearch_writes_and_alias_switch():
    client = FakeClient(aliases={"eaer-b1": {}})
    ix = OpenSearchIndex(client)
    doc = random_doc(random.Random(1))
    ix.upsert([doc])
    assert client.bulks[0][0] == {"index": {"_index": "eaer", "_id": doc.evidence_id}}
    ix.update_fields([doc.evidence_id], {"principals": ("A",), "valid_to": date(2026, 1, 1)})
    assert client.bulks[1][1] == {"doc": {"principals": ["A"], "valid_to": "2026-01-01"}}
    ix.delete_version("v1")
    assert client.deletes[0]["body"] == {"query": {"term": {"version_id": "v1"}}}
    ix.switch_alias("eaer-b2")      # 原子切换：同一请求里移除旧索引、加上新索引（AC-5-3）
    assert client.indices.calls[-1] == ("update_aliases", {"actions": [
        {"remove": {"index": "eaer-b1", "alias": "eaer"}}, {"add": {"index": "eaer-b2", "alias": "eaer"}}]})
    with pytest.raises(RuntimeError, match="批量写入"):
        OpenSearchIndex(FakeClient(bulk_errors=True)).upsert([doc])


def test_rrf_uses_ranks_only():
    fused = rrf([("a", 100.0), ("b", 50.0)], [("b", 0.9), ("c", 0.8)], k=60)
    assert [i for i, _ in fused] == ["b", "a", "c"]


# ---------- 索引同步、对账、检索后核对 ----------

class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.texts = 0

    def embed(self, texts):
        self.texts += len(texts)
        return super().embed(texts)


@pytest.fixture
def ix(env):
    index, emb = MemoryIndex(), CountingEmbedder()
    env.worker = Worker(env.ing, Indexer(env.conn, index, emb, "b1"))
    env.index, env.emb = index, emb
    env.retriever = Retriever(env.conn, index, HashingEmbedder())   # 查询向量化不计入索引侧的调用次数
    return env


def publish(env, data: bytes, name: str, **kw):
    r = env.ing.receive("t1", data, name, acl_policy_id=env.acl, **kw)
    env.worker.run_until_idle()
    task = str(env.conn.execute("SELECT id FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()["id"])
    res = env.review.decide("t1", [{"task_id": task, "decision": "approve"}], "alice")
    assert res[0]["ok"], res
    return r


TABLE_DOC = ("<h1>差旅报销管理办法</h1><p>第一条　为规范差旅报销，制定本办法。</p>"
             "<table><tr><th>项目</th><th>标准</th></tr><tr><td>一线城市住宿上限</td><td>650元/晚</td></tr></table>"
             "<p>第三条　本办法自2026年1月1日起施行。</p>")
SCOPE = Scope("t1", frozenset({"R_STAFF"}))


def test_text_searchable_before_deep_parse_then_tables(ix):
    r = ix.ing.receive("t1", TABLE_DOC.encode(), "p.html", acl_policy_id=ix.acl)
    ix.worker.run_until_idle()
    task = str(ix.conn.execute("SELECT id FROM review_task").fetchone()["id"])
    ix.review.decide("t1", [{"task_id": task, "decision": "approve"}], "alice")
    # 只完成发布索引，深度解析尚未执行（AC-6-7）
    ix.conn.execute("UPDATE ingest_job SET status = 'done' WHERE stage = 'parse_deep'")
    ix.worker.run_until_idle()
    hits = ix.retriever.search(SCOPE, "差旅报销", today=date(2026, 3, 1))
    assert hits and all(h.modality == "text" for h in hits)
    # 深度解析完成后，表格单元格可以检索到
    ix.conn.execute("UPDATE ingest_job SET status = 'queued' WHERE stage = 'parse_deep'")
    ix.worker.run_until_idle()
    hits = ix.retriever.search(SCOPE, "一线城市住宿上限", today=date(2026, 3, 1))
    assert hits[0].modality == "table" and "650元/晚" in hits[0].content
    assert hits[0].locator["row"] == 1 and hits[0].valid_grade == "explicit"
    assert reconcile(ix.conn, ix.index, "t1") == {"missing_in_index": set(), "unexpected_in_index": set()}


def test_archive_acl_and_correction_flow_through_index(ix):
    r = publish(ix, TABLE_DOC.encode(), "p.html")
    ix.worker.run_until_idle()
    embedded = ix.emb.texts
    q = dict(today=date(2026, 3, 1))
    assert ix.retriever.search(SCOPE, "一线城市住宿上限", **q)

    ix.clock.now += timedelta(days=1)
    ix.lc.archive_document(r.document_id, "alice")
    ix.worker.run_until_idle()
    assert ix.retriever.search(SCOPE, "一线城市住宿上限", **q) == []
    assert ix.index.version_ids("t1") == set()
    ix.lc.unarchive_document(r.document_id, "alice")
    ix.worker.run_until_idle()
    assert ix.retriever.search(SCOPE, "一线城市住宿上限", **q)

    ix.lc.change_acl(r.document_id, ["R_FIN"], "alice")
    ix.worker.run_until_idle()
    assert ix.retriever.search(SCOPE, "一线城市住宿上限", **q) == []
    assert ix.retriever.search(Scope("t1", frozenset({"R_FIN"})), "一线城市住宿上限", **q)
    assert ix.emb.texts == embedded            # 取消归档与权限变更都没有重新调用向量模型

    ix.lc.correct_validity(r.version_id, "carol", valid_from=date(2026, 6, 1))
    ix.worker.run_until_idle()
    fin = Scope("t1", frozenset({"R_FIN"}))
    assert ix.retriever.search(fin, "一线城市住宿上限", **q) == []          # 3 月 1 日尚未生效
    assert ix.retriever.search(fin, "一线城市住宿上限", today=date(2026, 7, 1))


def test_stale_index_is_caught_by_database_recheck(ix):
    r = publish(ix, TABLE_DOC.encode(), "p.html")
    ix.worker.run_until_idle()
    ix.lc.archive_document(r.document_id, "alice")      # 索引任务还没跑：索引里仍有该版本
    assert ix.index.version_ids("t1") == {str(r.version_id)}
    assert ix.retriever.search(SCOPE, "一线城市住宿上限", today=date(2026, 3, 1)) == []   # AC-5-5
    assert reconcile(ix.conn, ix.index, "t1")["unexpected_in_index"] == {str(r.version_id)}
    ix.worker.run_until_idle()
    assert reconcile(ix.conn, ix.index, "t1") == {"missing_in_index": set(), "unexpected_in_index": set()}


def test_time_filters_current_historical_and_backtrack(ix):
    v1 = TABLE_DOC.replace("650元/晚", "500元/晚").replace("2026年1月1日", "2025年1月1日")
    ix.clock.set(date(2024, 12, 20))
    publish(ix, v1.encode(), "v1.html", source_ref="P001")
    ix.clock.set(date(2025, 12, 18))
    publish(ix, TABLE_DOC.encode(), "v2.html", source_ref="P001")
    ix.worker.run_until_idle()

    def top(**kw):
        hits = ix.retriever.search(SCOPE, "一线城市住宿上限", **kw)
        return [h.content.split("：")[-1] for h in hits if h.modality == "table"]

    assert top(today=date(2026, 3, 1)) == ["650元/晚"]
    assert top(today=date(2026, 3, 1), as_of_valid=date(2025, 6, 1)) == ["500元/晚"]
    # 回溯到 v2 审核之前：系统只知道 v1，且 v1 当时尚未作废
    assert top(today=date(2026, 3, 1), as_of_valid=date(2026, 2, 1), as_of_system=date(2025, 12, 17)) == ["500元/晚"]
    assert top(today=date(2026, 3, 1), as_of_valid=date(2026, 2, 1), as_of_system=date(2025, 12, 19)) == ["650元/晚"]


# ---------- 检索评测流程 ----------

@pytest.mark.parametrize("fmt", ["docx", "html"])
def test_retrieval_evaluation_on_sample(conn, tmp_path, monkeypatch, fmt):
    from eaer.storage import LocalFileStore
    monkeypatch.setenv("EAER_ENV", "eval")
    fs = load(write(sample(), tmp_path / "s.xlsx"))
    manifest = Renderer(fs).render(tmp_path / "corpus", fmt)
    rep = run_eval(conn, fs, tmp_path / "corpus", manifest, store=LocalFileStore(tmp_path / "files"),
                   index=MemoryIndex(), embedder=HashingEmbedder())
    outcome = {r.query_id: r.outcome for r in rep.results}
    assert rep.leaks == 0 and rep.reconcile_diffs == 0
    assert sorted(q for q, o in outcome.items() if o == "refused") == ["Q09", "Q12", "Q14"]
    assert outcome["Q15"] == "needs_vision" and outcome["Q05"] == "miss"     # 版本差异由 Deep 处理
    assert abs(rep.recall - 11 / 12) < 1e-9


def test_evaluation_requires_eval_environment(conn, tmp_path, monkeypatch):
    monkeypatch.delenv("EAER_ENV", raising=False)
    with pytest.raises(PermissionError):
        run_eval(conn, None, tmp_path, {"versions": []}, store=None, index=MemoryIndex(), embedder=HashingEmbedder())
