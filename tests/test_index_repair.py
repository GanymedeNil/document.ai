"""任务失败后的退避与对账补写（开发文档 5.4、6.2；AC-5-2 对账、AC-6-2 续跑）。

2026-11-15 MultiHop-RAG 重跑时，本地 Embedding 服务在内存预算下短时间连续返回 503：任务出错后立即重排，几秒内用完
尝试次数，一篇文章的 publish_index 与 index_update 都标为失败——版本可搜索，索引里却没有它，也没有什么会补写。这里覆盖：
- 模型调用先失败几次再成功：任务按 30 秒、2 分钟退避后完成，索引里有该版本；评测用的 run_until_settled 会等到期；
- 退避期间其他任务照常领取，到期后仍按阶段的优先级；
- 用完尝试次数的版本经对账补写进入索引，补过的不重复补；已归档却删不掉的版本经对账补删；数据库里没有的版本直接删；
- 深度解析追加的表格证据没写进索引（之后的 index_update 用完尝试次数）：版本在索引里、证据不全，对账按证据条数查出、
  补写后表格检索得到（内存实现与真实 OpenSearch 各一次）；深度解析正在执行时不补，它中途出错没排上索引任务的下一轮补；
- 对账补写同一时刻只有一个进程在做；定期对账出错只记日志；真实 OpenSearch 上连同手动命令走一遍。
Embedding 经真实的模型客户端（OnlineModelClient，自带每次尝试 2 次重试），只把传输层换成会过载的假服务。
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import psycopg
import pytest

from dbutil import Clock
from eaer.adapters.online import Endpoint, OnlineModelClient
from eaer.ingestion.indexer import Indexer
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.ingestion.repair import repair_all
from eaer.ingestion.stats import queue_stats, repair_stats
from eaer.retrieval.embed import HashingEmbedder
from eaer.retrieval.index import IndexDoc, MemoryIndex
from eaer.retrieval.reconcile import reconcile
from eaer.retrieval.service import Retriever, Scope
from eaer.review.service import Review
from eaer.storage import LocalFileStore
from test_ingestion import html

DIMS = 8
OS_URL = os.environ.get("EAER_TEST_OPENSEARCH_URL")
CLEAN = {"missing_in_index": set(), "unexpected_in_index": set(), "incomplete_in_index": set()}
LOCK = "hashtext('eaer.reconcile_repair')"


class Overloaded:
    """Embedding 服务的假传输层：还有 down 次时返回 503（模型服务内存不足），之后按文字给出确定的向量。
    模型客户端每次尝试里自己重试 2 次，一次任务尝试是 3 个请求：down=6 即前两次任务尝试失败。"""

    def __init__(self):
        self.down, self.requests, self.vectors = 0, 0, HashingEmbedder(DIMS)

    def __call__(self, url, headers, body, timeout):
        self.requests += 1
        if self.down > 0:
            self.down -= 1
            return 503, {"error": {"message": "模型服务内存不足"}}
        data = [{"index": i, "embedding": v} for i, v in enumerate(self.vectors.embed(body["input"]))]
        return 200, {"data": data, "usage": {"prompt_tokens": len(body["input"])}}


def setup(conn, tmp_path, index):
    clock = Clock()
    ing = Ingestion(conn, LocalFileStore(tmp_path / "files"), clock)
    ing.lc.ensure_tenant("t1")
    service = Overloaded()
    model = OnlineModelClient(None, Endpoint("https://emb.example", "k", "emb-model", timeout=5, retries=2),
                              embedding_dims=DIMS, transport=service, sleep=lambda s: None)
    return SimpleNamespace(conn=conn, clock=clock, ing=ing, lc=ing.lc, acl=ing.lc.create_acl_policy("t1", ["R_STAFF"]),
                           service=service, index=index, review=Review(conn, ing.lc, ing.store),
                           worker=Worker(ing, Indexer(conn, index, model.embedder(), "b1")))


@pytest.fixture
def rp(conn, tmp_path):
    return setup(conn, tmp_path, MemoryIndex())


@pytest.fixture(params=["memory", pytest.param("opensearch", marks=pytest.mark.skipif(
    not OS_URL, reason="未设置 EAER_TEST_OPENSEARCH_URL"))])
def any_rp(request, conn, tmp_path):
    """内存实现与真实 OpenSearch 各一次：两者对证据条数的计数要一致。"""
    if request.param == "memory":
        yield setup(conn, tmp_path, MemoryIndex())
        return
    from eaer.ingestion.indexer import fresh_index
    from eaer.retrieval.opensearch import connect
    with fresh_index(connect(OS_URL), f"eaer-repair-{uuid.uuid4().hex[:8]}", build_id="b1", dims=DIMS) as ix:
        yield setup(conn, tmp_path, ix)


def table_doc(i: int) -> bytes:
    """正文两条之外带一张两行的表：表格单元格由深度解析在发布之后追加为证据（6.2 第 6 阶段）。"""
    return (f"<html><body><h1>差旅规定{i}</h1><p>第一条　为规范{i}号单位的差旅报销，制定本规定。</p>"
            f"<table><tr><th>项目</th><th>标准</th></tr><tr><td>{i}号城市住宿上限</td><td>{600 + i}元/晚</td></tr>"
            f"<tr><td>{i}号城市伙食补贴</td><td>{100 + i}元/天</td></tr></table>"
            "<p>第三条　本规定自2026年1月1日起施行。</p></body></html>").encode()


def approve(env, i: int, data: bytes | None = None):
    data = data or html(f"住宿规定{i}", f"第一条　{i}号城市住宿上限为每晚{600 + i}元。", "第二条　本规定自2026年1月1日起施行。")
    r = env.ing.receive("t1", data, f"r{i}.html", acl_policy_id=env.acl)
    env.worker.run_until_idle()                       # 解析、分类与抽取：不调用 Embedding
    task = str(env.conn.execute("SELECT id FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()["id"])
    assert env.review.decide("t1", [{"task_id": task, "decision": "approve"}], "alice")[0]["ok"]
    return r


def index_jobs(env, vid):
    return env.conn.execute(
        "SELECT stage, status, attempts, not_before, updated_at, origin, last_error FROM ingest_job "
        "WHERE version_id = %s AND stage IN ('publish_index', 'index_update') ORDER BY created_at", (vid,)).fetchall()


def waiting(env):
    """测试里的“等待”：回放时钟前进，不真的睡。返回 (sleep, 记下的秒数)。"""
    waits = []

    def sleep(seconds):
        waits.append(round(seconds))
        env.clock.now += timedelta(seconds=seconds)
    return sleep, waits


def stray(version_id: str, evidence_id: str) -> IndexDoc:
    """数据库里没有的版本留在索引里的证据（索引来自别的库，或版本早已不在）。"""
    return IndexDoc(evidence_id=evidence_id, version_id=version_id, document_id=str(uuid.uuid4()), tenant_id="t1",
                    principals=("R_STAFF",), doc_type="normative", modality="text", text="别的库留下的证据",
                    embedding=(1.0,) * DIMS, approved_at=datetime(2026, 1, 1, tzinfo=timezone.utc), valid_from=None,
                    valid_to=None, valid_grade=None, doc_date=None, index_build_id="b1")


# ---------- 退避 ----------

def test_overloaded_embedding_backs_off_and_the_job_completes(rp):
    """模型调用先失败几次再成功：退避 30 秒、2 分钟后再试，第 3 次尝试写入索引。"""
    del rp.worker.handlers["parse_deep"]                # 只看写入索引这一个任务
    r = approve(rp, 1)
    rp.service.down = 6                                 # 两次任务尝试都是 503
    assert rp.worker.run_until_idle() == 1
    (j,) = index_jobs(rp, r.version_id)
    assert (j["status"], j["attempts"]) == ("queued", 1) and "HTTP 503" in j["last_error"]
    assert j["not_before"] - j["updated_at"] == timedelta(seconds=30)
    assert queue_stats(rp.conn, rp.clock.now)["waiting_retry"] == {"publish_index": 1}
    assert rp.worker.run_until_idle() == 0 and rp.index.version_ids("t1") == set()      # 没到时间不领取
    rp.clock.now += timedelta(seconds=30)
    assert rp.worker.run_until_idle() == 1
    (j,) = index_jobs(rp, r.version_id)
    assert (j["status"], j["attempts"]) == ("queued", 2) and j["not_before"] - j["updated_at"] == timedelta(minutes=2)
    rp.clock.now += timedelta(minutes=2)
    assert rp.worker.run_until_idle() == 1
    (j,) = index_jobs(rp, r.version_id)
    assert (j["status"], j["attempts"], j["not_before"]) == ("done", 3, None)
    assert rp.index.version_ids("t1") == {str(r.version_id)} and reconcile(rp.conn, rp.index, "t1") == CLEAN
    assert rp.service.requests == 7                     # 3 + 3 个 503，第 3 次尝试一个请求成功
    assert queue_stats(rp.conn, rp.clock.now)["waiting_retry"] == {}


def test_run_until_settled_waits_out_the_backoff(rp):
    """评测运行器入库后紧接着检索：退避中的任务等到期再做，不会留在队列里随评测库一起删掉。"""
    del rp.worker.handlers["parse_deep"]
    r = approve(rp, 1)
    rp.service.down = 6
    sleep, waits = waiting(rp)
    assert rp.worker.run_until_settled(sleep=sleep) == 3
    assert waits == [30, 120] and rp.index.version_ids("t1") == {str(r.version_id)}


def test_jobs_in_backoff_do_not_hold_up_other_jobs(rp):
    """退避中的任务不占位：别的任务照常领取；到期后仍按阶段的优先级（写入索引 20 先于解析 50）。"""
    del rp.worker.handlers["parse_deep"]
    a = approve(rp, 1)
    rp.service.down = 3                                 # 一次任务尝试失败
    assert rp.worker.run_until_idle() == 1
    order = []
    for stage, fn in list(rp.worker.handlers.items()):
        rp.worker.handlers[stage] = lambda vid, fn=fn, stage=stage: (order.append((stage, vid)), fn(vid))[1]
    b = rp.ing.receive("t1", html("用车规定", "第一条　公务用车须提前申请。"), "b.html", acl_policy_id=rp.acl)
    assert rp.worker.run_until_idle() == 2
    assert order == [("parse_text", b.version_id), ("classify_extract", b.version_id)]
    c = rp.ing.receive("t1", html("用印规定", "第一条　用印须经审批。"), "c.html", acl_policy_id=rp.acl)
    order.clear()
    rp.clock.now += timedelta(seconds=30)
    assert rp.worker.run_until_idle() == 3
    assert order == [("publish_index", a.version_id), ("parse_text", c.version_id), ("classify_extract", c.version_id)]
    assert rp.index.version_ids("t1") == {str(a.version_id)}


# ---------- 对账补写 ----------

def test_exhausted_index_jobs_are_repaired_by_reconcile(rp):
    """2026-11-15 的情形：Embedding 服务长时间过载，publish_index 与深度解析之后的 index_update 都用完尝试次数，
    版本可搜索、索引里没有它。服务恢复后什么都不会自己补写；对账补一个 index_update，写入索引，补过的不重复补。"""
    r = approve(rp, 1)
    vid = str(r.version_id)
    rp.service.down = 10 ** 6
    sleep, _ = waiting(rp)
    rp.worker.run_until_settled(sleep=sleep)
    assert [(j["stage"], j["status"], j["attempts"]) for j in index_jobs(rp, r.version_id)] == \
        [("publish_index", "failed", 4), ("index_update", "failed", 4)]
    assert reconcile(rp.conn, rp.index, "t1") == CLEAN | {"missing_in_index": {vid}}

    rp.service.down = 0
    assert rp.worker.run_until_idle() == 0
    (res,) = repair_all(rp.conn, rp.index, clock=rp.clock)
    assert (res.tenant_id, res.missing_in_index, res.enqueued, res.pending, res.deleted) == ("t1", [vid], [vid], [], [])
    (again,) = repair_all(rp.conn, rp.index, clock=rp.clock)        # 补的任务还在排队：不重复补
    assert (again.enqueued, again.pending) == ([], [vid])
    assert rp.worker.run_until_idle() == 1
    assert rp.index.version_ids("t1") == {vid} and reconcile(rp.conn, rp.index, "t1") == CLEAN
    assert index_jobs(rp, r.version_id)[-1]["origin"] == "reconcile"
    assert repair_stats(rp.conn) == {"enqueued": 1, "queued": 0, "running": 0, "done": 1, "failed": 0}
    (after,) = repair_all(rp.conn, rp.index, clock=rp.clock)
    assert after.clean and after.enqueued == []
    hits = Retriever(rp.conn, rp.index, HashingEmbedder(DIMS)).search(
        Scope("t1", frozenset({"R_STAFF"})), "1号城市住宿上限", today=date(2026, 3, 1))
    assert hits and hits[0].version_id == vid


def test_reconcile_removes_what_should_no_longer_be_in_the_index(rp, monkeypatch):
    """已归档的版本：索引一直删不掉（OpenSearch 不可用）而用完尝试次数，对账补一个 index_update 去删；
    数据库里没有的版本（含格式不对的 id）排不了任务，直接从索引删除。"""
    r = approve(rp, 1)
    rp.worker.run_until_idle()
    vid = str(r.version_id)
    assert rp.index.version_ids("t1") == {vid}

    real = rp.index.delete_version

    def unavailable(version_id):
        raise ConnectionError("OpenSearch 不可用")
    monkeypatch.setattr(rp.index, "delete_version", unavailable)
    rp.lc.archive_document(r.document_id, "alice")
    sleep, _ = waiting(rp)
    rp.worker.run_until_settled(sleep=sleep)
    assert index_jobs(rp, r.version_id)[-1]["status"] == "failed"
    monkeypatch.setattr(rp.index, "delete_version", real)

    orphan = str(uuid.uuid4())
    rp.index.upsert([stray(orphan, "x1"), stray("not-a-uuid", "x2")])
    assert reconcile(rp.conn, rp.index, "t1") == CLEAN | {"unexpected_in_index": {vid, orphan, "not-a-uuid"}}
    (res,) = repair_all(rp.conn, rp.index, clock=rp.clock)
    assert res.enqueued == [vid] and sorted(res.deleted) == sorted([orphan, "not-a-uuid"])
    assert rp.index.version_ids("t1") == {vid}          # 数据库里有的交给索引任务，按执行时的状态对齐
    assert rp.worker.run_until_idle() == 1
    assert reconcile(rp.conn, rp.index, "t1") == CLEAN and rp.index.version_ids("t1") == set()


def publish_text_only(env, i: int):
    """审核通过并写入正文证据；深度解析先不做（由测试决定它什么时候、怎样执行）。"""
    deep = env.worker.handlers.pop("parse_deep")
    r = approve(env, i, table_doc(i))
    assert env.worker.run_until_idle() == 1                                   # publish_index
    env.worker.handlers["parse_deep"] = deep
    return r


def evidence_count(env, version_id, *modalities: str) -> int:
    rows = env.conn.execute("SELECT modality FROM evidence WHERE version_id = %s", (version_id,)).fetchall()
    return sum(1 for r in rows if not modalities or r["modality"] in modalities)


def table_hits(env, i: int) -> list:
    hits = Retriever(env.conn, env.index, HashingEmbedder(DIMS)).search(
        Scope("t1", frozenset({"R_STAFF"})), f"{i}号城市住宿上限", today=date(2026, 3, 1))
    return [h for h in hits if h.modality == "table" and f"{600 + i}元/晚" in h.content]


def test_deep_parse_evidence_lost_by_a_failed_index_update_is_repaired(any_rp):
    """深度解析追加了表格证据并排 index_update，这个任务因 Embedding 服务长时间过载用完尝试次数：版本在索引里，
    只有正文证据，表格永远检索不到。两边的版本集合一样，只比较版本集合的对账查不出来；按每个版本的证据条数比较，
    报为证据不全，补一个 index_update，执行后表格检索得到。"""
    rp = any_rp
    r = publish_text_only(rp, 1)
    vid = str(r.version_id)
    text = evidence_count(rp, r.version_id)
    assert rp.index.evidence_counts("t1") == {vid: text} and reconcile(rp.conn, rp.index, "t1") == CLEAN

    rp.service.down = 10 ** 6
    sleep, _ = waiting(rp)
    rp.worker.run_until_settled(sleep=sleep)
    assert [(j["stage"], j["status"], j["attempts"]) for j in index_jobs(rp, r.version_id)] == \
        [("publish_index", "done", 1), ("index_update", "failed", 4)]
    assert evidence_count(rp, r.version_id, "table") == 2                   # 两行数据，各一个单元格
    assert rp.index.evidence_counts("t1") == {vid: text} and rp.index.version_ids("t1") == {vid}
    assert reconcile(rp.conn, rp.index, "t1") == CLEAN | {"incomplete_in_index": {vid}}
    assert table_hits(rp, 1) == []

    rp.service.down = 0
    (res,) = repair_all(rp.conn, rp.index, clock=rp.clock)
    assert (res.incomplete_in_index, res.missing_in_index, res.enqueued, res.pending) == ([vid], [], [vid], [])
    assert rp.worker.run_until_idle() == 1
    last = index_jobs(rp, r.version_id)[-1]
    assert (last["origin"], last["status"]) == ("reconcile", "done")
    assert rp.index.evidence_counts("t1") == {vid: evidence_count(rp, r.version_id)}
    assert reconcile(rp.conn, rp.index, "t1") == CLEAN
    assert table_hits(rp, 1)


def test_repair_leaves_a_running_deep_parse_alone(rp, monkeypatch):
    """深度解析先提交表格、再逐张读图，全部写完才排 index_update；大文档要读很久，这段时间证据条数本来就对不上。
    对账照实报告证据不全，但不补任务——它结束时自己排，补写的统计里不出现。深度解析中途出错、没排上索引任务的，
    下一轮对账补。"""
    a, b = publish_text_only(rp, 1), publish_text_only(rp, 2)
    a_id, b_id = str(a.version_id), str(b.version_id)
    seen, real = [], rp.ing._read_figures

    def reading(version_id, parsed):                # 读图期间另一个后台进程做了一次对账
        if version_id == a.version_id:
            seen.extend(repair_all(rp.conn, rp.index, clock=rp.clock))
            return real(version_id, parsed)
        raise RuntimeError("版面模型崩溃")
    monkeypatch.setattr(rp.ing, "_read_figures", reading)
    sleep, _ = waiting(rp)
    rp.worker.run_until_settled(sleep=sleep)
    (res,) = seen
    assert (res.incomplete_in_index, res.enqueued, res.pending) == ([a_id], [], [a_id])
    assert [j["origin"] for j in index_jobs(rp, a.version_id)] == [None, None]          # publish_index 与它自己排的
    assert [j["stage"] for j in index_jobs(rp, b.version_id)] == ["publish_index"]       # b 的深度解析失败、没排上
    assert rp.conn.execute("SELECT status FROM ingest_job WHERE version_id = %s AND stage = 'parse_deep'",
                           (b.version_id,)).fetchone()["status"] == "failed"
    assert reconcile(rp.conn, rp.index, "t1") == CLEAN | {"incomplete_in_index": {b_id}}
    (res,) = repair_all(rp.conn, rp.index, clock=rp.clock)
    assert res.enqueued == [b_id]
    assert rp.worker.run_until_idle() == 1
    assert reconcile(rp.conn, rp.index, "t1") == CLEAN and table_hits(rp, 1) and table_hits(rp, 2)
    assert repair_stats(rp.conn)["enqueued"] == 1


def test_only_one_process_repairs_at_a_time(rp, db_url):
    with psycopg.connect(db_url, autocommit=True) as other:
        other.execute(f"SELECT pg_advisory_lock({LOCK})")
        assert repair_all(rp.conn, rp.index) is None
        other.execute(f"SELECT pg_advisory_unlock({LOCK})")
    assert [r.clean for r in repair_all(rp.conn, rp.index)] == [True]


def test_periodic_repair_logs_index_errors_and_carries_on(rp, caplog):
    """后台任务进程定期对账时 OpenSearch 暂时不可用：只记日志，不中断任务处理；锁已释放，下一轮照常。"""
    from eaer.ingestion.__main__ import periodic_repair

    class Down(MemoryIndex):
        def evidence_counts(self, tenant_id):
            raise ConnectionError("OpenSearch 不可用")

    with caplog.at_level(logging.ERROR, logger="eaer.ingestion"):
        assert periodic_repair(rp.conn, Down()) is None
    assert "索引对账失败" in caplog.text
    assert [r.clean for r in periodic_repair(rp.conn, rp.index)] == [True]


# ---------- 真实 OpenSearch ----------

@pytest.mark.skipif(not OS_URL, reason="未设置 EAER_TEST_OPENSEARCH_URL")
def test_repair_on_real_opensearch_with_the_command(conn, db_url, tmp_path, monkeypatch, capsys):
    """真实引擎上：一份正常写入，一份写索引用完尝试次数，另有数据库里没有的版本留在索引里。
    `reconcile` 只读、有差异时退出码 1；`reconcile --repair` 补任务并删掉多余的；后台任务执行后对账为 0、检索得到。"""
    from eaer.ingestion.__main__ import main
    from eaer.ingestion.indexer import fresh_index
    from eaer.retrieval.opensearch import connect

    alias = f"eaer-repair-{uuid.uuid4().hex[:8]}"
    with fresh_index(connect(OS_URL), alias, build_id="b1", dims=DIMS) as ix:
        env = setup(conn, tmp_path, ix)
        a = approve(env, 1)
        env.worker.run_until_idle()
        b = approve(env, 2)
        env.service.down = 10 ** 6
        sleep, _ = waiting(env)
        env.worker.run_until_settled(sleep=sleep)
        orphan = str(uuid.uuid4())
        ix.upsert([stray(orphan, "x1")])
        a_id, b_id = str(a.version_id), str(b.version_id)

        for k, v in {"EAER_DATABASE_URL": db_url, "EAER_OPENSEARCH_URL": OS_URL, "EAER_INDEX_ALIAS": alias}.items():
            monkeypatch.setenv(k, v)
        capsys.readouterr()
        assert main(["reconcile"]) == 1
        assert json.loads(capsys.readouterr().out) == {"t1": {"missing_in_index": [b_id], "unexpected_in_index": [orphan],
                                                              "incomplete_in_index": []}}
        assert main(["reconcile", "--repair"]) == 0
        out = json.loads(capsys.readouterr().out)["t1"]
        assert (out["enqueued"], out["deleted"], out["pending"]) == ([b_id], [orphan], [])

        env.service.down = 0
        assert env.worker.run_until_idle() == 1
        assert ix.version_ids("t1") == {a_id, b_id} and reconcile(conn, ix, "t1") == CLEAN
        assert main(["reconcile"]) == 0
        hits = Retriever(conn, ix, HashingEmbedder(DIMS)).search(
            Scope("t1", frozenset({"R_STAFF"})), "2号城市住宿上限", today=date(2026, 3, 1))
        assert hits and hits[0].version_id == b_id
