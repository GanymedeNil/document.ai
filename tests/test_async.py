"""AC-8-6 异步执行（开发文档 8.4、8.5、11.2）：超时转后台、查看进度、取消、进程重启后从检查点恢复。离线：脚本化模型、
内存索引，文档经真实的上传、审核、入库与索引流程；检查点写入真实的 PostgreSQL（LangGraph PostgresSaver）。

- 超过同步时限：接口返回 running 且其余字段为空；执行线程在当前步骤结束后交出，后台任务进程从检查点接着执行，
  已完成的步骤不再执行（模型调用次数不变）；流式请求同样以 running 的 done 结束；
- 进度：执行中的查询在会话接口里显示已完成的步骤、下一步与当前模式路径；
- 取消：执行中的在当前步骤结束后停止，排队中的立即取消，取消后后台不再执行；没有进行中的查询时 409；
- 进程中断：执行者中断后任务保持租约，租约未到期别人不能接手；到期后后台任务进程领取并从最后一个检查点继续；
- 恢复时重新解析授权范围，期间撤销的权限即时生效；返回前再核对一次所引证据的权限与状态（10.1）。
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from eaer.adapters.authorizer import LocalAuthorizer
from eaer.adapters.online import _COLLECTOR, CallRecord
from eaer.api.app import TokenAuthenticator, create_app
from eaer.domain.identity import Identity
from eaer.ingestion.indexer import Indexer
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.orchestrator import tasks
from eaer.orchestrator.engine import Engine
from eaer.orchestrator.tasks import TaskRunner
from eaer.retrieval.embed import HashingEmbedder
from eaer.retrieval.index import MemoryIndex
from eaer.retrieval.service import Retriever
from eaer.storage import LocalFileStore
from test_engine import Scripted, claim
from test_query_api import assert_log_complete

TOKENS = {"tok-ed": Identity("ed", "t1", frozenset({"editor"})),
          "tok-rv": Identity("rv", "t1", frozenset({"reviewer"})),
          "tok-staff": Identity("staff", "t1", frozenset())}
NOW = datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc)
Q = "一线城市住宿上限是多少"
ANSWER = lambda p: {"abstain": False, "claims": [claim("一线城市住宿上限为每晚650元。", p, "650元")]}  # noqa: E731


class Crash(BaseException):
    """模拟进程被杀：执行者来不及做任何收尾。"""


class Gated(Scripted):
    """可以在某类模型调用处停住，用来把执行卡在某个步骤里。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gates: dict[str, tuple[threading.Event, threading.Event]] = {}
        self.crash_once: set[str] = set()

    def gate(self, kind: str):
        self.gates[kind] = (threading.Event(), threading.Event())
        return self.gates[kind]

    def generate(self, messages, schema):
        props = (schema or {}).get("properties", {})
        kind = "judge" if "verdict" in props else "time" if "as_of_valid" in props else "draft"
        if kind in self.crash_once:
            self.crash_once.discard(kind)
            self.calls.append(f"{kind}-crash")
            raise Crash()
        if kind in self.gates:
            entered, release = self.gates[kind]
            entered.set()
            assert release.wait(10), "测试没有放行"
        out = super().generate(messages, schema)
        calls = _COLLECTOR.get()                 # 与线上客户端一样记下调用，核对成本跨步骤、跨进程累加
        if calls is not None:
            calls.append(CallRecord("generate", "scripted", 10, 2, 1, 0.01, True))
        return out


def wait_for(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    raise AssertionError("等待超时")


def H(t):
    return {"Authorization": f"Bearer {t}"}


@pytest.fixture
def aq(db_url, tmp_path):
    store, index, emb = LocalFileStore(tmp_path / "files"), MemoryIndex(), HashingEmbedder()
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    accounts = {"staff": {"tenant": "t1", "principals": ["R_STAFF"]}}
    st = {"model": Gated(ANSWER), "sync": 5.0}

    def engine(c, tenant):
        return Engine(c, Retriever(c, index, emb), st["model"], LocalAuthorizer(c, accounts), store=store,
                      budgets={"sync_seconds": st["sync"]})

    def authz(c):
        return LocalAuthorizer(c, accounts)

    client = TestClient(create_app(db_url, store, TokenAuthenticator(TOKENS), lambda: NOW, authorizer=authz,
                                   engine=engine, owner="test"))
    ingest = Worker(Ingestion(conn, store, lambda: NOW), Indexer(conn, index, emb, "b1"))
    body = ("<h1>差旅报销管理办法</h1><p>第一条　为规范差旅报销，制定本办法。</p><p>第二条　一线城市住宿上限为每晚650元。</p>"
            "<p>第三条　本办法自2026年1月1日起施行。</p>")
    up = client.post("/v1/documents", headers=H("tok-ed"), data={"principals": "R_STAFF"},
                     files={"file": ("p.html", body.encode(), "text/html")}).json()
    ingest.run_until_idle()
    task = client.get("/v1/review/tasks", headers=H("tok-rv")).json()["items"][0]["task_id"]
    client.post("/v1/review/decisions", headers=H("tok-rv"), json={"decisions": [
        {"task_id": task, "decision": "approve", "corrections": {"doc_type": "normative"}}]})
    ingest.run_until_idle()

    def bg(owner="worker"):
        """后台任务进程的执行者。"""
        return TaskRunner(db_url, engine, authz, clock=lambda: NOW, owner=owner, heartbeat_seconds=0.05)

    ns = type("NS", (), {})()
    ns.__dict__.update(client=client, conn=conn, st=st, bg=bg, accounts=accounts, doc=up["document_id"],
                       ingest=ingest, engine=engine, authz=authz, db_url=db_url)
    yield ns
    conn.close()


def task_row(conn):
    return conn.execute("SELECT * FROM query_task").fetchone()


def ask(aq, **kw):
    return aq.client.post("/v1/query", headers=H("tok-staff"), json={"question": Q, **kw})


def run_in_background(aq, **kw):
    """把一次提问卡在“验证”步骤里，直到接口超时返回 running。前两步在毫秒级完成，1 秒的时限足够它们走到“验证”。"""
    aq.st["sync"] = 1.0
    entered, release = aq.st["model"].gate("judge")
    r = ask(aq, **kw)
    assert entered.wait(5)
    return r, release


# ---------- 超时转后台、进度、后台接着执行 ----------

def test_timeout_hands_off_and_the_worker_resumes_from_the_checkpoint(aq):
    r, release = run_in_background(aq)
    b = r.json()
    assert r.status_code == 200 and b["status"] == "running" and b["session_id"]
    assert b["answer"] is None and b["claims"] == [] and b["mode"] is None and b["time_basis"] is None
    item = aq.client.get(f"/v1/sessions/{b['session_id']}", headers=H("tok-staff")).json()["items"][0]
    assert item["status"] == "running" and item["answer"] is None
    assert item["progress"]["done"] == ["检索与生成", "升级判断"] and item["progress"]["next"] == "验证"
    release.set()                                            # 执行线程做完“验证”后交出任务
    wait_for(lambda: task_row(aq.conn)["status"] == "queued")
    t = task_row(aq.conn)
    assert t["background"] and t["locked_by"] is None
    assert aq.conn.execute("SELECT count(*) AS n FROM checkpoints WHERE thread_id = %s",
                           (str(t["id"]),)).fetchone()["n"] > 0
    assert aq.bg().run_pending() == 1                        # 后台任务进程：只剩“组装”
    calls = aq.st["model"].calls
    assert calls.count("draft") == 1 and calls.count("judge") == 1          # 已完成的步骤没有重做
    item = aq.client.get(f"/v1/sessions/{b['session_id']}", headers=H("tok-staff")).json()["items"][0]
    assert item["status"] == "completed" and "650" in item["answer"] and item["background"]
    assert item["mode_path"] == ["fast", "verified"] and "progress" not in item
    rows = assert_log_complete(aq.conn)
    assert rows[0]["status"] == "completed" and rows[0]["cost"]["by_kind"]["generate"]["calls"] == 2  # 两段执行累加
    assert task_row(aq.conn)["status"] == "completed" and aq.bg().run_pending() == 0


def test_streaming_request_ends_with_running_when_handed_off(aq):
    r, release = run_in_background(aq, stream=True)
    events = [(m.group(1), json.loads(m.group(2))) for m in re.finditer(r"event: (\w+)\ndata: (.*)\n\n", r.text)]
    assert [e for e, _ in events] == ["mode", "notice", "done"]
    assert events[0][1] == {"mode": "fast"} and events[-1][1]["status"] == "running"
    release.set()
    wait_for(lambda: task_row(aq.conn)["status"] == "queued")
    aq.bg().run_pending()
    assert task_row(aq.conn)["status"] == "completed"


def test_fast_queries_complete_within_the_time_limit(aq):
    b = ask(aq).json()
    assert b["status"] == "completed" and "650" in b["answer"]
    t = task_row(aq.conn)
    assert t["status"] == "completed" and not t["background"] and t["attempts"] == 1


def test_checkpointer_does_not_share_the_engine_connection(aq, monkeypatch):
    """检查点与节点不共用连接：LangGraph 在后台线程里写各节点的结果（put_writes），共用时两边会同时在一个连接上
    发命令（psycopg：another command is already in progress）。2026-11-16 首次在 CI 上运行时 10 个用例因此失败，
    本机快、碰不上。"""
    from eaer.orchestrator.engine import Engine

    seen, real = [], Engine.graph

    def graph(self, checkpointer=None):
        seen.append((self.conn, getattr(checkpointer, "conn", None)))
        return real(self, checkpointer)

    monkeypatch.setattr(Engine, "graph", graph)
    assert ask(aq).json()["status"] == "completed"
    assert seen and all(cp is not None and cp is not c for c, cp in seen)


# ---------- 取消 ----------

def test_cancel_a_running_query_stops_after_the_current_step(aq):
    r, release = run_in_background(aq)
    sid = r.json()["session_id"]
    c = aq.client.post(f"/v1/sessions/{sid}/cancel", headers=H("tok-staff"))
    assert c.status_code == 200 and c.json()["items"][0]["status"] == "cancelling"
    release.set()
    wait_for(lambda: task_row(aq.conn)["status"] == "cancelled")
    log = aq.conn.execute("SELECT * FROM query_log").fetchone()
    assert log["status"] == "cancelled" and log["answer"] is None and log["claims"] == []
    assert aq.bg().run_pending() == 0                        # 取消后不再执行
    item = aq.client.get(f"/v1/sessions/{sid}", headers=H("tok-staff")).json()["items"][0]
    assert item["status"] == "cancelled"
    again = aq.client.post(f"/v1/sessions/{sid}/cancel", headers=H("tok-staff"))
    assert again.status_code == 409 and again.json()["code"] == "not_running"


def test_cancel_a_queued_query_is_immediate(aq):
    r, release = run_in_background(aq)
    sid = r.json()["session_id"]
    release.set()
    wait_for(lambda: task_row(aq.conn)["status"] == "queued")
    c = aq.client.post(f"/v1/sessions/{sid}/cancel", headers=H("tok-staff")).json()
    assert c["items"][0]["status"] == "cancelled" and task_row(aq.conn)["status"] == "cancelled"
    assert aq.bg().run_pending() == 0
    assert aq.conn.execute("SELECT status FROM query_log").fetchone()["status"] == "cancelled"


def test_cancel_other_users_or_unknown_sessions_is_not_found(aq):
    sid = ask(aq).json()["session_id"]
    for path in (f"/v1/sessions/{uuid.uuid4()}/cancel", "/v1/sessions/nope/cancel"):
        assert aq.client.post(path, headers=H("tok-staff")).status_code == 404
    assert aq.client.post(f"/v1/sessions/{sid}/cancel", headers=H("tok-ed")).status_code == 404


# ---------- 进程中断后从检查点恢复 ----------

def start_task(aq, owner: str):
    """不经接口建立一个由 owner 持有的任务，模拟接口进程收到提问。"""
    scope = LocalAuthorizer(aq.conn, aq.accounts).resolve(TOKENS["tok-staff"])
    eng = aq.engine(aq.conn, "t1")
    eng.scope = scope
    qid = uuid.uuid4()
    tasks.create(aq.conn, query_id=qid, session_id=uuid.uuid4(), ident=TOKENS["tok-staff"], scope=scope, question=Q,
                 asked_at=NOW, prepared=eng.prepare(Q, today=NOW.date()), model_id="scripted",
                 index_build_id="MemoryIndex", owner=owner)
    return qid, scope


def test_interrupted_executor_is_resumed_from_the_last_checkpoint(aq):
    model = aq.st["model"]
    model.crash_once.add("judge")
    qid, scope = start_task(aq, "api-dead")
    with pytest.raises(Crash):                               # 接口进程在“验证”步骤中被杀
        aq.bg("api-dead").run(qid, scope=scope)
    t = task_row(aq.conn)
    assert t["status"] == "running" and t["locked_by"] == "api-dead"
    assert t["progress"]["done"] == ["检索与生成", "升级判断"]
    assert aq.bg().run_pending() == 0                        # 租约未到期：别人不能接手
    aq.conn.execute("UPDATE query_task SET locked_until = %s", (NOW - timedelta(seconds=1),))
    assert aq.bg().run_pending() == 1                        # 租约到期：后台任务进程接手
    t = task_row(aq.conn)
    assert t["status"] == "completed" and t["attempts"] == 2 and t["background"]
    assert model.calls.count("draft") == 1 and model.calls.count("judge") == 1   # 只重做中断的“验证”
    log = aq.conn.execute("SELECT * FROM query_log").fetchone()
    assert log["status"] == "completed" and "650" in log["answer"]


def test_tasks_fail_after_too_many_interrupted_attempts(aq):
    qid, scope = start_task(aq, "api-dead")
    aq.conn.execute("UPDATE query_task SET attempts = 3, locked_until = %s", (NOW - timedelta(seconds=1),))
    assert aq.bg().run_pending() == 0
    assert task_row(aq.conn)["status"] == "failed"
    assert aq.conn.execute("SELECT status FROM query_log").fetchone()["status"] == "failed"


def test_resumed_query_uses_current_permissions(aq):
    r, release = run_in_background(aq)
    release.set()
    wait_for(lambda: task_row(aq.conn)["status"] == "queued")
    aq.accounts["staff"]["principals"] = ["R_OTHER"]          # 后台执行之前，提问人的权限被撤销
    aq.bg().run_pending()
    log = aq.conn.execute("SELECT * FROM query_log").fetchone()
    assert log["status"] == "completed" and log["abstained"] and "650" not in log["answer"]
    assert log["authz_decision"]["principals"] == ["R_OTHER"]
    assert any("返回前核对未通过" in a for a in log["authz_decision"]["alerts"])


def test_archived_before_assembly_is_removed_from_the_answer(aq):
    r, release = run_in_background(aq)
    release.set()
    wait_for(lambda: task_row(aq.conn)["status"] == "queued")
    assert aq.client.post(f"/v1/documents/{aq.doc}/archive", headers=H("tok-ed")).status_code == 200
    aq.bg().run_pending()
    log = aq.conn.execute("SELECT * FROM query_log").fetchone()
    assert log["abstained"] and "650" not in log["answer"]


def test_authorizer_outage_on_resume_requeues_the_task(aq):
    from eaer.adapters.authorizer import AuthorizerUnavailable

    class Down:
        def resolve(self, i):
            raise AuthorizerUnavailable("down")

    r, release = run_in_background(aq)
    release.set()
    wait_for(lambda: task_row(aq.conn)["status"] == "queued")
    down = TaskRunner(aq.db_url, aq.engine, lambda c: Down(), clock=lambda: NOW, owner="w2")
    assert down.run_pending() == 1                           # 退回队列，不在同一轮里反复重试
    t = task_row(aq.conn)
    assert t["status"] == "queued" and "授权服务不可用" in t["last_error"] and t["progress"]["retry_after"]
    assert aq.bg().run_pending() == 0                        # 等待期内不领取
    later = TaskRunner(aq.db_url, aq.engine, aq.authz, clock=lambda: NOW + timedelta(minutes=1), owner="w3")
    assert later.run_pending() == 1
    t = task_row(aq.conn)
    assert t["status"] == "completed" and t["attempts"] == 3
