"""异步执行（开发文档 8.4、8.5、11.2、AC-8-6）。

经接口的每次提问都是一个查询任务（query_task，与 query_log 同 id），按 engine 的图执行，每个节点之后把状态写入
LangGraph 的 PostgreSQL 检查点（thread_id 为提问记录 id）：
- 同步时限：接口进程在请求线程之外执行，最多等 budgets.sync_seconds（初值 30 秒）。超时即返回 running，
  执行线程在当前节点结束后交出任务（status=queued、background=true），由后台任务进程从检查点接着执行；
- 进度：每个节点之后写入 query_task.progress（已完成的步骤、下一步、当前模式路径、检索轮数），并把当前的
  mode、mode_path 写入 query_log，会话接口据此展示；
- 取消：记下取消请求。排队中的任务立即取消；执行中的在当前节点结束后停止，不再执行后面的节点；
- 进程重启：执行者以租约持有任务，执行期间后台线程定期续约。进程中断后租约到期，后台任务进程领取任务，
  从最后一个检查点继续，已完成的节点不再执行。恢复时按提问人身份重新解析授权范围，期间撤销的权限即时生效；
  授权服务暂不可用时任务退回队列，等待 RETRY_DELAY 后再试（进度中记 retry_after），达到最大尝试次数标为失败。
每个任务同一时间只有一个执行者：领取、续约、写进度、结束都以 locked_by 为条件，失去租约的执行者停下不写。
"""
from __future__ import annotations

import os
import socket
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Callable

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from eaer.adapters.authorizer import AuthorizationError, AuthorizerUnavailable
from eaer.domain.identity import Identity, Scope
from eaer.orchestrator.engine import NODE_NAMES, QueryResult, log_fields

LEASE = timedelta(seconds=60)
HEARTBEAT_SECONDS = 15.0
MAX_ATTEMPTS = 3
RETRY_DELAY = timedelta(seconds=30)


class CheckpointSchemaError(Exception):
    pass


class _Lost(Exception):
    """本执行者已不再持有任务的租约。"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def checkpointer(conn: psycopg.Connection):
    """检查点存取器。表由迁移 0005 建立；库的迁移步骤数与库表记录的不一致时报错（升级库要写新的迁移）。"""
    from langgraph.checkpoint.postgres import PostgresSaver

    have = conn.execute("SELECT max(v) AS v FROM checkpoint_migrations").fetchone()["v"]
    want = len(PostgresSaver.MIGRATIONS) - 1
    if have != want:
        raise CheckpointSchemaError(f"检查点表的版本是 {have}，langgraph-checkpoint-postgres 需要 {want}：写新的迁移补上")
    return PostgresSaver(conn)


def progress_of(values: dict, next_nodes: tuple) -> dict:
    return {"done": [NODE_NAMES[s["node"]] for s in values.get("steps", []) if s["node"] in NODE_NAMES],
            "next": NODE_NAMES.get(next_nodes[0]) if next_nodes else None,
            "mode_path": values.get("mode_path", []), "retrieval_rounds": values.get("rounds", 0)}


def create(conn: psycopg.Connection, *, query_id: uuid.UUID, session_id: uuid.UUID, ident: Identity, scope: Scope,
           question: str, asked_at: datetime, prepared: dict, model_id: str, index_build_id: str, owner: str,
           lease: timedelta = LEASE) -> None:
    """建立提问记录（running）与查询任务，任务由 owner 持有租约。两条记录在同一事务中写入。"""
    authz = {"principals": sorted(scope.principals),
             "expires_at": scope.expires_at.isoformat() if scope.expires_at else None,
             "removed_evidence": [], "alerts": []}
    identity = {"user": ident.user, "tenant_id": ident.tenant_id, "roles": sorted(ident.roles)}
    with conn.transaction():
        conn.execute(
            "INSERT INTO query_log (id, tenant_id, session_id, user_id, asked_at, question, time_basis, mode, mode_path, "
            "status, authz_decision, model_id, index_build_id, cost) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'running', %s, %s, %s, %s)",
            (query_id, ident.tenant_id, session_id, ident.user, asked_at, question, Jsonb(prepared["tb"]),
             prepared["mode_path"][-1], prepared["mode_path"], Jsonb(authz), model_id, index_build_id,
             Jsonb({"router": prepared["route"]})))
        conn.execute(
            "INSERT INTO query_task (id, tenant_id, session_id, user_id, identity, state, status, progress, locked_by, "
            "locked_until, attempts, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, 'running', %s, %s, %s, 1, %s, %s)",
            (query_id, ident.tenant_id, session_id, ident.user, Jsonb(identity), Jsonb(prepared),
             Jsonb(progress_of(prepared, ("first",))), owner, asked_at + lease, asked_at, asked_at))


def cancel(conn: psycopg.Connection, session_id: uuid.UUID, *, tenant_id: str, user_id: str,
           now: datetime) -> list[dict]:
    """取消会话中进行中的查询：排队中的立即取消，执行中的在当前节点结束后停止（返回 cancelling）。"""
    rows = conn.execute(
        "UPDATE query_task SET cancel_requested_at = coalesce(cancel_requested_at, %s), updated_at = %s "
        "WHERE session_id = %s AND tenant_id = %s AND user_id = %s AND status IN ('queued', 'running') "
        "RETURNING id, status", (now, now, session_id, tenant_id, user_id)).fetchall()
    out = []
    for r in rows:
        done = False
        if r["status"] == "queued":
            with conn.transaction():
                done = conn.execute("UPDATE query_task SET status = 'cancelled', updated_at = %s "
                                    "WHERE id = %s AND status = 'queued' RETURNING id", (now, r["id"])).fetchone()
                if done:
                    conn.execute("UPDATE query_log SET status = 'cancelled' WHERE id = %s AND status = 'running'",
                                 (r["id"],))
        out.append({"query_id": str(r["id"]), "status": "cancelled" if done else "cancelling"})
    return out


class TaskRunner:
    """执行者：接口进程用它在同步时限内执行，后台任务进程用它领取并接着执行。

    engine_factory(conn, tenant_id) -> Engine；authorizer_factory(conn) -> Authorizer（恢复时重新解析授权范围）。
    每次执行打开自己的数据库连接，可以在线程里运行。
    """

    def __init__(self, db_url: str, engine_factory, authorizer_factory=None, *,
                 clock: Callable[[], datetime] = _utcnow, owner: str | None = None, lease: timedelta = LEASE,
                 heartbeat_seconds: float = HEARTBEAT_SECONDS, max_attempts: int = MAX_ATTEMPTS):
        self.db_url, self.engine_factory, self.authorizer_factory = db_url, engine_factory, authorizer_factory
        self.clock, self.lease, self.heartbeat_seconds, self.max_attempts = clock, lease, heartbeat_seconds, max_attempts
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.db_url, autocommit=True, row_factory=dict_row)

    # ---------- 执行 ----------

    def run(self, query_id: uuid.UUID, *, scope: Scope | None = None,
            handoff: threading.Event | None = None) -> tuple[str, QueryResult | None]:
        """执行或接着执行一个本执行者已持有租约的任务。

        返回 (结果, 回答)：completed / cancelled / handed_off（超过同步时限，已交给后台）/ retry（授权服务暂不可用，
        已退回队列）/ lost（租约已被别的执行者接手）。执行出错时任务与提问记录标为失败，异常继续抛出。
        """
        with self.connect() as conn:
            task = conn.execute("SELECT * FROM query_task WHERE id = %s", (query_id,)).fetchone()
            if task is None or task["status"] != "running" or task["locked_by"] != self.owner:
                return "lost", None
            try:
                if scope is None:
                    scope = self._resolve(conn, task)
            except AuthorizerUnavailable as e:
                if task["attempts"] >= self.max_attempts:
                    self._fail(conn, query_id, e)
                    raise
                self._release(conn, query_id, error=f"授权服务不可用：{e}", retry_after=self.clock() + RETRY_DELAY)
                return "retry", None
            except AuthorizationError as e:
                self._fail(conn, query_id, e)
                raise
            eng = self.engine_factory(conn, task["tenant_id"])
            eng.scope = scope
            sid = str(task["session_id"])
            try:
                with self._heartbeat(query_id):
                    return self._execute(conn, eng, task, scope, sid, handoff)
            except _Lost:
                return "lost", None
            except Exception as e:
                self._fail(conn, query_id, e)
                raise

    def _execute(self, conn, eng, task, scope: Scope, sid: str, handoff):
        qid = task["id"]
        graph = eng.graph(checkpointer(conn))
        config = {"configurable": {"thread_id": str(qid)}}
        snap = graph.get_state(config)
        if task["cancel_requested_at"]:
            return self._cancelled(conn, eng, qid, snap.values or task["state"], sid, scope)
        for _ in graph.stream(None if snap.values else task["state"], config, stream_mode="updates", durability="sync"):
            snap = graph.get_state(config)
            row = conn.execute(
                "UPDATE query_task SET progress = %s, locked_until = %s, updated_at = %s "
                "WHERE id = %s AND status = 'running' AND locked_by = %s RETURNING cancel_requested_at",
                (Jsonb(progress_of(snap.values, snap.next)), self.clock() + self.lease, self.clock(), qid,
                 self.owner)).fetchone()
            if row is None:
                raise _Lost()
            conn.execute("UPDATE query_log SET mode = %s, mode_path = %s WHERE id = %s AND status = 'running'",
                         (snap.values["mode_path"][-1], snap.values["mode_path"], qid))
            if row["cancel_requested_at"]:
                return self._cancelled(conn, eng, qid, snap.values, sid, scope)
            if snap.next and handoff is not None and handoff.is_set():
                self._release(conn, qid)
                return "handed_off", None
        r = eng.result(graph.get_state(config).values, sid)
        self._finish(conn, qid, r, scope, "completed")
        return "completed", r

    def _cancelled(self, conn, eng, qid, values: dict, sid: str, scope: Scope):
        r = eng.result(values, sid, status="cancelled")
        r.answer, r.claims, r.abstained = None, [], False          # 没有给用户的内容不记为回答
        self._finish(conn, qid, r, scope, "cancelled")
        return "cancelled", None

    def _finish(self, conn, qid, r: QueryResult, scope: Scope, status: str) -> None:
        f = log_fields(r, scope)
        with conn.transaction():                       # 失去租约时整个事务回滚，提问记录也不改
            done = conn.execute(
                "UPDATE query_task SET status = %s, locked_by = NULL, locked_until = NULL, updated_at = %s "
                "WHERE id = %s AND status = 'running' AND locked_by = %s RETURNING id",
                (status, self.clock(), qid, self.owner)).fetchone()
            if done is None:
                raise _Lost()
            conn.execute("UPDATE query_log SET " + ", ".join(f"{k} = %s" for k in f) + " WHERE id = %s",
                         (*f.values(), qid))

    def _fail(self, conn, qid, e: BaseException) -> None:
        msg = f"{type(e).__name__}: {e}"[:2000]
        with conn.transaction():
            done = conn.execute(
                "UPDATE query_task SET status = 'failed', last_error = %s, locked_by = NULL, locked_until = NULL, "
                "updated_at = %s WHERE id = %s AND status = 'running' AND locked_by = %s RETURNING id",
                (msg, self.clock(), qid, self.owner)).fetchone()
            if done:
                conn.execute("UPDATE query_log SET status = 'failed' WHERE id = %s AND status = 'running'", (qid,))

    def _release(self, conn, qid, error: str | None = None, retry_after: datetime | None = None) -> None:
        """退回队列：超过同步时限交给后台（立即可领取），或等待重试（retry_after 之前不领取）。"""
        wait = {"retry_after": retry_after.isoformat(), "waiting": error} if retry_after else {}
        conn.execute(
            "UPDATE query_task SET status = 'queued', background = true, locked_by = NULL, locked_until = NULL, "
            "last_error = coalesce(%s, last_error), progress = (progress - 'retry_after' - 'waiting') || %s, "
            "updated_at = %s WHERE id = %s AND status = 'running' AND locked_by = %s",
            (error, Jsonb(wait), self.clock(), qid, self.owner))

    def _resolve(self, conn, task) -> Scope:
        if self.authorizer_factory is None:
            raise AuthorizerUnavailable("后台执行没有配置授权实现")
        i = task["identity"]
        return self.authorizer_factory(conn).resolve(Identity(i["user"], i["tenant_id"], frozenset(i["roles"])))

    @contextmanager
    def _heartbeat(self, qid):
        """执行期间定期续约，节点耗时较长时租约也不会到期。"""
        stop = threading.Event()

        def beat():
            with self.connect() as c:
                while not stop.wait(self.heartbeat_seconds):
                    c.execute("UPDATE query_task SET locked_until = %s WHERE id = %s AND status = 'running' "
                              "AND locked_by = %s", (self.clock() + self.lease, qid, self.owner))

        t = threading.Thread(target=beat, name=f"eaer-lease-{qid}", daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            t.join(timeout=5)

    # ---------- 后台任务进程 ----------

    def claim(self, conn: psycopg.Connection) -> uuid.UUID | None:
        """领取一个排队中的任务，或租约已到期（执行者中断）的任务；尝试次数用尽的标为失败。"""
        now = self.clock()
        with conn.transaction():
            for r in conn.execute(
                    "UPDATE query_task SET status = 'failed', locked_by = NULL, locked_until = NULL, updated_at = %s, "
                    "last_error = coalesce(last_error || '；', '') || '租约到期且已达最大尝试次数' "
                    "WHERE status = 'running' AND locked_until < %s AND attempts >= %s RETURNING id",
                    (now, now, self.max_attempts)).fetchall():
                conn.execute("UPDATE query_log SET status = 'failed' WHERE id = %s AND status = 'running'", (r["id"],))
        row = conn.execute(
            "UPDATE query_task SET status = 'running', locked_by = %s, locked_until = %s, attempts = attempts + 1, "
            "background = true, updated_at = %s WHERE id = (SELECT id FROM query_task WHERE attempts < %s AND "
            "((status = 'queued' AND coalesce((progress->>'retry_after')::timestamptz, '-infinity') <= %s) "
            " OR (status = 'running' AND locked_until < %s)) ORDER BY created_at "
            "FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING id",
            (self.owner, now + self.lease, now, self.max_attempts, now, now)).fetchone()
        return row["id"] if row else None

    def run_pending(self, limit: int = 100) -> int:
        """领取并执行任务，直到没有可领取的任务或达到 limit；返回处理的任务数。单个任务出错不影响其他任务。"""
        n = 0
        with self.connect() as conn:
            while n < limit:
                qid = self.claim(conn)
                if qid is None:
                    break
                n += 1
                try:
                    self.run(qid)
                except Exception:  # noqa: BLE001 — 已记入任务与提问记录
                    pass
        return n


# ---------- Trace 清理（10.2、AC-10-10） ----------

def purge(conn: psycopg.Connection, now: datetime, *, limit: int = 5000) -> dict[str, int]:
    """删除超过租户 trace_retention_days（默认 30 天）的执行记录：已结束任务的检查点（含证据原文的各步状态）与任务行；
    同时删除已过期的回答缓存。执行中、排队中的任务不动；query_log 是审计记录（10.4），不在此清理。
    每次最多处理 limit 个任务，后台任务进程定期调用，积压时分几次删完。"""
    with conn.transaction():
        old = [r["id"] for r in conn.execute(
            "SELECT t.id FROM query_task t LEFT JOIN tenant_config c ON c.tenant_id = t.tenant_id "
            "WHERE t.status IN ('completed', 'cancelled', 'failed') "
            "AND t.updated_at < %s - make_interval(days => coalesce(c.trace_retention_days, 30)) "
            "ORDER BY t.updated_at LIMIT %s", (now, limit))]
        threads = [str(i) for i in old]
        checkpoints = conn.execute("DELETE FROM checkpoints WHERE thread_id = ANY(%s)", (threads,)).rowcount
        conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ANY(%s)", (threads,))
        conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ANY(%s)", (threads,))
        tasks = conn.execute("DELETE FROM query_task WHERE id = ANY(%s)", (old,)).rowcount
        cached = conn.execute("DELETE FROM query_cache WHERE expires_at <= %s", (now,)).rowcount
    return {"tasks": tasks, "checkpoints": checkpoints, "cache": cached}
