"""对外接口：提问、会话、上传、审核、归档、取消归档、证据、强制验证清单（开发文档 11.1、11.2）。

接口的唯一定义是同目录的 openapi.yaml，/openapi.json 与 /docs 展示的就是它；实现与它的一致性由契约测试
tests/test_contract.py 检查（AC-11-1）。改接口时先改 openapi.yaml。

注意：本模块不能使用 from __future__ import annotations，FastAPI 需要在运行时解析依赖注解。

身份校验（Authenticator）的参考实现是配置好的令牌表，客户现场换成 SSO 时只换它。
文档权限走授权接口（adapters.authorizer.Authorizer，10.1 第 2、5 步）：未配置或不可用时返回 503，拒绝请求。
错误一律返回 {"code", "message", "request_id"}；无权访问与不存在不区分，都返回 404。
"""

import functools
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Callable, Iterator, Literal, Protocol

import psycopg
import yaml
from fastapi import Depends, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from eaer.adapters.authorizer import Authorizer, AuthorizationError, AuthorizerUnavailable, ScopeExpired
from eaer.domain.identity import Identity, Scope
from eaer.ingestion.parsers import UnsupportedFormat
from eaer.ingestion.pipeline import Ingestion
from eaer.lifecycle.service import InvalidState, Lifecycle, utcnow
from eaer.orchestrator import forced_verify as fv
from eaer.review.service import Review
from eaer.storage import FileStore

MAX_DECISIONS = 1000
OPENAPI_FILE = Path(__file__).with_name("openapi.yaml")


@functools.cache
def openapi_spec() -> dict:
    """对外接口的定义（openapi.yaml）。"""
    with OPENAPI_FILE.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class Authenticator(Protocol):
    def authenticate(self, token: str) -> Identity | None: ...


class TokenAuthenticator:
    """参考实现：令牌 -> 身份，来自配置文件。"""

    def __init__(self, tokens: dict[str, Identity]):
        self.tokens = tokens

    def authenticate(self, token: str) -> Identity | None:
        return self.tokens.get(token)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


Grade = Literal["explicit", "inferred", "unknown"]


class DatedValue(BaseModel):
    value: date
    grade: Grade = "explicit"


class RelationInput(BaseModel):
    relation: Literal["supersedes", "conflicts", "references"] = "supersedes"
    to_document_id: uuid.UUID
    to_version_no: int = Field(ge=1)


class Corrections(BaseModel):
    """审核人的更正（6.3）。字段不给与给 null 含义不同（null 表示清除），转交审核服务时只带出现过的字段。
    doc_type 不能为 null：类型类注解里不含 None，默认值不经校验，所以“不给”可以、显式 null 被拒绝。"""
    doc_type: Literal["normative", "record"] = None
    valid_from: DatedValue | None = None
    valid_to: DatedValue | None = None
    relations: list[RelationInput] | None = None


class Decision(BaseModel):
    task_id: str
    decision: Literal["approve", "delete"]
    corrections: Corrections | None = None


class Decisions(BaseModel):
    decisions: list[Decision] = Field(min_length=1, max_length=MAX_DECISIONS)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: uuid.UUID | None = None
    as_of_valid: date | None = None
    as_of_system: date | None = None
    mode_hint: Literal["fast", "deep", "verified"] | None = None
    stream: bool = False


class ForcedVerifyRule(BaseModel):
    kind: Literal["keyword", "document", "doc_type", "all"]
    value: str | None = None
    description: str | None = Field(default=None, max_length=200)


def create_app(database_url: str, store: FileStore, auth: Authenticator,
               clock: Callable[[], datetime] = utcnow,
               authorizer: Callable[[psycopg.Connection], Authorizer] | None = None,
               engine: Callable[[psycopg.Connection, str], Any] | None = None, *,
               query_threads: int = 8, owner: str | None = None) -> FastAPI:
    """authorizer：给定数据库连接返回授权接口实例；为 None 时需要文档权限的接口一律 503。
    engine：给定数据库连接与租户返回查询编排器（orchestrator.engine.Engine）；为 None 时提问接口 503。
    query_threads：同时执行的提问数（每个提问在请求线程之外执行，以便超过同步时限时转入后台，8.5）。"""
    from eaer.orchestrator.tasks import TaskRunner

    spec = openapi_spec()
    app = FastAPI(title=spec["info"]["title"], version=spec["info"]["version"])
    app.openapi = openapi_spec                # /openapi.json 与 /docs 展示 openapi.yaml，而不是由代码推出的定义
    runner = TaskRunner(database_url, engine, authorizer, clock=clock,
                        owner=owner and f"api:{owner}") if engine is not None else None
    pool = ThreadPoolExecutor(max_workers=query_threads, thread_name_prefix="eaer-query") if engine is not None else None

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = f"req_{uuid.uuid4().hex[:16]}"
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

    def error(request: Request, status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse({"code": code, "message": message, "request_id": request.state.request_id},
                            status_code=status, headers={"X-Request-Id": request.state.request_id})

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, e: ApiError):
        return error(request, e.status, e.code, e.message)

    @app.exception_handler(InvalidState)
    async def _invalid_state(request: Request, e: InvalidState):
        return error(request, 409, "invalid_state", str(e))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, e: StarletteHTTPException):
        """框架层的错误（未知路径、方法不允许等）也用统一格式（11.1）。"""
        code = {404: "not_found", 405: "invalid_request"}.get(e.status_code, "invalid_request")
        return error(request, e.status_code, code, "资源不存在" if e.status_code == 404 else str(e.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, e: RequestValidationError):
        first = e.errors()[0] if e.errors() else {}
        where = ".".join(str(x) for x in first.get("loc", []))
        return error(request, 400, "invalid_request", f"{where}：{first.get('msg', '参数错误')}")

    def get_conn() -> Iterator[psycopg.Connection]:
        conn = psycopg.connect(database_url, autocommit=True, row_factory=dict_row)
        try:
            yield conn
        finally:
            conn.close()

    def identity(authorization: Annotated[str | None, Header()] = None) -> Identity:
        if not authorization or not authorization.startswith("Bearer "):
            raise ApiError(401, "unauthenticated", "缺少身份令牌")
        ident = auth.authenticate(authorization.removeprefix("Bearer ").strip())
        if ident is None:
            raise ApiError(401, "unauthenticated", "身份令牌无效或已过期")
        return ident

    def require(role: str):
        def dep(ident: Annotated[Identity, Depends(identity)]) -> Identity:
            if role not in ident.roles:
                raise ApiError(403, "forbidden", f"需要 {role} 角色")
            return ident
        return dep

    Conn = Annotated[psycopg.Connection, Depends(get_conn)]
    Editor = Annotated[Identity, Depends(require("editor"))]
    Reviewer = Annotated[Identity, Depends(require("reviewer"))]
    Admin = Annotated[Identity, Depends(require("admin"))]
    Anyone = Annotated[Identity, Depends(identity)]

    def authorize(conn: Conn, ident: Annotated[Identity, Depends(identity)]) -> tuple[Authorizer, Scope]:
        """10.1 第 2 步：解析授权范围；授权接口不可用时直接拒绝（AC-10-5）。"""
        if authorizer is None:
            raise ApiError(503, "authz_unavailable", "授权服务不可用")
        a = authorizer(conn)
        try:
            return a, a.resolve(ident)
        except AuthorizerUnavailable:
            raise ApiError(503, "authz_unavailable", "授权服务不可用")
        except AuthorizationError:
            raise ApiError(403, "forbidden", "无法确定授权范围")

    Authz = Annotated[tuple[Authorizer, Scope], Depends(authorize)]
    EVIDENCE_404 = (404, "not_found", "证据不存在")

    def own_document(conn, ident: Identity, document_id: str) -> uuid.UUID:
        try:
            did = uuid.UUID(document_id)
        except ValueError:
            raise ApiError(404, "not_found", "文档不存在")
        row = conn.execute("SELECT 1 FROM document WHERE id = %s AND tenant_id = %s", (did, ident.tenant_id)).fetchone()
        if row is None:
            raise ApiError(404, "not_found", "文档不存在")
        return did

    @app.post("/v1/documents", status_code=201)
    def upload(conn: Conn, ident: Editor,
               file: Annotated[UploadFile, File()],
               principals: Annotated[str, Form(description="可访问的角色，分号或逗号分隔")],
               source_ref: Annotated[str | None, Form()] = None,
               title: Annotated[str | None, Form()] = None,
               import_batch_id: Annotated[str | None, Form()] = None,
               batch_id: Annotated[uuid.UUID | None, Form()] = None):
        roles = sorted({p.strip() for p in principals.replace("；", ";").replace(",", ";").split(";") if p.strip()})
        if not roles:
            raise ApiError(400, "invalid_request", "principals 不能为空")
        lc = Lifecycle(conn, clock)
        ing = Ingestion(conn, store, clock)
        data = file.file.read()
        try:
            with conn.transaction():
                lc.ensure_tenant(ident.tenant_id)
                acl = lc.create_acl_policy(ident.tenant_id, roles)
                r = ing.receive(ident.tenant_id, data, file.filename or "upload", acl_policy_id=acl,
                                source_ref=source_ref, title=title, import_batch_id=import_batch_id,
                                batch_id=batch_id)
        except UnsupportedFormat as e:
            raise ApiError(400, "invalid_request", str(e))
        body = {"document_id": str(r.document_id), "version_id": str(r.version_id),
                "status": "pending_review", "duplicate": r.duplicate}
        return JSONResponse(body, status_code=200 if r.duplicate else 201)

    @app.get("/v1/review/tasks")
    def review_tasks(conn: Conn, ident: Reviewer, status: Literal["open", "approved", "deleted"] = "open",
                     batch_id: uuid.UUID | None = None, cursor: str | None = None,
                     limit: Annotated[int, Query(ge=1, le=200)] = 50):
        items, nxt = Review(conn, Lifecycle(conn, clock), store).list_tasks(
            ident.tenant_id, status, batch_id, cursor, limit)
        return jsonable_encoder({"items": items, "next_cursor": nxt})

    @app.post("/v1/review/decisions")
    def review_decisions(conn: Conn, ident: Reviewer, body: Decisions):
        results = Review(conn, Lifecycle(conn, clock), store).decide(
            ident.tenant_id, [d.model_dump(mode="json", exclude_unset=True) for d in body.decisions], ident.user)
        return {"results": results}

    @app.post("/v1/documents/{document_id}/archive")
    def archive(conn: Conn, ident: Editor, document_id: str):
        did = own_document(conn, ident, document_id)
        ids = Lifecycle(conn, clock).archive_document(did, ident.user)
        return {"document_id": str(did), "versions": [str(i) for i in ids], "status": "archived"}

    @app.post("/v1/documents/{document_id}/unarchive")
    def unarchive(conn: Conn, ident: Editor, document_id: str):
        did = own_document(conn, ident, document_id)
        ids = Lifecycle(conn, clock).unarchive_document(did, ident.user)
        return {"document_id": str(did), "versions": [str(i) for i in ids], "status": "searchable"}

    # ---------- 提问与会话（8、11.2） ----------

    def _own_session(conn, ident: Identity, sid: uuid.UUID) -> bool:
        """会话只属于发起它的用户；别人的会话与不存在的会话不区分（404）。"""
        rows = conn.execute("SELECT DISTINCT tenant_id, user_id FROM query_log WHERE session_id = %s", (sid,)).fetchall()
        return all(r["tenant_id"] == ident.tenant_id and r["user_id"] == ident.user for r in rows)

    def _sid(session_id: str) -> uuid.UUID:
        try:
            return uuid.UUID(session_id)
        except ValueError:
            raise ApiError(404, "not_found", "会话不存在")

    @app.post("/v1/query")
    def query(conn: Conn, ident: Anyone, authz: Authz, body: QueryRequest):
        """提问（11.2）。先查回答缓存（同租户、同授权范围、同索引构建，命中后仍核对权限）；未命中则在请求线程之外
        执行，等待同步时限（budgets.sync_seconds）；超时返回 status=running，其余字段为空，任务在当前步骤结束后
        交给后台任务进程，客户端轮询会话接口（8.5）。同步完成的回答写入缓存。"""
        from eaer.adapters.online import ModelCallError
        from eaer.orchestrator import cache, tasks
        from eaer.observability import set_attrs, span
        from eaer.orchestrator.engine import QueryResult, log_query, trace_result

        if engine is None:
            raise ApiError(503, "query_unavailable", "提问服务未配置模型或检索服务")
        if body.session_id is not None and not _own_session(conn, ident, body.session_id):
            raise ApiError(404, "not_found", "会话不存在")
        authorizer_, scope = authz
        eng = engine(conn, ident.tenant_id)
        eng.scope = scope
        sid, qid, asked = body.session_id or uuid.uuid4(), uuid.uuid4(), clock()
        model_id = getattr(getattr(eng.model, "gen", None), "model", None) or "unknown"
        build_id, ttl = eng._build_id(), float(eng.budgets["cache_ttl_seconds"])
        # 回答缓存（10.1 第 7 步）：键含租户、授权范围哈希与索引构建编号；命中后仍按当前授权范围核对所引证据
        ck = cache.key(ident.tenant_id, scope, build_id, body.question, today=asked.date(), as_of_valid=body.as_of_valid,
                       as_of_system=body.as_of_system, mode_hint=body.mode_hint, model=model_id,
                       deep_enabled=eng.deep_enabled, verified_enabled=eng.verified_enabled)
        # 一次提问一条 trace（13.3）：根 span 在这里，执行段（本进程或后台进程）经查询状态里的 traceparent 挂在它下面
        with span("eaer.query", **{"eaer.tenant": ident.tenant_id, "eaer.session_id": str(sid),
                                   "eaer.query_id": str(qid)}) as root:
            hit = cache.get(conn, ck, tenant_id=ident.tenant_id, scope=scope, authorizer=authorizer_, now=asked,
                            session_id=str(sid)) if ttl > 0 else None
            prepared = None
            if hit is not None:
                log_query(conn, hit, scope=scope, user_id=ident.user, question=body.question, asked_at=asked, query_id=qid)
                outcome, result = "completed", hit
            else:
                try:
                    prepared = eng.prepare(body.question, today=asked.date(), as_of_valid=body.as_of_valid,
                                           as_of_system=body.as_of_system, mode_hint=body.mode_hint)
                except ModelCallError as e:
                    failed = QueryResult(str(sid), "failed", "fast", ["fast"], {}, {}, None, [], False, [], [], [], 0,
                                         model_id, build_id)
                    log_query(conn, failed, scope=scope, user_id=ident.user, question=body.question, asked_at=asked,
                              query_id=qid)
                    raise ApiError(502, "model_error", f"模型服务调用失败，请稍后重试：{e}")
                tasks.create(conn, query_id=qid, session_id=sid, ident=ident, scope=scope, question=body.question,
                             asked_at=asked, prepared=prepared, model_id=model_id, index_build_id=build_id,
                             owner=runner.owner)
                handoff = threading.Event()
                future = pool.submit(runner.run, qid, scope=scope, handoff=handoff)
                try:
                    outcome, result = future.result(timeout=float(eng.budgets["sync_seconds"]))
                except TimeoutError:
                    handoff.set()                       # 当前步骤结束后交给后台任务进程
                    outcome, result = "running", None
                except AuthorizerUnavailable:
                    raise ApiError(503, "authz_unavailable", "授权服务不可用")
                except ModelCallError as e:
                    raise ApiError(502, "model_error", f"模型服务调用失败，请稍后重试：{e}")
                if outcome == "completed":
                    cache.put(conn, ck, result, tenant_id=ident.tenant_id, scope=scope, build_id=build_id, ttl_seconds=ttl,
                              source_query=qid, now=asked)
            if outcome == "completed":
                resp = jsonable_encoder(result.response())
            else:                                    # running（含已交给后台）或 cancelled：其余字段为空（11.2）
                resp = {"session_id": str(sid), "status": "cancelled" if outcome == "cancelled" else "running",
                        "mode": None, "mode_path": [], "time_basis": None, "answer": None, "claims": [],
                        "abstained": None, "notices": []}
            set_attrs(root, {"eaer.cache_hit": hit is not None, "eaer.outcome": outcome})
            if result is not None:
                trace_result(root, result)
        if not body.stream:
            return resp

        def events():
            def ev(name, data):
                return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            if result is None:
                yield ev("mode", {"mode": prepared["mode_path"][-1]})
                if resp["status"] == "running":
                    yield ev("notice", {"text": "超过同步时限，已转为后台执行，请通过会话接口查看进度"})
                yield ev("done", resp)
                return
            for m in result.mode_path:
                yield ev("mode", {"mode": m})
            for n in result.notices:
                yield ev("notice", {"text": n})
            text = result.answer or ""
            for i in range(0, len(text), 40):
                yield ev("delta", {"text": text[i:i + 40]})
            for c in resp["claims"]:
                yield ev("claim", c)
            yield ev("done", resp)
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v1/sessions/{session_id}")
    def session(conn: Conn, ident: Anyone, session_id: str):
        """会话历史、引用记录与进行中查询的进度（11.2）。"""
        sid = _sid(session_id)
        rows = conn.execute(
            "SELECT l.id, l.asked_at, l.question, l.time_basis, l.mode, l.mode_path, l.status, l.answer, l.claims, "
            "l.abstained, t.progress, t.background FROM query_log l LEFT JOIN query_task t ON t.id = l.id "
            "WHERE l.session_id = %s AND l.tenant_id = %s AND l.user_id = %s ORDER BY l.asked_at",
            (sid, ident.tenant_id, ident.user)).fetchall()
        if not rows or not _own_session(conn, ident, sid):
            raise ApiError(404, "not_found", "会话不存在")
        items = []
        for r in rows:
            item = {"query_id": str(r["id"]), **{k: r[k] for k in ("asked_at", "question", "time_basis", "mode",
                                                                     "mode_path", "status", "answer", "abstained")},
                    "background": bool(r["background"]),
                    "claims": [{k: c[k] for k in ("text", "status", "core", "citations")}
                               for c in r["claims"] if c["status"] != "unsupported"]}
            if r["status"] == "running":
                item["progress"] = r["progress"]
            items.append(item)
        return jsonable_encoder({"session_id": str(sid), "items": items})

    @app.post("/v1/sessions/{session_id}/cancel")
    def cancel_session(conn: Conn, ident: Anyone, session_id: str):
        """取消会话中进行中的查询（11.2）：排队中的立即取消；执行中的在当前步骤结束后停止（cancelling）。"""
        from eaer.orchestrator import tasks

        sid = _sid(session_id)
        if not conn.execute("SELECT 1 FROM query_log WHERE session_id = %s", (sid,)).fetchone() \
                or not _own_session(conn, ident, sid):
            raise ApiError(404, "not_found", "会话不存在")
        items = tasks.cancel(conn, sid, tenant_id=ident.tenant_id, user_id=ident.user, now=clock())
        if not items:
            raise ApiError(409, "not_running", "会话中没有进行中的查询")
        return {"session_id": str(sid), "items": items}

    # 强制验证清单：只有读取与追加，不提供删除或修改的路径（AC-11-5）
    @app.get("/v1/config/forced-verify")
    def forced_verify_list(conn: Conn, ident: Anyone):
        return {"builtin": fv.BUILTIN, "rules": fv.tenant_rules(conn, ident.tenant_id)}

    @app.post("/v1/config/forced-verify", status_code=201)
    def forced_verify_add(conn: Conn, ident: Admin, body: ForcedVerifyRule):
        try:
            rule, created = fv.add_rule(conn, ident.tenant_id, body.kind, body.value, ident.user, clock(),
                                        body.description)
        except fv.ForcedVerifyError as e:
            raise ApiError(400, "invalid_request", str(e))
        return JSONResponse(jsonable_encoder(rule), status_code=201 if created else 200)

    @app.get("/v1/evidence/{evidence_id}")
    def evidence(conn: Conn, authz: Authz, evidence_id: str):
        """取证据原文与定位（11.2）。可搜索且有权：返回原文；已归档且有权：只返回归档提示（6.6）；
        其余一律 404，不论不存在、无权、待审核还是已删除（10.2、AC-10-6）。"""
        a, scope = authz
        try:
            allowed = a.check(scope, [evidence_id])[0]
        except AuthorizerUnavailable:
            raise ApiError(503, "authz_unavailable", "授权服务不可用")
        except ScopeExpired:
            raise ApiError(401, "unauthenticated", "授权范围已过期，请重新登录")
        visible = ("SELECT e.id, e.modality, e.content, e.locator, e.parse_stage, v.id AS version_id, v.version_no, "
                   "  v.approved_at, v.status, d.id AS document_id, d.title, d.doc_type "
                   "FROM evidence e JOIN document_version v ON v.id = e.version_id "
                   "JOIN document d ON d.id = v.document_id JOIN acl_policy p ON p.id = v.acl_policy_id "
                   "WHERE e.id = %s AND d.tenant_id = %s AND p.principals && %s AND v.status = %s")
        args = (evidence_id, scope.tenant_id, sorted(scope.principals))
        if allowed:
            r = conn.execute(visible, args + ("searchable",)).fetchone()
            if r is not None:
                val = conn.execute(
                    "SELECT valid_from, valid_from_grade, valid_to, valid_to_grade FROM version_validity "
                    "WHERE version_id = %s AND recorded_at IS NOT NULL AND replaced_at IS NULL",
                    (r["version_id"],)).fetchone()
                return jsonable_encoder({
                    "evidence_id": r["id"], "status": "searchable", "modality": r["modality"],
                    "content": r["content"], "locator": r["locator"], "parse_stage": r["parse_stage"],
                    "document": {"id": r["document_id"], "title": r["title"], "doc_type": r["doc_type"]},
                    "version": {"id": r["version_id"], "version_no": r["version_no"], "approved_at": r["approved_at"]},
                    "validity": val})
        r = conn.execute(visible, args + ("archived",)).fetchone()
        if r is not None:
            return jsonable_encoder({
                "evidence_id": r["id"], "status": "archived", "notice": "该资料已归档",
                "document": {"id": r["document_id"], "title": r["title"]},
                "version": {"id": r["version_id"], "version_no": r["version_no"]}})
        raise ApiError(*EVIDENCE_404)

    return app
