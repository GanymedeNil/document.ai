"""对外接口：上传、审核、归档、取消归档（开发文档 11.1、11.2）。

注意：本模块不能使用 from __future__ import annotations，FastAPI 需要在运行时解析依赖注解。

身份校验走授权接口；这里的参考实现是配置好的令牌表。客户现场换成 SSO 时只换 Authenticator。
错误一律返回 {"code", "message", "request_id"}；无权访问与不存在不区分，都返回 404。
"""

import uuid
from datetime import datetime
from typing import Annotated, Any, Callable, Iterator, Literal, Protocol

import psycopg
from fastapi import Depends, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from eaer.domain.identity import Identity
from eaer.ingestion.parsers import UnsupportedFormat
from eaer.ingestion.pipeline import Ingestion
from eaer.lifecycle.service import InvalidState, Lifecycle, utcnow
from eaer.review.service import Review
from eaer.storage import FileStore

MAX_DECISIONS = 1000


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


class Decision(BaseModel):
    task_id: str
    decision: Literal["approve", "delete"]
    corrections: dict[str, Any] | None = None


class Decisions(BaseModel):
    decisions: list[Decision] = Field(min_length=1, max_length=MAX_DECISIONS)


def create_app(database_url: str, store: FileStore, auth: Authenticator,
               clock: Callable[[], datetime] = utcnow) -> FastAPI:
    app = FastAPI(title="EAER", version="0.1.0")

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
            ident.tenant_id, [d.model_dump() for d in body.decisions], ident.user)
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

    return app
