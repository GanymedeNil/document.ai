"""HTTP 接口：OpenAI 兼容的 /v1/embeddings 与常见的 /v1/rerank。错误用 OpenAI 风格 {"error": {"message", "type"}}。"""
from __future__ import annotations

import hmac
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MAX_INPUTS = 512


class Embedding(Protocol):
    dims: int
    def encode(self, texts: list[str]) -> Any: ...       # 返回带 vectors、tokens 的对象


class Reranking(Protocol):
    def score(self, query: str, documents: list[str], instruction: str | None = None) -> tuple[list[float], int]: ...


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    encoding_format: str | None = None


class RerankRequest(BaseModel):
    model: str | None = None
    query: str
    documents: list[str] = Field(max_length=MAX_INPUTS)
    top_n: int | None = Field(default=None, ge=1)
    return_documents: bool = False
    instruction: str | None = None


class _MemoryFull:
    pass


def _err(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def create_app(embedding: Embedding | None, reranking: Reranking | None, *, embed_name: str = "",
               rerank_name: str = "", api_key: str | None = None, guard=None) -> FastAPI:
    """guard：modelserver.memguard.MemoryGuard。内存不足时这一次请求返回 503（进程不死），请求后按需释放缓存。"""
    from modelserver.memguard import is_out_of_memory

    app = FastAPI(title="EAER 本地模型服务")

    def run(fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001 — 只把内存不足转成 503，其他照常抛出
            if not is_out_of_memory(e):
                raise
            if guard is not None:
                guard.relieve()
            return _MemoryFull()
        finally:
            if guard is not None:
                guard.relieve()

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if api_key and request.url.path.startswith("/v1/"):
            given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(given, api_key):
                return _err(401, "密钥无效", "authentication_error")
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def _bad(request: Request, e: RequestValidationError):
        first = e.errors()[0] if e.errors() else {}
        return _err(400, f"{'.'.join(str(x) for x in first.get('loc', []))}：{first.get('msg', '参数错误')}")

    @app.get("/health")
    def health():
        return {"status": "ok", "embedding": bool(embedding), "rerank": bool(reranking),
                **({"memory": guard.stats()} if guard is not None else {})}

    @app.get("/v1/models")
    def models():
        data = [{"id": n, "object": "model", "owned_by": "local", **extra}
                for n, ok, extra in ((embed_name, embedding, {"dims": getattr(embedding, "dims", None)}),
                                     (rerank_name, reranking, {})) if ok]
        return {"object": "list", "data": data}

    @app.post("/v1/embeddings")
    def embeddings(body: EmbeddingRequest):
        if embedding is None:
            return _err(404, "本服务没有加载 Embedding 模型", "not_found_error")
        texts = [body.input] if isinstance(body.input, str) else body.input
        if not texts or len(texts) > MAX_INPUTS:
            return _err(400, f"input 应有 1–{MAX_INPUTS} 条")
        if body.model and body.model != embed_name:
            return _err(404, f"没有模型 {body.model}", "not_found_error")
        out = run(embedding.encode, texts)
        if isinstance(out, _MemoryFull):
            return _err(503, "模型服务内存不足（超过 MODELSERVER_MEMORY_LIMIT），请减小批量或稍后重试", "server_error")
        return {"object": "list", "model": embed_name,
                "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(out.vectors)],
                "usage": {"prompt_tokens": out.tokens, "total_tokens": out.tokens}}

    @app.post("/v1/rerank")
    def rerank(body: RerankRequest):
        if reranking is None:
            return _err(404, "本服务没有加载重排模型", "not_found_error")
        if body.model and body.model != rerank_name:
            return _err(404, f"没有模型 {body.model}", "not_found_error")
        if not body.documents:
            return {"model": rerank_name, "results": [], "usage": {"total_tokens": 0}}
        out = run(reranking.score, body.query, body.documents, body.instruction)
        if isinstance(out, _MemoryFull):
            return _err(503, "模型服务内存不足（超过 MODELSERVER_MEMORY_LIMIT），请减少段落数或稍后重试", "server_error")
        scores, tokens = out
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: body.top_n or len(scores)]
        results = [{"index": i, "relevance_score": scores[i],
                    **({"document": {"text": body.documents[i]}} if body.return_documents else {})} for i in order]
        return {"model": rerank_name, "results": results, "usage": {"total_tokens": tokens}}

    return app
