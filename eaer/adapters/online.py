"""线上模型通用客户端（开发文档 11.3 ModelClient 的参考实现、12.2）。

生成、Embedding、重排可分别配置不同的服务（12.2）：
- 生成：OpenAI 兼容的 /chat/completions；结构化输出用 JSON 模式（response_format json_object），
  把 JSON Schema 写进系统提示，再在本地按 schema 校验（多数服务不支持严格的 json_schema）；
- Embedding：OpenAI 兼容的 /embeddings；
- 重排：常见的 /rerank 协议（{model, query, documents} → {results: [{index, relevance_score}]}）。
只用标准库发 HTTP 请求，不依赖任何厂商 SDK（3.2）。

超时与重试（12.2、AC-12-5）：单次调用超时 30 秒（初值），可重试错误（超时、429、5xx）重试 2 次并退避；
最终失败抛 ModelCallError，不返回空答案。每次调用记录 token 数、尝试次数和耗时，交给计量回调（汇总到 query_log.cost）。
错误信息中不出现密钥。

配置来自部署环境变量（13.2“模型服务地址与密钥”，部署时必填）：
  EAER_GEN_BASE_URL / EAER_GEN_API_KEY / EAER_GEN_MODEL     生成服务；EAER_GEN_THINKING=disabled 关闭推理过程
  EAER_GEN_CONTEXT_TOKENS / EAER_GEN_VISION=1                上下文长度、是否支持图像输入
  EAER_GEN_TEMPERATURE                                       采样温度；问答与评测建议 0，便于复现（AC-14-5）
  EAER_EMBED_BASE_URL / EAER_EMBED_API_KEY / EAER_EMBED_MODEL / EAER_EMBED_DIMS
  EAER_EMBED_QUERY_INSTRUCTION                               查询向量化的指令前缀（Qwen3-Embedding 需要）
  EAER_RERANK_BASE_URL / EAER_RERANK_API_KEY / EAER_RERANK_MODEL
  EAER_MODEL_TIMEOUT（秒，默认 30）/ EAER_MODEL_RETRIES（默认 2）
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from eaer.adapters.model import Capabilities, CapabilityNotSupported, Completion, Message, Usage

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}
EMBED_BATCH = 64


class ModelCallError(Exception):
    """模型调用最终失败。调用方据此返回明确错误，不返回空答案。"""


class StructuredOutputError(ModelCallError):
    """要求结构化输出，但返回的不是符合 schema 的 JSON。"""


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    api_key: str
    model: str
    timeout: float = 30.0
    retries: int = 2
    extra: dict = field(default_factory=dict)      # 附加到请求体的参数，如 {"thinking": {"type": "disabled"}}


@dataclass(frozen=True)
class CallRecord:
    kind: str                # generate / embed / rerank
    model: str
    input_tokens: int
    output_tokens: int
    attempts: int
    seconds: float
    ok: bool


Transport = Callable[[str, dict, dict, float], tuple[int, Any]]


def urllib_transport(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, Any]:
    """发送 JSON 请求，返回 (状态码, 解析后的响应)；超时抛 TimeoutError。"""
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(), method="POST",
                                 headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b"null")
        except ValueError:
            payload = None
        return e.code, payload
    except (socket.timeout, TimeoutError) as e:
        raise TimeoutError(str(e)) from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, (socket.timeout, TimeoutError)):
            raise TimeoutError(str(e.reason)) from e
        raise ConnectionError(str(e.reason)) from e


class OnlineModelClient:
    def __init__(self, generation: Endpoint | None, embedding: Endpoint | None = None,
                 rerank: Endpoint | None = None, *, embedding_dims: int = 0, context_tokens: int = 128_000,
                 vision: bool = False, query_instruction: str | None = None,
                 meter: Callable[[CallRecord], None] | None = None,
                 transport: Transport = urllib_transport, sleep: Callable[[float], None] = time.sleep):
        if embedding is not None and embedding_dims <= 0:
            raise ValueError("配置了 Embedding 服务时必须给出向量维度")
        self.gen, self.emb, self.rr = generation, embedding, rerank
        self.embedding_dims, self.context_tokens, self.vision = embedding_dims, context_tokens, vision
        self.query_instruction = query_instruction
        self.meter, self.transport, self.sleep = meter, transport, sleep

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, **kw) -> "OnlineModelClient":
        e = os.environ if env is None else env
        timeout, retries = float(e.get("EAER_MODEL_TIMEOUT", "30")), int(e.get("EAER_MODEL_RETRIES", "2"))

        def ep(prefix: str, extra: dict | None = None) -> Endpoint | None:
            if not e.get(f"{prefix}_BASE_URL"):
                return None
            missing = [k for k in ("API_KEY", "MODEL") if not e.get(f"{prefix}_{k}")]
            if missing:
                raise ValueError(f"{prefix}_BASE_URL 已设置，但缺少 {'、'.join(prefix + '_' + m for m in missing)}")
            return Endpoint(e[f"{prefix}_BASE_URL"].rstrip("/"), e[f"{prefix}_API_KEY"], e[f"{prefix}_MODEL"],
                            timeout, retries, extra or {})

        gen_extra: dict[str, Any] = {}
        if e.get("EAER_GEN_THINKING"):
            gen_extra["thinking"] = {"type": e["EAER_GEN_THINKING"]}
        if e.get("EAER_GEN_TEMPERATURE"):
            gen_extra["temperature"] = float(e["EAER_GEN_TEMPERATURE"])
        return cls(ep("EAER_GEN", gen_extra or None),
                   ep("EAER_EMBED"), ep("EAER_RERANK"),
                   embedding_dims=int(e.get("EAER_EMBED_DIMS", "0")),
                   context_tokens=int(e.get("EAER_GEN_CONTEXT_TOKENS", "128000")),
                   vision=e.get("EAER_GEN_VISION") == "1",
                   query_instruction=e.get("EAER_EMBED_QUERY_INSTRUCTION") or None, **kw)

    # ---------- ModelClient ----------

    def capabilities(self) -> Capabilities:
        return Capabilities(self.gen is not None, self.context_tokens if self.gen else 0,
                            self.vision and self.gen is not None, self.embedding_dims if self.emb else 0)

    def embedder(self):
        """检索与索引使用的 Embedder；名字取 Embedding 模型名，向量缓存按它区分。"""
        from eaer.adapters.model import ModelEmbedder
        if self.emb is None:
            raise CapabilityNotSupported("未配置 Embedding 服务")
        return ModelEmbedder(self, self.emb.model, query_instruction=self.query_instruction)

    def missing_for_fast(self) -> list[str]:
        """Fast 模式需要生成、Embedding、重排三项（12.1）；缺任何一项都无法运行，部署时报错。"""
        return [n for n, ep in (("生成", self.gen), ("Embedding", self.emb), ("重排", self.rr)) if ep is None]

    def generate(self, messages: list[Message], schema: dict | None) -> Completion:
        if self.gen is None:
            raise CapabilityNotSupported("未配置生成服务")
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        body: dict[str, Any] = {"model": self.gen.model, **self.gen.extra}
        if schema is not None:
            instruction = ("只输出一个 JSON 对象，不要输出其他文字。对象必须符合下面的 JSON Schema：\n"
                           + json.dumps(schema, ensure_ascii=False))
            msgs = [{"role": "system", "content": instruction}] + msgs
            body["response_format"] = {"type": "json_object"}
        body["messages"] = msgs
        out, rec = self._post("generate", self.gen, "/chat/completions", body)
        try:
            text = out["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise ModelCallError("生成服务返回格式无法识别")
        if not text.strip():
            raise ModelCallError("生成服务返回了空内容")
        parsed = None
        if schema is not None:
            try:
                parsed = json.loads(text)
            except ValueError:
                raise StructuredOutputError("要求结构化输出，返回的不是 JSON")
            problem = validate(parsed, schema)
            if problem:
                raise StructuredOutputError(f"结构化输出不符合 schema：{problem}")
        return Completion(text, parsed, Usage(rec.input_tokens, rec.output_tokens), out.get("model") or self.gen.model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.emb is None:
            raise CapabilityNotSupported("未配置 Embedding 服务")
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            out, _ = self._post("embed", self.emb, "/embeddings", {"model": self.emb.model, "input": batch, **self.emb.extra})
            data = sorted(out.get("data") or [], key=lambda d: d.get("index", 0))
            if len(data) != len(batch):
                raise ModelCallError(f"Embedding 服务返回 {len(data)} 个向量，应为 {len(batch)} 个")
            for d in data:
                v = [float(x) for x in d["embedding"]]
                if len(v) != self.embedding_dims:
                    raise ModelCallError(f"向量维度 {len(v)} 与配置的 {self.embedding_dims} 不符")
                vectors.append(v)
        return vectors

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if self.rr is None:
            raise CapabilityNotSupported("未配置重排服务")
        if not passages:
            return []
        out, _ = self._post("rerank", self.rr, "/rerank", {"model": self.rr.model, "query": query, "documents": passages,
                                                          "top_n": len(passages), "return_documents": False, **self.rr.extra})
        scores = [float("-inf")] * len(passages)
        for r in out.get("results") or []:
            scores[int(r["index"])] = float(r.get("relevance_score", r.get("score")))
        if any(s == float("-inf") for s in scores):
            raise ModelCallError("重排服务没有给出全部段落的分数")
        return scores

    # ---------- 内部 ----------

    def _post(self, kind: str, ep: Endpoint, path: str, body: dict) -> tuple[dict, CallRecord]:
        headers = {"Authorization": f"Bearer {ep.api_key}"}
        started, attempts, last = time.monotonic(), 0, ""
        for attempt in range(ep.retries + 1):
            attempts += 1
            try:
                status, out = self.transport(ep.base_url + path, headers, body, ep.timeout)
            except TimeoutError:
                status, out, last = None, None, f"超时（{ep.timeout:g} 秒）"
            except ConnectionError as e:
                status, out, last = None, None, f"连接失败：{e}"
            if status is not None and 200 <= status < 300 and isinstance(out, dict):
                usage = out.get("usage") or {}
                rec = CallRecord(kind, ep.model, int(usage.get("prompt_tokens", usage.get("total_tokens", 0)) or 0),
                                 int(usage.get("completion_tokens", 0) or 0), attempts, time.monotonic() - started, True)
                self._record(rec)
                return out, rec
            if status is not None:
                last = f"HTTP {status}：{_error_text(out)}"
                if status not in RETRYABLE:
                    break
            if attempt < ep.retries:
                self.sleep(0.5 * 2 ** attempt)
        self._record(CallRecord(kind, ep.model, 0, 0, attempts, time.monotonic() - started, False))
        raise ModelCallError(f"{kind} 调用失败（{ep.model}，尝试 {attempts} 次）：{last}".replace(ep.api_key, "***"))

    def _record(self, rec: CallRecord) -> None:
        if self.meter is not None:
            self.meter(rec)


def _error_text(out: Any) -> str:
    if isinstance(out, dict):
        err = out.get("error")
        if isinstance(err, dict):
            return str(err.get("message", err))[:200]
        if err:
            return str(err)[:200]
    return "无错误说明"


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def validate(value: Any, schema: dict, path: str = "$") -> str | None:
    """校验 JSON Schema 的常用子集：type、properties、required、items、enum。返回第一个问题，没有则 None。"""
    t = schema.get("type")
    if t is not None:
        py = _TYPES.get(t)
        if py is None:
            return f"{path}：不支持的类型 {t}"
        if isinstance(value, bool) and t in ("integer", "number"):
            return f"{path} 应为 {t}"
        if not isinstance(value, py):
            return f"{path} 应为 {t}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} 取值应在 {schema['enum']} 中"
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                return f"{path} 缺少字段 {k}"
        for k, sub in (schema.get("properties") or {}).items():
            if k in value:
                problem = validate(value[k], sub, f"{path}.{k}")
                if problem:
                    return problem
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            problem = validate(item, schema["items"], f"{path}[{i}]")
            if problem:
                return problem
    return None
