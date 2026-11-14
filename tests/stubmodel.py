"""测试用的模型服务替身：OpenAI 兼容的 /chat/completions、/embeddings，以及常见协议的 /rerank。

供需要真实进程的集成测试使用：接口进程、后台任务进程各自用 OnlineModelClient（EAER_GEN_*、EAER_EMBED_*、
EAER_RERANK_*）连它，走的是与线上模型完全相同的配置与调用代码，但不调用任何线上服务，也不需要密钥。
- Embedding：HashingEmbedder 的向量（测试替身，得出的召回数字不能当产品结论）；
- 重排：问题与段落的字二元组重合比例；
- 生成：按系统提示里的 JSON Schema 区分请求（与 tests/test_engine.py 的 Scripted 相同的分法）——回答草稿选与问题
  重合最多的那条证据、原文照抄为结论，重合太少就拒答；语义检查判 supported；时间解析留空；Deep 的规划给出合法的
  最小结构。
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from eaer.retrieval.embed import HashingEmbedder

_EVIDENCE = re.compile(r"\[(ev_[0-9a-f]+)\][^\n]*\n([^\n]*)")
_ARTICLE = re.compile(r"^第[一二三四五六七八九十百]+条\s*")


def bigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def overlap(query: str, passage: str) -> float:
    q = bigrams(query)
    return len(q & bigrams(passage)) / len(q) if q else 0.0


def _question(prompt: str) -> str:
    m = re.search(r"问题：(.*)", prompt)
    return m.group(1) if m else ""


def _draft(prompt: str) -> dict:
    q = _question(prompt)
    evidence = _EVIDENCE.findall(prompt)
    best = max(evidence, key=lambda e: overlap(q, e[1]), default=None)
    if best is None or overlap(q, best[1]) < 0.3:
        return {"abstain": True, "missing": "没有与问题相关的资料", "claims": []}
    return {"abstain": False, "claims": [{"text": _ARTICLE.sub("", best[1]).strip(), "evidence_ids": [best[0]],
                                          "core": True}]}


def respond(schema: dict | None, prompt: str):
    if schema is None:
        d = _draft(prompt)
        return "无法回答。" if d["abstain"] else f"{d['claims'][0]['text']}[{d['claims'][0]['evidence_ids'][0]}]"
    props = schema.get("properties", {})
    q = _question(prompt) or "取值"
    if "verdict" in props:
        return {"verdict": "supported", "confidence": 0.9, "supporting_ids": [i for i, _ in _EVIDENCE.findall(prompt)]}
    if "as_of_valid" in props:
        return {"as_of_valid": ""}
    if "claims" in props:
        return _draft(prompt)
    if "objects" in props:
        return {"field": q, "objects": []}
    if "first_hop" in props:
        return {"first_hop": q}
    if "phenomenon" in props:
        return {"phenomenon": q, "change_query": q}
    if "closed" in props:
        return {"closed": True}
    if "queries" in props:
        return {"queries": []}
    return {}


class StubModelServer:
    """在 127.0.0.1 的随机端口上启动；base_url 形如 http://127.0.0.1:PORT/v1。calls 按种类计数。"""

    def __init__(self, dims: int = 64):
        self.embedder = HashingEmbedder(dims)
        self.dims = dims
        self.calls = {"generate": 0, "embed": 0, "rerank": 0}
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):                   # 安静
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                try:
                    out = stub.handle(self.path, body)
                except KeyError:
                    self.send_response(404)
                    self.end_headers()
                    return
                raw = json.dumps(out, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def _count(self, kind: str) -> None:
        with self._lock:
            self.calls[kind] += 1

    def handle(self, path: str, body: dict) -> dict:
        if path.endswith("/embeddings"):
            self._count("embed")
            vecs = self.embedder.embed(list(body["input"]))
            return {"data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)], "model": body.get("model"),
                    "usage": {"prompt_tokens": sum(len(t) for t in body["input"])}}
        if path.endswith("/rerank"):
            self._count("rerank")
            return {"results": [{"index": i, "relevance_score": overlap(body["query"], d)}
                                for i, d in enumerate(body["documents"])]}
        if path.endswith("/chat/completions"):
            self._count("generate")
            msgs = body["messages"]
            schema = None
            if msgs and msgs[0]["role"] == "system" and "JSON Schema：\n" in msgs[0]["content"]:
                schema = json.loads(msgs[0]["content"].split("JSON Schema：\n", 1)[1])
            prompt = "\n".join(m["content"] for m in msgs)
            out = respond(schema, prompt)
            content = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            return {"model": body.get("model"), "choices": [{"index": 0, "message": {"role": "assistant",
                                                                                     "content": content}}],
                    "usage": {"prompt_tokens": len(prompt) // 2, "completion_tokens": len(content) // 2}}
        raise KeyError(path)

    def env(self) -> dict[str, str]:
        """连这个替身所需的环境变量（与线上模型同一套配置项）。"""
        return {"EAER_GEN_BASE_URL": self.base_url, "EAER_GEN_API_KEY": "stub", "EAER_GEN_MODEL": "stub-gen",
                "EAER_EMBED_BASE_URL": self.base_url, "EAER_EMBED_API_KEY": "stub", "EAER_EMBED_MODEL": "stub-embed",
                "EAER_EMBED_DIMS": str(self.dims),
                "EAER_RERANK_BASE_URL": self.base_url, "EAER_RERANK_API_KEY": "stub", "EAER_RERANK_MODEL": "stub-rerank",
                "EAER_MODEL_TIMEOUT": "10", "EAER_MODEL_RETRIES": "0"}

    def __enter__(self) -> "StubModelServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
