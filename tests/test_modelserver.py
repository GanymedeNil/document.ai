"""本地模型服务 modelserver 的接口（离线：用假模型；真实模型经 tests/test_model_live.py 验证）。

覆盖：/v1/embeddings 与 /v1/rerank 的请求与响应格式、top_n、排序、鉴权、错误格式；
EAER 的线上客户端 OnlineModelClient 与本服务的协议对得上；查询指令前缀只加在查询上；
重排提示按 Qwen3-Reranker 官方模板拼接。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eaer.adapters.model import ModelEmbedder
from eaer.adapters.online import Endpoint, OnlineModelClient
from modelserver.app import create_app
from modelserver.backends import RERANK_PREFIX, RERANK_SUFFIX, rerank_pair


class FakeEmbedding:
    dims = 3

    def __init__(self):
        self.seen = []

    def encode(self, texts):
        self.seen += texts
        return type("E", (), {"vectors": [[float(len(t)), 0.0, 1.0] for t in texts], "tokens": sum(len(t) for t in texts)})


class FakeReranking:
    def score(self, query, documents, instruction=None):
        return [float(sum(ch in d for ch in query)) / len(query) for d in documents], 7


KEY = "local-secret"


@pytest.fixture
def server():
    emb = FakeEmbedding()
    app = create_app(emb, FakeReranking(), embed_name="Qwen3-Embedding-0.6B", rerank_name="Qwen3-Reranker-0.6B",
                     api_key=KEY)
    return TestClient(app), emb


H = {"Authorization": f"Bearer {KEY}"}


def test_embeddings_contract(server):
    c, _ = server
    r = c.post("/v1/embeddings", headers=H, json={"model": "Qwen3-Embedding-0.6B", "input": ["一", "二三"]})
    b = r.json()
    assert r.status_code == 200 and [d["index"] for d in b["data"]] == [0, 1]
    assert b["data"][1]["embedding"] == [2.0, 0.0, 1.0] and b["usage"]["prompt_tokens"] == 3
    assert c.post("/v1/embeddings", headers=H, json={"input": "单条"}).json()["data"][0]["index"] == 0


def test_rerank_contract(server):
    c, _ = server
    docs = ["伙食补贴80元", "一线城市住宿上限650元", "采购审批"]
    b = c.post("/v1/rerank", headers=H, json={"query": "住宿上限", "documents": docs, "top_n": 2}).json()
    assert [x["index"] for x in b["results"]] == [1, 0] and b["results"][0]["relevance_score"] == 1.0
    assert "document" not in b["results"][0]
    assert c.post("/v1/rerank", headers=H, json={"query": "q", "documents": []}).json()["results"] == []


@pytest.mark.parametrize("headers, status", [({}, 401), ({"Authorization": "Bearer wrong"}, 401)])
def test_auth_and_error_format(server, headers, status):
    c, _ = server
    r = c.post("/v1/embeddings", headers=headers, json={"input": ["x"]})
    assert r.status_code == status and r.json()["error"]["type"] == "authentication_error"
    assert c.get("/health").status_code == 200                    # 健康检查不需要密钥


@pytest.mark.parametrize("body, status", [({"input": []}, 400), ({"model": "other", "input": ["x"]}, 404), ({}, 400)])
def test_bad_embedding_requests(server, body, status):
    c, _ = server
    r = c.post("/v1/embeddings", headers=H, json=body)
    assert r.status_code == status and "message" in r.json()["error"]


def test_eaer_client_speaks_the_same_protocol(server):
    """OnlineModelClient 经测试客户端直接打到本服务：Embedding、查询前缀、重排都对得上。"""
    c, emb = server

    def transport(url, headers, body, timeout):
        r = c.post(url.removeprefix("http://modelserver"), headers=headers, json=body)
        return r.status_code, r.json()

    ep = lambda m: Endpoint("http://modelserver/v1", KEY, m)  # noqa: E731
    client = OnlineModelClient(None, ep("Qwen3-Embedding-0.6B"), ep("Qwen3-Reranker-0.6B"), embedding_dims=3,
                               query_instruction="检索制度条款", transport=transport)
    assert client.embed(["甲", "乙丙"]) == [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]
    assert client.rerank("住宿上限", ["伙食", "住宿上限650元"]) == [0.0, 1.0]
    e = client.embedder()
    assert isinstance(e, ModelEmbedder) and e.name == "Qwen3-Embedding-0.6B" and e.dims == 3
    e.embed_query("住宿上限是多少")
    assert emb.seen[-1] == "Instruct: 检索制度条款\nQuery:住宿上限是多少"
    e.embed(["第二条　住宿上限650元"])
    assert emb.seen[-1] == "第二条　住宿上限650元"                 # 文档不加前缀


def test_reranker_prompt_follows_the_official_template():
    assert rerank_pair("I", "Q", "D") == "<Instruct>: I\n<Query>: Q\n<Document>: D"
    assert RERANK_PREFIX.startswith("<|im_start|>system\nJudge whether the Document meets the requirements")
    assert RERANK_SUFFIX == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ---------- Qwen3-VL-Embedding / Qwen3-VL-Reranker 的输入格式（按官方仓库，不加载模型） ----------

def test_vl_embedding_takes_the_instruction_from_the_eaer_query_prefix():
    from modelserver.backends import split_instruction, vl_embedding_messages

    assert split_instruction("Instruct: Given a question, retrieve passages\nQuery:一线城市住宿上限") == (
        "Given a question, retrieve passages", "一线城市住宿上限")
    assert split_instruction("第二条 一线城市住宿上限为每晚650元。") == (None, "第二条 一线城市住宿上限为每晚650元。")
    doc = vl_embedding_messages("第二条 一线城市住宿上限为每晚650元。")
    assert doc[0] == {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]}
    assert doc[1]["content"] == [{"type": "text", "text": "第二条 一线城市住宿上限为每晚650元。"}]
    q = vl_embedding_messages("一线城市住宿上限", instruction="Given a question, retrieve passages")
    assert q[0]["content"][0]["text"] == "Given a question, retrieve passages."          # 指令补句末标点
    img = object()
    assert [c["type"] for c in vl_embedding_messages("图注", [img])[1]["content"]] == ["image", "text"]


def test_vl_reranker_prompt_follows_the_official_format():
    from modelserver.backends import VL_RERANK_SYSTEM, vl_rerank_messages

    m = vl_rerank_messages("Retrieve passages.", "一线城市住宿上限是多少", "第二条 一线城市住宿上限为每晚650元。")
    assert m[0]["content"][0]["text"] == VL_RERANK_SYSTEM and "\"yes\" or \"no\"" in VL_RERANK_SYSTEM
    assert [c["text"] for c in m[1]["content"]] == ["<Instruct>: Retrieve passages.", "<Query>:", "一线城市住宿上限是多少",
                                                    "\n<Document>:", "第二条 一线城市住宿上限为每晚650元。"]


def test_known_revisions_are_pinned_full_commits():
    from modelserver.__main__ import DEFAULTS, KNOWN_REVISIONS

    assert all(len(r) == 40 for r in KNOWN_REVISIONS.values())
    assert KNOWN_REVISIONS[DEFAULTS["MODELSERVER_EMBED_MODEL"]] == DEFAULTS["MODELSERVER_EMBED_REVISION"]
    assert {"Qwen/Qwen3-VL-Embedding-2B", "Qwen/Qwen3-VL-Reranker-2B"} <= set(KNOWN_REVISIONS)
