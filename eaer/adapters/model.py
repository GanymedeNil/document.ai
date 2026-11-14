"""模型接口（开发文档 11.3、12.1）。

所有模型调用都经 ModelClient；业务模块不导入模型厂商的 SDK（3.2）。参考实现是“调用线上模型的通用客户端，
生成、Embedding、重排可分别配置”，需要用户提供接入配置，尚未实现（见 CLAUDE.md 下一步第 5 项）。

DeterministicTestModel 是测试替身：结果可复现、不联网，只用来跑通流程与一致性套件，不代表任何模型的效果。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from eaer.retrieval.embed import HashingEmbedder
from eaer.retrieval.index import tokens


class CapabilityNotSupported(Exception):
    """调用了模型不具备的能力（如不支持结构化输出却传了 schema）。调用方据此降级（AC-12-1）。"""


@dataclass(frozen=True)
class Image:
    """随消息发给模型的图片（12.1 图像输入能力）。mime 为 image/png、image/jpeg、image/gif、image/webp。"""
    mime: str
    data: bytes


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
    images: tuple[Image, ...] = ()      # 只有声明了 vision 能力的模型接受；否则抛 CapabilityNotSupported


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Completion:
    text: str
    parsed: dict[str, Any] | None      # 传了 schema 时为解析后的对象
    usage: Usage
    model: str


@dataclass(frozen=True)
class Capabilities:
    structured_output: bool
    context_tokens: int
    vision: bool
    embedding_dims: int


class ModelClient(Protocol):
    def generate(self, messages: list[Message], schema: dict | None) -> Completion: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def rerank(self, query: str, passages: list[str]) -> list[float]: ...
    def capabilities(self) -> Capabilities: ...


class ModelEmbedder:
    """把 ModelClient.embed 包装成检索使用的 Embedder（retrieval.embed.Embedder）。

    query_instruction：查询向量化时加的指令前缀（Qwen3-Embedding 等模型要求查询带指令、文档不带）。
    前缀只影响查询，文档向量与缓存（evidence_embedding，按模型名区分）不受影响。
    """

    def __init__(self, client: ModelClient, name: str, *, query_instruction: str | None = None):
        self.client, self.name, self.query_instruction = client, name, query_instruction
        self.dims = client.capabilities().embedding_dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts)

    def embed_query(self, question: str) -> list[float]:
        text = f"Instruct: {self.query_instruction}\nQuery:{question}" if self.query_instruction else question
        return self.client.embed([text])[0]


_DEFAULTS = {"string": "", "integer": 0, "number": 0, "boolean": False, "array": [], "object": {}}


@dataclass
class DeterministicTestModel:
    """测试替身。structured_output=False 用来模拟不支持结构化输出的模型。"""
    structured_output: bool = True
    dims: int = 64
    name: str = "deterministic-test-double"
    _emb: HashingEmbedder = field(init=False, repr=False)

    def __post_init__(self):
        self._emb = HashingEmbedder(self.dims)

    def capabilities(self) -> Capabilities:
        return Capabilities(self.structured_output, 32_000, False, self.dims)

    def generate(self, messages: list[Message], schema: dict | None) -> Completion:
        if schema is not None and not self.structured_output:
            raise CapabilityNotSupported("该模型不支持结构化输出")
        if any(m.images for m in messages):
            raise CapabilityNotSupported("该模型不支持图像输入")
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        text = f"[测试替身] {last[:200]}"
        parsed = None
        if schema is not None:
            props = schema.get("properties", {})
            parsed = {k: _DEFAULTS.get(v.get("type"), None) for k, v in props.items()}
            text = str(parsed)
        n_in = sum(len(tokens(m.content)) for m in messages)
        return Completion(text, parsed, Usage(n_in, len(tokens(text))), self.name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed(texts)

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        q = set(tokens(query))
        return [len(q & set(tokens(p))) / (len(q) or 1) for p in passages]
