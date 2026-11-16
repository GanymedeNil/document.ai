"""向量化接口。生产环境由模型接口（开发文档 11.3 的 ModelClient.embed）提供。

HashingEmbedder 是测试替身：把字和相邻二字散列到固定维度，只用来让检索流程跑通，
不代表任何真实模型的效果；评测报告里的向量检索结果不能拿它来说明产品水平。
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

from eaer.retrieval.index import tokens


class Embedder(Protocol):
    name: str
    dims: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dims: int = 256):
        self.dims = dims
        self.name = f"hashing-test-double-{dims}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dims
            for tok in tokens(t):
                h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")
                v[h % self.dims] += 1.0 if (h >> 63) == 0 else -1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out
