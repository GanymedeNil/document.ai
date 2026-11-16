"""证据 id 规则（开发文档 4.3）。

id 由版本 id、模态、规范化后的定位和内容哈希计算，同一版本重新解析时，
内容不变的证据得到相同的 id，历史引用不会失效。
"""
from __future__ import annotations

import hashlib
import json
import unicodedata


def content_hash(content: str) -> str:
    """内容哈希：先做 Unicode 规范化并压缩空白，避免解析器的空白差异导致 id 变化。"""
    text = " ".join(unicodedata.normalize("NFKC", content).split())
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_locator(locator: dict) -> str:
    return json.dumps(locator, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def evidence_id(version_id: str, modality: str, locator: dict, content_hash_value: str) -> str:
    raw = f"{version_id}|{modality}|{normalize_locator(locator)}|{content_hash_value}"
    return "ev_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
