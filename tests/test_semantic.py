"""语义检查（开发文档 9.3）：置信度低于阈值一律 unverified；只承认给定证据中的 id；判为支持却没指出依据时不算支持。"""
from __future__ import annotations

from datetime import date

import pytest

from eaer.adapters.model import Capabilities, Completion, Usage
from eaer.verifier.semantic import SemanticChecker

EV = [("ev_a", "一线城市住宿上限为每晚650元"), ("ev_b", "出差伙食补贴每天80元")]


class Fixed:
    def __init__(self, parsed):
        self.parsed, self.prompts = parsed, []

    def capabilities(self):
        return Capabilities(True, 32000, False, 0)

    def generate(self, messages, schema):
        self.prompts.append("\n".join(m.content for m in messages))
        assert set(schema["required"]) == {"verdict", "confidence", "supporting_ids"}
        return Completion("{}", self.parsed, Usage(1, 1), "fixed")


@pytest.mark.parametrize("parsed, status, ids", [
    ({"verdict": "supported", "confidence": 0.9, "supporting_ids": ["ev_a"]}, "supported", ["ev_a"]),
    ({"verdict": "supported", "confidence": 0.5, "supporting_ids": ["ev_a"]}, "unverified", ["ev_a"]),     # 低于阈值
    ({"verdict": "unsupported", "confidence": 0.5, "supporting_ids": []}, "unverified", []),              # 拿不准
    ({"verdict": "unsupported", "confidence": 0.95, "supporting_ids": []}, "unsupported", []),
    ({"verdict": "supported", "confidence": 0.95, "supporting_ids": ["ev_zzz"]}, "unverified", []),        # 编造的 id
    ({"verdict": "supported", "confidence": 0.95, "supporting_ids": []}, "unverified", []),               # 没指出依据
    ({"verdict": "unverified", "confidence": 0.99, "supporting_ids": ["ev_b"]}, "unverified", ["ev_b"]),
])
def test_verdicts(parsed, status, ids):
    v = SemanticChecker(Fixed(parsed), 0.7).judge("一线城市住宿上限为每晚650元。", EV, date(2026, 6, 1))
    assert (v.status, v.supporting_ids) == (status, ids)


def test_no_evidence_is_unsupported_without_calling_the_model():
    m = Fixed({})
    assert SemanticChecker(m).judge("任意结论", [], date(2026, 6, 1)).status == "unsupported" and m.prompts == []


def test_prompt_contains_claim_evidence_and_target_date():
    m = Fixed({"verdict": "supported", "confidence": 0.9, "supporting_ids": ["ev_a"]})
    SemanticChecker(m).judge("一线城市住宿上限为每晚650元。", EV, date(2026, 6, 1))
    assert "目标日期：2026-06-01" in m.prompts[0] and "[ev_a] 一线城市" in m.prompts[0]
