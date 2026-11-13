"""语义检查（开发文档 9.3）：通过确定性检查的结论交给模型判断，给定结论和所引证据原文，
输出 supported、unsupported 或 unverified 以及置信度；置信度低于阈值的一律视为 unverified。
阈值按所用模型标定（tenant_config.thresholds.semantic_min_confidence，初值 0.7），宁可多标“未验证”。
模型只判断“证据是否真正支持结论”，数值、日期、有效期、权限由代码做的确定性检查负责（9.2）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from eaer.adapters.model import Message, ModelClient

DEFAULT_MIN_CONFIDENCE = 0.7
SCHEMA = {
    "type": "object", "required": ["verdict", "confidence", "supporting_ids"],
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "unsupported", "unverified"]},
        "confidence": {"type": "number"},
        "supporting_ids": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
}
SYSTEM = """你是严格的事实核查员。判断给定的结论是否被给定的证据完整支持。
- supported：结论的每一部分（主体、数值、单位、日期、适用范围）都能在证据中找到依据；
- unsupported：证据与结论矛盾，或证据根本没有提到结论中的关键内容；
- unverified：证据相关但不足以确认，或你拿不准。
只依据给出的证据，不使用常识补全。supporting_ids 只列真正支持结论的证据 id。
confidence 是你对判断的把握，0 到 1。"""


@dataclass
class Verdict:
    status: str
    confidence: float
    supporting_ids: list[str]
    reason: str

    def to_json(self) -> dict:
        return asdict(self)


class SemanticChecker:
    def __init__(self, model: ModelClient, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        self.model, self.min_confidence = model, min_confidence

    def judge(self, claim: str, evidence: list[tuple[str, str]], on: date) -> Verdict:
        """evidence：[(证据 id, 带版本信息的证据文本)]。"""
        if not evidence:
            return Verdict("unsupported", 1.0, [], "没有证据")
        blocks = "\n\n".join(f"[{i}] {t}" for i, t in evidence)
        out = self.model.generate([Message("system", SYSTEM),
                                   Message("user", f"目标日期：{on.isoformat()}\n结论：{claim}\n\n证据：\n{blocks}")], SCHEMA)
        p = out.parsed or {}
        given = {i for i, _ in evidence}
        ids = [i for i in dict.fromkeys(p.get("supporting_ids", [])) if i in given]
        status, conf = p.get("verdict", "unverified"), float(p.get("confidence", 0.0))
        reason = p.get("reason", "")
        if conf < self.min_confidence and status != "unverified":
            status, reason = "unverified", f"置信度 {conf:.2f} 低于阈值 {self.min_confidence}；{reason}"
        if status == "supported" and not ids:
            status, reason = "unverified", f"模型判为支持但没有指出依据；{reason}"
        return Verdict(status, conf, ids, reason)
