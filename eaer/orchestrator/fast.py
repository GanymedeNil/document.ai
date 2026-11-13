"""Fast 模式（开发文档 8.3、8.6）：检索 → 生成 → 引用检查（9.2 的确定性检查）→ 判断是否需要验证 → 组装。

- 生成：模型只看检索到的证据原文和必要的版本信息，用结构化输出给出结论列表，每条结论列出所依据的证据 id，
  并标出哪条是回答问题的核心结论；授权范围不进入模型上下文（3.2 第 4 条）。
- 引用检查：每条结论过确定性检查（引用核对、authorized、locator_exists、value_match、time_valid）。
- 组装（8.6）：通过的结论作为事实陈述并附引用；unverified 的保留并标“未验证”；unsupported 的删除，
  若是核心结论则整个回答改为拒答并说明缺少什么。最终回答文本由保留下来的结论拼成，
  模型自由发挥的文字不直接给用户，因此 unsupported 的内容不会出现（AC-8-5）。
- 是否需要验证：按强制验证清单在检索后再判断一次（8.1）；命中时由 Router 转入 Verified（Router 尚未实现，
  这里在结果里给出命中的规则）。
拒答文本不透露任何用户无权查看的文档的存在（8.6、10.2）：检索结果本身已按授权范围过滤，拒答只说没有找到依据。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

import psycopg

from eaer.adapters.model import Message, ModelClient
from eaer.domain.identity import Scope
from eaer.orchestrator.forced_verify import Cited, must_verify
from eaer.retrieval.service import Hit, Retriever
from eaer.lifecycle.asof import end_of_day
from eaer.verifier.deterministic import DeterministicChecker, Target

ABSTAIN = "没有找到能够回答这个问题的资料。"
SCHEMA = {
    "type": "object", "required": ["abstain", "claims"],
    "properties": {
        "abstain": {"type": "boolean"},
        "missing": {"type": "string"},
        "claims": {"type": "array", "items": {
            "type": "object", "required": ["text", "evidence_ids", "core"],
            "properties": {"text": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}},
                           "core": {"type": "boolean"}}}},
    },
}
SYSTEM = """你是企业制度问答助手，只能依据下面给出的证据回答，不得使用证据以外的知识。
要求：
1. 把回答拆成若干条结论，每条结论是一句完整的陈述，并列出它所依据的证据 id（只能用给出的 id）。
2. 结论中出现的数值、日期、单位必须与证据原文一致，不要换算或改写数值。
3. 把直接回答问题的那条结论标为 core=true，其余补充说明标为 core=false。
4. 注意每条证据所属版本的有效期：只用在目标日期有效的版本回答；有效期标为“推断”或“未知”的，照常引用即可。
5. 证据不足以回答时，abstain=true，claims 为空，并在 missing 中说明缺少什么；不要猜测，不要提及证据之外的文件。"""


@dataclass
class Citation:
    evidence_id: str
    document_id: str
    version_id: str
    version_no: int | None
    locator: dict
    valid_grade: str | None


@dataclass
class ClaimOut:
    text: str
    status: str                        # supported / unverified / unsupported
    core: bool
    citations: list[Citation]
    checks: dict[str, dict]


@dataclass
class FastAnswer:
    answer: str
    claims: list[ClaimOut]
    abstained: bool
    notices: list[str]
    forced_verify: list[str]           # 命中的强制验证规则；非空时 Router 应转入 Verified
    time_basis: dict
    mode: str = "fast"
    mode_path: list[str] = field(default_factory=lambda: ["fast"])
    model: str = ""
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0


def _grade_text(g: str | None) -> str:
    return {"explicit": "正文写明", "inferred": "推断", "unknown": "未知", None: "不适用"}[g]


def _evidence_block(h: Hit, meta: dict) -> str:
    vf, vt = meta.get("valid_from"), meta.get("valid_to")
    span = f"生效 {vf or '未知'}" + (f"，作废 {vt}" if vt else "")
    return (f"[{h.evidence_id}] 文档《{meta['title']}》第 {meta['version_no']} 版；{span}；有效期来源：{_grade_text(h.valid_grade)}\n"
            f"{h.content}")


class FastPipeline:
    def __init__(self, conn: psycopg.Connection, retriever: Retriever, model: ModelClient, authorizer, *, store=None):
        self.conn, self.retriever, self.model, self.authorizer, self.store = conn, retriever, model, authorizer, store

    def answer(self, scope: Scope, question: str, *, today: date, as_of_valid: date | None = None,
               as_of_system: date | None = None, time_params: bool | None = None) -> FastAnswer:
        started = time.monotonic()
        on = as_of_valid or as_of_system or today
        target = Target(on, end_of_day(as_of_system) if as_of_system else None)
        basis = {"as_of_valid": as_of_valid, "as_of_system": as_of_system,
                 "source": "parameter" if (as_of_valid or as_of_system) else "today"}
        notices = []
        if as_of_valid:
            notices.append(f"按 {as_of_valid.year} 年 {as_of_valid.month} 月 {as_of_valid.day} 日生效的版本回答")
        if as_of_system:
            notices.append(f"按截至 {as_of_system.year} 年 {as_of_system.month} 月 {as_of_system.day} 日的系统记录回答")
        hits = self.retriever.search(scope, question, today=today, as_of_valid=as_of_valid, as_of_system=as_of_system)
        meta = self._meta(hits, target.system_t)
        timed = bool(time_params if time_params is not None else (as_of_valid or as_of_system))
        if not hits:
            return FastAnswer(ABSTAIN, [], True, notices,
                              must_verify(self.conn, scope.tenant_id, question=question, time_params=timed),
                              basis, seconds=time.monotonic() - started)

        evidence = "\n\n".join(_evidence_block(h, meta[h.version_id]) for h in hits)
        user = f"目标日期：{on.isoformat()}\n问题：{question}\n\n证据：\n{evidence}"
        out = self.model.generate([Message("system", SYSTEM), Message("user", user)], SCHEMA)
        draft = out.parsed or {}
        by_id = {h.evidence_id: h for h in hits}
        checker = DeterministicChecker(self.conn, self.authorizer, scope, self.store)
        claims: list[ClaimOut] = []
        for c in draft.get("claims", []) if not draft.get("abstain") else []:
            ids = list(dict.fromkeys(c["evidence_ids"]))
            r = checker.check(c["text"], ids, retrieved=set(by_id), target=target)
            status = "supported" if r.outcome == "pass" else r.outcome
            cites = [Citation(i, by_id[i].document_id, by_id[i].version_id, meta[by_id[i].version_id]["version_no"],
                              by_id[i].locator, by_id[i].valid_grade) for i in r.evidence_ids if i in by_id]
            claims.append(ClaimOut(c["text"], status, bool(c.get("core")), cites,
                                   {k: {"outcome": v.outcome, "reason": v.reason} for k, v in r.checks.items()}))

        if claims and not any(c.core for c in claims):
            claims[0].core = True                      # 模型没有标核心结论时，第一条视为核心
        kept = [c for c in claims if c.status != "unsupported"]
        core_failed = any(c.core and c.status == "unsupported" for c in claims)
        abstained = bool(draft.get("abstain")) or not kept or core_failed
        if abstained:
            reason = draft.get("missing") if draft.get("abstain") else None
            text = ABSTAIN + (f"缺少：{reason}" if reason else "")
            if core_failed:
                text = ABSTAIN + "核心结论未能在证据中得到核实。"
            final = []
        else:
            final = kept
            text = "".join(c.text + ("（未验证）" if c.status == "unverified" else "") for c in kept)
        cited = [Cited(c.document_id, meta[c.version_id]["doc_type"]) for k in final for c in k.citations]
        return FastAnswer(text, claims, abstained, notices,
                          must_verify(self.conn, scope.tenant_id, question=question, time_params=timed, cited=cited),
                          basis, model=out.model,
                          usage={"input_tokens": out.usage.input_tokens, "output_tokens": out.usage.output_tokens},
                          seconds=time.monotonic() - started)

    def _meta(self, hits: list[Hit], system_t) -> dict[str, dict]:
        """版本号、标题、类型与有效期；回溯查询给模型看的是回溯时间点当时记录的有效期。"""
        if not hits:
            return {}
        t_cond = ("val.recorded_at <= %(t)s AND (val.replaced_at IS NULL OR val.replaced_at > %(t)s)" if system_t
                  else "val.replaced_at IS NULL")
        rows = self.conn.execute(
            "SELECT v.id, v.version_no, d.title, d.doc_type, val.valid_from, val.valid_to FROM document_version v "
            "JOIN document d ON d.id = v.document_id LEFT JOIN version_validity val ON val.version_id = v.id "
            f"AND val.recorded_at IS NOT NULL AND {t_cond} WHERE v.id = ANY(%(ids)s::uuid[])",
            {"ids": [h.version_id for h in hits], "t": system_t}).fetchall()
        return {str(r["id"]): r for r in rows}
