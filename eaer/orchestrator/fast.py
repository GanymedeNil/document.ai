"""Fast 模式与各模式共用的步骤（开发文档 8.3、8.6）。

Fast：检索 → 生成 → 引用检查（9.2 的确定性检查）→ 判断是否需要验证 → 组装。
这里的 Steps 拆成可复用的几步，Verified 与编排器（engine.py）直接复用，Verified 优先复用 Fast 已取得的证据。

- 生成：模型只看检索到的证据原文和版本信息（回溯时取当时记录的有效期），给出结论列表，每条结论列出依据的
  证据 id，并标出核心结论；证据之间对所问取值说法不一致时列出冲突的证据（8.2 的升级信号）。
  模型支持结构化输出时用 JSON；不支持时退回纯文本、每句末尾用 [证据 id] 标注，Fast 照常工作（12.1、AC-12-1）。
  授权范围不进入模型上下文（3.2 第 4 条）。
- 组装（8.6）：supported 附引用；unverified 标“未验证”；unsupported 删除，核心结论不通过则整体拒答。
  最终回答由保留的结论拼成，模型的自由文字不直接给用户（AC-8-5）。
拒答文本不透露用户无权查看的文档：模型只见过本次检索到的、用户有权查看的证据。
提示注入（10.3）：证据放进带随机边界的证据区，系统提示声明区内指令不执行（eaer/domain/injection.py）。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime

import psycopg

from eaer.adapters.model import Message, ModelClient, Usage
from eaer.domain.identity import Scope
from eaer.domain.injection import DATA_RULE, escape, fence
from eaer.lifecycle.asof import end_of_day
from eaer.orchestrator.forced_verify import Cited, must_verify
from eaer.retrieval.service import Hit, Retriever
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
        "conflicts": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
    },
}
SYSTEM = """你是企业制度问答助手，只能依据下面给出的证据回答，不得使用证据以外的知识。
要求：
1. 把回答拆成若干条结论，每条结论是一句完整的陈述，并列出它所依据的证据 id（只能用给出的 id）。
2. 结论中出现的数值、日期、单位必须与证据原文一致，不要换算或改写数值。
3. 把直接回答问题的那条结论标为 core=true，其余补充说明标为 core=false。
4. 注意每条证据所属版本的有效期：只用在目标日期有效的版本回答；有效期标为“推断”或“未知”的，照常引用即可。
   记录类文档（报告、新闻、纪要等）标的是文档日期与出处：问题按出处或日期指认某篇文档时（如“某来源某日的报道”），
   按标注对应到那篇，不要把别的文档的内容算到它头上。
5. 如果不同证据对所问的取值说法不一致，在 conflicts 中列出互相冲突的证据 id，并为每种说法各写一条结论，写明出自哪份文件
   （如“《差旅报销管理办法》第 3 版规定……”“《关于……的补充通知》规定……”），各自只引用支持它的那条证据（一条结论里出现的
   每个数值都必须能在它引用的证据中找到，不要在一条结论里同时写两种说法），不要替用户选择其中一种。
6. 证据不足以回答时，abstain=true，claims 为空，并在 missing 中说明缺少哪类信息；不要猜测，不要提及证据之外的文件，
   missing 里不要复述证据原文或其中的取值。
7. """ + DATA_RULE
SYSTEM_PLAIN = """你是企业制度问答助手，只能依据下面给出的证据回答，不得使用证据以外的知识。
用中文回答，每句话写一条结论，句末用方括号标注所依据的证据 id，例如“……为 650 元/晚。[ev_0123456789abcdef]”。
第一句直接回答问题。数值、日期、单位必须与证据原文一致。只用在目标日期有效的版本回答。
证据不足以回答时只输出：无法回答。
""" + DATA_RULE
_ID = re.compile(r"\[(ev_[0-9a-f]+)\]")


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
    semantic: dict | None = None       # Verified 的语义检查结果
    repaired: bool = False             # 经过定向补检索
    removed: list[str] = field(default_factory=list)   # 授权核对未通过而移除的证据（记安全告警）


@dataclass
class Draft:
    claims: list[dict]
    abstain: bool
    missing: str | None
    conflicts: list[list[str]]
    model: str
    usage: Usage


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


def evidence_block(h: Hit, meta: dict) -> str:
    """一条证据的标注：文档标题与版本；规范类给有效期及其来源，记录类给文档日期（记录类没有有效期，原先标“生效 未知”）；
    有出处时给出处。记录类的问题常按出处与日期指认文档（“TechCrunch 10 月 7 日那篇报道”），而正文的其余段落看不出
    属于哪一篇，所以标在每条证据上。出处来自数据源，与标题一样转义后放在证据区内。"""
    if meta.get("doc_type") == "record":
        when = f"文档日期 {meta.get('doc_date') or '未知'}"
    else:
        vf, vt = meta.get("valid_from"), meta.get("valid_to")
        when = (f"生效 {vf or '未知'}" + (f"，作废 {vt}" if vt else "")
                + f"；有效期来源：{_grade_text(h.valid_grade)}")
    source = f"；出处：{escape(meta['publisher'])}" if meta.get("publisher") else ""
    return fence(f"[{h.evidence_id}] 文档《{escape(meta['title'])}》第 {meta['version_no']} 版；{when}{source}", h.content)


def target_of(on: date, as_of_system: date | None) -> Target:
    return Target(on, end_of_day(as_of_system) if as_of_system else None)


class Steps:
    def __init__(self, conn: psycopg.Connection, retriever: Retriever, model: ModelClient, authorizer, *, store=None):
        self.conn, self.retriever, self.model, self.authorizer, self.store = conn, retriever, model, authorizer, store

    # ---------- 检索 ----------

    def retrieve(self, scope: Scope, query: str, *, today: date, as_of_valid: date | None,
                 as_of_system: date | None) -> list[Hit]:
        return self.retriever.search(scope, query, today=today, as_of_valid=as_of_valid, as_of_system=as_of_system)

    def meta(self, hits: list[Hit], system_t: datetime | None) -> dict[str, dict]:
        """版本号、标题、类型、有效期、文档日期与出处；回溯查询取回溯时间点当时记录的有效期。"""
        if not hits:
            return {}
        t_cond = ("val.recorded_at <= %(t)s AND (val.replaced_at IS NULL OR val.replaced_at > %(t)s)" if system_t
                  else "val.replaced_at IS NULL")
        rows = self.conn.execute(
            "SELECT v.id, v.version_no, d.title, d.doc_type, v.doc_date, v.publisher, val.valid_from, val.valid_to "
            "FROM document_version v "
            "JOIN document d ON d.id = v.document_id LEFT JOIN version_validity val ON val.version_id = v.id "
            f"AND val.recorded_at IS NOT NULL AND {t_cond} WHERE v.id = ANY(%(ids)s::uuid[])",
            {"ids": list({h.version_id for h in hits}), "t": system_t}).fetchall()
        return {str(r["id"]): r for r in rows}

    # ---------- 生成 ----------

    def generate(self, question: str, hits: list[Hit], meta: dict, on: date) -> Draft:
        evidence = "\n\n".join(evidence_block(h, meta[h.version_id]) for h in hits)
        user = f"目标日期：{on.isoformat()}\n问题：{escape(question)}\n\n证据：\n{evidence}"
        if self.model.capabilities().structured_output:
            out = self.model.generate([Message("system", SYSTEM), Message("user", user)], SCHEMA)
            p = out.parsed or {}
            return Draft(p.get("claims", []) if not p.get("abstain") else [], bool(p.get("abstain")), p.get("missing"),
                         [list(c) for c in p.get("conflicts", []) if len(c) > 1], out.model, out.usage)
        out = self.model.generate([Message("system", SYSTEM_PLAIN), Message("user", user)], None)
        return self._plain(out.text, out.model, out.usage)

    @staticmethod
    def _plain(text: str, model: str, usage: Usage) -> Draft:
        """纯文本退回：按句切分，句中 [证据 id] 即依据；第一句为核心结论。"""
        if text.strip().startswith("无法回答"):
            return Draft([], True, None, [], model, usage)
        claims = []
        # 模型常把引用写在句号之后（“……元。[ev_x]”），先把引用挪到句号之前再按句切分
        text = re.sub(r"([。！？!?])\s*((?:\[ev_[0-9a-f]+\]\s*)+)", r"\2\1", text)
        for s in re.split(r"(?<=[。！？!?])", text):
            ids = _ID.findall(s)
            body = _ID.sub("", s).strip()
            if body and ids:
                claims.append({"text": body, "evidence_ids": ids, "core": not claims})
        return Draft(claims, not claims, None, [], model, usage)

    # ---------- 引用检查 ----------

    def check(self, scope: Scope, claims: list[dict], hits: list[Hit], meta: dict, target: Target,
              version_targets: dict[str, Target] | None = None) -> list[ClaimOut]:
        """version_targets：Deep 比较不同版本或时间点时，各版本按自己的目标时间核对（见 DeterministicChecker）。"""
        by_id = {h.evidence_id: h for h in hits}
        checker = DeterministicChecker(self.conn, self.authorizer, scope, self.store)
        out: list[ClaimOut] = []
        for c in claims:
            ids = list(dict.fromkeys(c["evidence_ids"]))
            r = checker.check(c["text"], ids, retrieved=set(by_id), target=target, version_targets=version_targets)
            status = "supported" if r.outcome == "pass" else r.outcome
            out.append(ClaimOut(c["text"], status, bool(c.get("core")), self.citations(r.evidence_ids, by_id, meta),
                                {k: {"outcome": v.outcome, "reason": v.reason} for k, v in r.checks.items()},
                                removed=list(r.removed)))
        if out and not any(c.core for c in out):
            out[0].core = True                     # 模型没有标核心结论时，第一条视为核心
        return out

    @staticmethod
    def citations(ids: list[str], by_id: dict[str, Hit], meta: dict) -> list[Citation]:
        return [Citation(i, by_id[i].document_id, by_id[i].version_id, meta[by_id[i].version_id]["version_no"],
                         by_id[i].locator, by_id[i].valid_grade) for i in ids if i in by_id]

    # ---------- 组装 ----------

    @staticmethod
    def assemble(claims: list[ClaimOut], draft: Draft | None) -> tuple[str, bool]:
        kept = [c for c in claims if c.status != "unsupported"]
        if any(c.core and c.status == "unsupported" for c in claims):
            return ABSTAIN + "核心结论未能在证据中得到核实。", True
        if (draft is not None and draft.abstain) or not kept:
            reason = draft.missing if draft is not None and draft.abstain else None
            return ABSTAIN + (f"缺少：{reason}" if reason else ""), True
        return "".join(c.text + ("（未验证）" if c.status == "unverified" else "") for c in kept), False

    def cited(self, claims: list[ClaimOut], meta: dict) -> list[Cited]:
        return [Cited(c.document_id, meta[c.version_id]["doc_type"]) for k in claims if k.status != "unsupported"
                for c in k.citations]


class FastPipeline:
    """单独运行 Fast（供试跑与测试）；完整的模式选择与升级见 engine.Engine。"""

    def __init__(self, conn: psycopg.Connection, retriever: Retriever, model: ModelClient, authorizer, *, store=None):
        self.steps = Steps(conn, retriever, model, authorizer, store=store)
        self.conn = conn

    def answer(self, scope: Scope, question: str, *, today: date, as_of_valid: date | None = None,
               as_of_system: date | None = None, time_params: bool | None = None) -> FastAnswer:
        started = time.monotonic()
        s = self.steps
        on = as_of_valid or as_of_system or today
        target = target_of(on, as_of_system)
        basis = {"as_of_valid": as_of_valid, "as_of_system": as_of_system,
                 "source": "parameter" if (as_of_valid or as_of_system) else "today"}
        notices = []
        if as_of_valid:
            notices.append(f"按 {as_of_valid.year} 年 {as_of_valid.month} 月 {as_of_valid.day} 日生效的版本回答")
        if as_of_system:
            notices.append(f"按截至 {as_of_system.year} 年 {as_of_system.month} 月 {as_of_system.day} 日的系统记录回答")
        timed = bool(time_params if time_params is not None else (as_of_valid or as_of_system))
        hits = s.retrieve(scope, question, today=today, as_of_valid=as_of_valid, as_of_system=as_of_system)
        if not hits:
            return FastAnswer(ABSTAIN, [], True, notices,
                              must_verify(self.conn, scope.tenant_id, question=question, time_params=timed),
                              basis, seconds=time.monotonic() - started)
        meta = s.meta(hits, target.system_t)
        draft = s.generate(question, hits, meta, on)
        claims = s.check(scope, draft.claims, hits, meta, target)
        text, abstained = s.assemble(claims, draft)
        cited = [] if abstained else s.cited(claims, meta)
        return FastAnswer(text, claims, abstained, notices,
                          must_verify(self.conn, scope.tenant_id, question=question, time_params=timed, cited=cited),
                          basis, model=draft.model,
                          usage={"input_tokens": draft.usage.input_tokens, "output_tokens": draft.usage.output_tokens},
                          seconds=time.monotonic() - started)
