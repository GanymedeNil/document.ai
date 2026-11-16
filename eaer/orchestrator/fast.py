"""Fast 模式与各模式共用的步骤（开发文档 8.3、8.6）。

Fast：检索 → 生成 → 引用检查（9.2 的确定性检查）→ 判断是否需要验证 → 组装。
这里的 Steps 拆成可复用的几步，Verified 与编排器（engine.py）直接复用，Verified 优先复用 Fast 已取得的证据。

- 生成：模型只看检索到的证据原文和版本信息（回溯时取当时记录的有效期），给出结论列表，每条结论列出依据的
  证据 id，并标出核心结论；证据之间对所问取值说法不一致时列出冲突的证据（8.2 的升级信号）。
  模型支持结构化输出时用 JSON；不支持时退回纯文本、每句末尾用 [证据 id] 标注，Fast 照常工作（12.1、AC-12-1）。
  授权范围不进入模型上下文（3.2 第 4 条）。
- 组装（8.6）：supported 附引用；unverified 标“未验证”；unsupported 删除，核心结论不通过则整体拒答。
  最终回答由保留的结论拼成，模型的自由文字不直接给用户（AC-8-5）。答了一部分、另有要点没有证据时，回答末尾写明缺什么。
- 严格模式与宽松模式（迁移 0015，设计变更记录 2026-11-16）：严格模式只认证据原文；证据按类别给规定、没写明问题中的
  对象属于哪一类时逐类给出。宽松模式（默认）另外允许：按证据里的单价与问题给的数量写出算式算合计（代码核对算术与
  单价的来历，结果标“未验证”并写出算式）；按常识判断对象属于哪一类（单列一条 assumption 结论，不引证据，标“常识推断”）；
  证据只覆盖一部分时先答能答的部分。两种模式下制度的规定、标准与取值都只能出自证据。
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
from eaer.verifier.deterministic import FORMULA_REASON, DeterministicChecker, Target, formula_results

ABSTAIN = "没有找到能够回答这个问题的资料。"
SCHEMA = {
    "type": "object", "required": ["abstain", "claims"],
    "properties": {
        "abstain": {"type": "boolean"},
        "missing": {"type": "string"},
        "partial": {"type": "boolean"},
        "claims": {"type": "array", "items": {
            "type": "object", "required": ["text", "evidence_ids", "core"],
            "properties": {"text": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}},
                           "core": {"type": "boolean"}, "assumption": {"type": "boolean"}}}},
        "conflicts": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
    },
}
# 两种模式共用、措辞不同的几条（Deep 的生成提示也用）
HEAD_STRICT = "你是企业制度问答助手，只能依据下面给出的证据回答，不得使用证据以外的知识。"
HEAD_LENIENT = ("你是企业制度问答助手。制度的规定、标准与取值只能依据下面给出的证据，不得使用证据以外的知识；"
                "只有下文所说的计算与对象归类可以超出证据原文，并且要按要求写明。")
VALUES_STRICT = "结论中出现的数值、日期、单位必须与证据原文一致，不要换算或改写数值。"
VALUES_LENIENT = "结论中出现的数值、日期、单位必须与证据原文一致，不要换算单位或改写数值；需要计算时按下文的要求写出算式。"
ABSTAIN_RULE = ("证据不足以回答时，abstain=true，claims 为空，并在 missing 中说明缺少哪类信息；不要猜测，不要提及证据之外的文件，\n"
                "   missing 里不要复述证据原文或其中的取值。")
PARTIAL_STRICT = ("证据覆盖了问题的一部分、只缺另一部分时（问题有几个要点或比较几方，\n"
                  "   证据只覆盖其中一些，如两篇报道只找到一篇），另设 partial=true；证据与问题无关时不设。")
PARTIAL_LENIENT = ("证据只覆盖问题的一部分时（问题有几个要点或比较几方，证据只覆盖其中一些），不要整题拒答：\n"
                   "   abstain=false，回答能回答的部分，并在 missing 中说明还缺什么。")
CATEGORY_STRICT = ("证据按类别（如城市等级、职级、地区）给出不同规定，而问题说的是具体对象、证据中又没有写明它属于哪一类时，\n"
                   "   不要拒答，也不要替它归类：写一条结论说明所引规定按哪些类别区分、证据没有写明该对象属于哪一类（core=false），\n"
                   "   再为每一类各写一条结论、各自引用（core=true）。")
CALC_STRICT = ("问题要的是按数量算出的合计（如住几晚、出差几天一共多少），而证据只给出单价或标准时，不要拒答，也不要自己计算：\n"
               "   写出证据中的单价与标准（core=true），再写一条结论说明证据只规定了单价、没有直接给出所问的合计（core=false，\n"
               "   引用单价所在的证据）。")
CALC_LENIENT = ("问题给出了天数、晚数、人数、次数等数量，需要按证据中的单价或标准算出合计时，可以计算：在结论里写出完整算式\n"
                "   （如“每晚 500 元 × 3 晚 = 1,500 元”；合计接着写在同一条结论里，如“1,500 + 300 = 1,800 元”），算式中的单价与\n"
                "   标准必须出自所引证据，乘除用的数量取自问题；问题的说法不是确切数量时（如“一周”“半个月”），写明按多少计（如“按 7 晚计”）。")
CATEGORY_LENIENT = ("证据按类别（如城市等级、职级、地区）给出不同规定，而问题说的是具体对象、证据中又没有写明它属于哪一类时，\n"
                    "   可以按常识判断它属于哪一类：单独写一条结论陈述这个判断，并说明证据没有写明（如“所引办法没有列出一线城市的范围，\n"
                    "   按常识，A 市属于一线城市”），这条结论设 assumption=true、evidence_ids 为空、core=false；再按判断出的类别回答。\n"
                    "   拿不准时不要判断，为每一类各写一条结论。常识只用于判断问题中的对象属于哪一类，制度的规定、标准与取值仍只能出自证据。")


def policy_rules(strict: bool) -> list[str]:
    """两种模式各自多出的几条要求（Fast 与 Deep 的生成提示共用）。"""
    return [CALC_STRICT, CATEGORY_STRICT] if strict else [CALC_LENIENT, CATEGORY_LENIENT]


def system_prompt(strict: bool) -> str:
    rules = [
        "把回答拆成若干条结论，每条结论是一句完整的陈述，并列出它所依据的证据 id（只能用给出的 id）。",
        VALUES_STRICT if strict else VALUES_LENIENT,
        "把直接回答问题的那条结论标为 core=true，其余补充说明标为 core=false。",
        "注意每条证据所属版本的有效期：只用在目标日期有效的版本回答；有效期标为“推断”或“未知”的，照常引用即可。\n"
        "   记录类文档（报告、新闻、纪要等）标的是文档日期与出处：问题按出处或日期指认某篇文档时（如“某来源某日的报道”），\n"
        "   按标注对应到那篇，不要把别的文档的内容算到它头上。",
        "如果不同证据对所问的取值说法不一致，在 conflicts 中列出互相冲突的证据 id，并为每种说法各写一条结论，写明出自哪份文件\n"
        "   （如“《差旅报销管理办法》第 3 版规定……”“《关于……的补充通知》规定……”），各自只引用支持它的那条证据（一条结论里出现的\n"
        "   每个数值都必须能在它引用的证据中找到，不要在一条结论里同时写两种说法），不要替用户选择其中一种。",
        ABSTAIN_RULE + (PARTIAL_STRICT if strict else PARTIAL_LENIENT),
        *policy_rules(strict),
        DATA_RULE,
    ]
    return ((HEAD_STRICT if strict else HEAD_LENIENT) + "\n要求：\n"
            + "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1)))


def plain_prompt(strict: bool) -> str:
    """不支持结构化输出的模型用的纯文本提示。"""
    extra = ("问题要合计而证据只给单价时，写出单价、不要自己计算。证据按类别给出不同规定、没写明问题中的对象属于哪一类时，"
             "逐类写出规定。" if strict else
             "需要按证据里的单价与问题给的数量算合计时，写出完整算式（如“每晚 500 元 × 3 晚 = 1,500 元”），单价必须出自证据。"
             "证据按类别给出不同规定、没写明问题中的对象属于哪一类时，逐类写出规定。")
    return ((HEAD_STRICT if strict else HEAD_LENIENT) + "\n"
            "用中文回答，每句话写一条结论，句末用方括号标注所依据的证据 id，例如“……为 650 元/晚。[ev_0123456789abcdef]”。\n"
            "第一句直接回答问题。数值、日期、单位必须与证据原文一致。只用在目标日期有效的版本回答。" + extra + "\n"
            "证据不足以回答时只输出：无法回答。\n" + DATA_RULE)


SYSTEM, SYSTEM_STRICT = system_prompt(False), system_prompt(True)
SYSTEM_PLAIN, SYSTEM_PLAIN_STRICT = plain_prompt(False), plain_prompt(True)
COMMON_KNOWLEDGE = "常识推断：所引文件没有写明，由模型按常识判断"
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
    partial: bool = False              # 拒答时证据覆盖了问题的一部分（转入 Deep 的信号“部分要点没有证据覆盖”）


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

    def generate(self, question: str, hits: list[Hit], meta: dict, on: date, *, strict: bool = False) -> Draft:
        evidence = "\n\n".join(evidence_block(h, meta[h.version_id]) for h in hits)
        user = f"目标日期：{on.isoformat()}\n问题：{escape(question)}\n\n证据：\n{evidence}"
        if self.model.capabilities().structured_output:
            out = self.model.generate([Message("system", SYSTEM_STRICT if strict else SYSTEM), Message("user", user)],
                                      SCHEMA)
            p = out.parsed or {}
            return Draft(p.get("claims", []) if not p.get("abstain") else [], bool(p.get("abstain")), p.get("missing"),
                         [list(c) for c in p.get("conflicts", []) if len(c) > 1], out.model, out.usage,
                         partial=bool(p.get("abstain") and p.get("partial")))
        out = self.model.generate([Message("system", SYSTEM_PLAIN_STRICT if strict else SYSTEM_PLAIN),
                                   Message("user", user)], None)
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
              version_targets: dict[str, Target] | None = None, *, strict: bool = False) -> list[ClaimOut]:
        """version_targets：Deep 比较不同版本或时间点时，各版本按自己的目标时间核对（见 DeterministicChecker）。
        宽松模式：核对结论里写明的算式，前面结论算出的数后面可以接着用；assumption 结论（常识归类）不引证据、
        标为未验证且不作核心结论。严格模式下 assumption 结论判为不支持（删除）。"""
        by_id = {h.evidence_id: h for h in hits}
        checker = DeterministicChecker(self.conn, self.authorizer, scope, self.store)
        out: list[ClaimOut] = []
        derived: set = set()
        for c in claims:
            if c.get("assumption"):
                status, why = (("unsupported", "严格模式只依据证据回答，不采用常识推断") if strict
                               else ("unverified", COMMON_KNOWLEDGE))
                out.append(ClaimOut(c["text"], status, False, [], {"common_knowledge": {"outcome": status, "reason": why}}))
                continue
            ids = list(dict.fromkeys(c["evidence_ids"]))
            r = checker.check(c["text"], ids, retrieved=set(by_id), target=target, version_targets=version_targets,
                              formulas=not strict, derived=frozenset(derived))
            if not strict and r.outcome != "unsupported":
                derived |= formula_results(c["text"])
            status = "supported" if r.outcome == "pass" else r.outcome
            out.append(ClaimOut(c["text"], status, bool(c.get("core")), self.citations(r.evidence_ids, by_id, meta),
                                {k: {"outcome": v.outcome, "reason": v.reason} for k, v in r.checks.items()},
                                removed=list(r.removed)))
        if out and not any(c.core for c in out):
            first = next((c for c in out if "common_knowledge" not in c.checks), None)
            if first is not None:
                first.core = True                  # 模型没有标核心结论时，第一条（常识推断以外的）视为核心
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
        if (draft is not None and draft.abstain) or not kept or all("common_knowledge" in c.checks for c in kept):
            reason = draft.missing if draft is not None and draft.abstain else None
            return ABSTAIN + (f"缺少：{reason}" if reason else ""), True
        text = "".join(c.text + _mark(c) for c in kept)
        if draft is not None and draft.missing:              # 答了一部分（宽松模式）：写明还缺什么
            text += f"另外，没有找到：{draft.missing.rstrip('。')}。"
        return text, False

    def cited(self, claims: list[ClaimOut], meta: dict) -> list[Cited]:
        return [Cited(c.document_id, meta[c.version_id]["doc_type"]) for k in claims if k.status != "unsupported"
                for c in k.citations]


def claim_basis(checks: dict) -> str:
    """结论的依据：evidence 证据原文；calculation 由证据里的单价按算式算出；common_knowledge 常识推断（宽松模式）。"""
    if "common_knowledge" in checks:
        return "common_knowledge"
    if (checks.get("value_match") or {}).get("reason", "").startswith(FORMULA_REASON):
        return "calculation"
    return "evidence"


def derived_claim(c: ClaimOut) -> bool:
    """宽松模式下超出原文的结论：按算式算出的数、常识推断。已标未验证，不再交给语义检查（9.2：全部通过才做语义检查）。"""
    return claim_basis(c.checks) != "evidence"


def _mark(c: ClaimOut) -> str:
    if c.status != "unverified":
        return ""
    return "（常识推断，文件未写明）" if "common_knowledge" in c.checks else "（未验证）"


class FastPipeline:
    """单独运行 Fast（供试跑与测试）；完整的模式选择与升级见 engine.Engine。"""

    def __init__(self, conn: psycopg.Connection, retriever: Retriever, model: ModelClient, authorizer, *, store=None):
        self.steps = Steps(conn, retriever, model, authorizer, store=store)
        self.conn = conn

    def answer(self, scope: Scope, question: str, *, today: date, as_of_valid: date | None = None,
               as_of_system: date | None = None, time_params: bool | None = None, strict: bool = False) -> FastAnswer:
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
        draft = s.generate(question, hits, meta, on, strict=strict)
        claims = s.check(scope, draft.claims, hits, meta, target, strict=strict)
        text, abstained = s.assemble(claims, draft)
        cited = [] if abstained else s.cited(claims, meta)
        return FastAnswer(text, claims, abstained, notices,
                          must_verify(self.conn, scope.tenant_id, question=question, time_params=timed, cited=cited),
                          basis, model=draft.model,
                          usage={"input_tokens": draft.usage.input_tokens, "output_tokens": draft.usage.output_tokens},
                          seconds=time.monotonic() - started)
