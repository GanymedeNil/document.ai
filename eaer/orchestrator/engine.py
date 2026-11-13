"""查询执行的编排（开发文档 8.1–8.6）：一次提问是一个有预算上限的状态机。

1. 时间点：接口参数优先，其次规则解析，规则定不了的交给模型（7.5）；解析出的时间点回显在回答开头；
2. Router 决定入口模式；mode_hint 只升不降（8.1、AC-8-4）；
3. 入口为 Deep：按计划模板检索与生成（deep.py）；入口为 Fast 或 Verified：首轮检索与生成，Verified 复用这些证据；
4. 检索后再判断（8.1、8.2）：
   - Fast 转入 Deep 的信号：重排后最高分低于阈值；引用检查失败；问题中有部分要点没有证据覆盖。
     转入后按多跳模板，Fast 的检索结果是第一跳，Fast 已用的轮数计入 Deep 的预算；
   - Fast 或 Deep 转入 Verified 的信号：命中强制验证清单；证据之间存在冲突；命中已被取代或已过期的版本；
     结论依赖图表数值；
   - Deep 不可用（deep_enabled 关闭，或模型没有结构化输出、上下文不够长）时，记录信号并说明，由 Fast 回答（12.1）；
5. Verified（8.3、9.2–9.4）：确定性检查 → 语义检查 → 不通过的结论以结论内容定向补检索一轮 → 再验证；
   Deep 比较不同版本时，各版本按自己的日期核对与补检索；
   模型不支持结构化输出时 Verified 关闭，时效性和强制验证类结论以“未验证”作答（12.1、AC-12-1）；
6. 组装（8.6）。检索轮数受预算限制（8.5：Fast 1 轮；Deep 3 轮，含转入前 Fast 已用的；Verified 补检索 1 轮），
   用尽仍有缺口则拒答。
结果写入 query_log（log_query），记录授权决定、模型标识、索引构建编号、成本与 Deep 的计划和动作（10.1 第 8 步、12.2）。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone

import psycopg
from psycopg.types.json import Jsonb

from eaer.adapters.model import Message, ModelClient
from eaer.adapters.online import collect_calls, summarize
from eaer.domain.identity import Scope
from eaer.orchestrator.deep import DeepOutcome, DeepRunner
from eaer.orchestrator.fast import ABSTAIN, ClaimOut, Draft, Steps, evidence_block, target_of
from eaer.orchestrator.forced_verify import Cited, must_verify
from eaer.orchestrator.router import decide
from eaer.orchestrator.timeparse import TimeBasis, resolve
from eaer.retrieval.service import Retriever
from eaer.verifier.deterministic import DeterministicChecker, Target
from eaer.verifier.semantic import DEFAULT_MIN_CONFIDENCE, SemanticChecker

STATUS_RANK = {"supported": 0, "unverified": 1, "unsupported": 2}
DEFAULT_THRESHOLDS = {"semantic_min_confidence": DEFAULT_MIN_CONFIDENCE, "rerank_min_top": 0.3,
                      "deep_min_context_tokens": 16000}
DEFAULT_BUDGETS = {"fast_rounds": 1, "deep_rounds": 3, "verified_repair_rounds": 1}


@dataclass
class QueryResult:
    session_id: str
    status: str
    mode: str
    mode_path: list[str]
    router: dict
    time_basis: dict
    answer: str
    claims: list[ClaimOut]
    abstained: bool
    notices: list[str]
    forced_verify: list[str]
    escalations: list[str]             # 实际发生的模式转换及原因
    retrieval_rounds: int
    model: str
    index_build_id: str
    cost: dict = field(default_factory=dict)
    seconds: float = 0.0
    alerts: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)   # 检索后观察到的 8.2 信号（不论是否引起转换）
    plan: dict | None = None           # Deep 的计划模板、分组、缺口与动作记录（8.4 的 plan、gaps、history）

    @property
    def evidence_ids(self) -> list[str]:
        return list(dict.fromkeys(c.evidence_id for k in self.claims if k.status != "unsupported" for c in k.citations))

    def response(self) -> dict:
        """接口响应（11.2）。"""
        claims = [{"text": c.text, "status": c.status, "core": c.core,
                   "citations": [asdict(x) for x in c.citations]} for c in self.claims if c.status != "unsupported"]
        return {"session_id": self.session_id, "status": self.status, "mode": self.mode, "mode_path": self.mode_path,
                "time_basis": self.time_basis, "answer": self.answer, "claims": [] if self.abstained else claims,
                "abstained": self.abstained, "notices": self.notices}


def _worst(a: str, b: str) -> str:
    return a if STATUS_RANK[a] >= STATUS_RANK[b] else b


def model_id(model: ModelClient) -> str:
    gen = getattr(model, "gen", None)
    return getattr(gen, "model", None) or type(model).__name__


TIME_SCHEMA = {"type": "object", "required": ["as_of_valid"],
               "properties": {"as_of_valid": {"type": "string"}, "as_of_system": {"type": "string"}}}


class Engine:
    def __init__(self, conn: psycopg.Connection, retriever: Retriever, model: ModelClient, authorizer, *,
                 store=None, thresholds: dict | None = None, budgets: dict | None = None, deep_enabled: bool = True):
        self.conn, self.retriever, self.model, self.authorizer = conn, retriever, model, authorizer
        self.steps = Steps(conn, retriever, model, authorizer, store=store)
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.budgets = {**DEFAULT_BUDGETS, **(budgets or {})}
        self.deep_enabled = deep_enabled       # 准入检查（12.3）不达标时关闭

    def deep_unavailable(self) -> str | None:
        """Deep 不可用的原因；可用时返回 None（12.1：需要稳定的结构化输出与足够的上下文长度）。"""
        if not self.deep_enabled:
            return "Deep 尚未开放"
        caps = self.model.capabilities()
        if not caps.structured_output:
            return "所用模型不支持结构化输出，Deep 不可用"
        if caps.context_tokens < int(self.thresholds["deep_min_context_tokens"]):
            return f"所用模型上下文长度 {caps.context_tokens} 不足，Deep 不可用"
        return None

    def _deep(self) -> DeepRunner:
        return DeepRunner(self.steps, self.retriever, self.model, max_rounds=int(self.budgets["deep_rounds"]))

    # ---------- 时间点的模型解析（规则定不了时） ----------

    def _resolve_with_model(self, question: str, today: date):
        if not self.model.capabilities().structured_output:
            return None, None
        out = self.model.generate([Message("system",
            "把问题里的相对时间换算成具体日期（YYYY-MM-DD）。as_of_valid 是问题关心的生效日期；问题问的是“当时系统里记录的”"
            "才给 as_of_system。换算不了时 as_of_valid 输出空字符串。只输出 JSON。"),
            Message("user", f"今天是 {today.isoformat()}。问题：{question}")], TIME_SCHEMA)
        p = out.parsed or {}

        def d(v):
            try:
                return date.fromisoformat(v) if v else None
            except ValueError:
                return None
        return d(p.get("as_of_valid")), d(p.get("as_of_system"))

    # ---------- 主流程 ----------

    def ask(self, scope: Scope, question: str, *, today: date, as_of_valid: date | None = None,
            as_of_system: date | None = None, mode_hint: str | None = None,
            session_id: str | None = None) -> QueryResult:
        started = time.monotonic()
        with collect_calls() as calls:
            r = self._ask(scope, question, today, as_of_valid, as_of_system, mode_hint)
        r.session_id = session_id or str(uuid.uuid4())
        r.cost = summarize(calls)
        r.seconds = round(time.monotonic() - started, 3)
        return r

    def _ask(self, scope, question, today, as_of_valid, as_of_system, mode_hint) -> QueryResult:
        s = self.steps
        tb = resolve(question, today=today, as_of_valid=as_of_valid, as_of_system=as_of_system,
                     resolver=self._resolve_with_model)
        notices = tb.echo()
        route = decide(question, tb, mode_hint)
        verified_on = self.model.capabilities().structured_output
        deep_off = self.deep_unavailable()
        on = tb.as_of_valid or tb.as_of_system or today
        target = target_of(on, tb.as_of_system)
        escalations: list[str] = []
        start = route.mode
        if start == "deep" and deep_off:
            notices.append(f"这个问题需要 Deep 模式（{route.reason}），{deep_off}，先按 Fast 检索回答")
            start = "fast"
        mode_path = [start]
        deep: DeepOutcome | None = None

        if start == "deep":
            deep = self._deep().run(scope, question, tb, today=today, target=target)
            hits, meta, draft, rounds = deep.hits, deep.meta, deep.draft, deep.rounds
        else:
            hits = s.retrieve(scope, question, today=today, as_of_valid=tb.as_of_valid, as_of_system=tb.as_of_system)
            rounds = 1
            meta = s.meta(hits, target.system_t)
            draft = s.generate(question, hits, meta, on) if hits else None
        vt = deep.version_targets if deep else None
        claims = s.check(scope, draft.claims, hits, meta, target, vt) if draft else []

        # Fast 检索后判断是否转入 Deep（8.2）
        deep_signals = self._deep_signals(hits, claims, draft) if mode_path == ["fast"] else []
        if deep_signals and deep_off is None:
            mode_path.append("deep")
            escalations.append("转入 Deep：" + "、".join(deep_signals))
            deep = self._deep().run(scope, question, tb, today=today, target=target, template="multihop", seed=hits,
                                    seed_missing=draft.missing if draft else None, rounds_used=rounds)
            hits, meta, draft, rounds, vt = deep.hits, deep.meta, deep.draft, deep.rounds, deep.version_targets
            claims = s.check(scope, draft.claims, hits, meta, target, vt)
        elif deep_signals:
            escalations.append("应转入 Deep：" + "、".join(deep_signals) + f"（{deep_off}）")

        # Fast 或 Deep 检索后判断是否转入 Verified（8.1、8.2）
        forced, signals = self._verified_signals(scope, question, tb, claims, draft, meta)
        need_verified = mode_path[-1] == "verified" or bool(signals)
        alerts = [f"安全告警：授权核对未通过，已移除证据 {i}" for c in claims for i in c.removed]
        if need_verified and not verified_on:
            notices.append("验证模式不可用（所用模型不支持结构化输出），时效性和强制验证类结论标为“未验证”")
            for c in claims:
                if c.status == "supported":
                    c.status = "unverified"
        elif need_verified:
            if mode_path[-1] != "verified":
                mode_path.append("verified")
                escalations.append("转入 Verified：" + "、".join(signals))
            rounds += self._verify(scope, question, claims, hits, meta, target, today, tb, vt)
        text, abstained = s.assemble(claims, draft)
        if deep is not None and deep.unresolved and not abstained:
            notices.append("以下要点没有找到证据，回答未涉及：" + "；".join(deep.unresolved))
        if not hits and draft is None:
            text = ABSTAIN
        if notices and not abstained:
            text = "；".join(n for n in notices if n.startswith("按")) + ("。" if any(n.startswith("按") for n in notices) else "") + text
        return QueryResult("", "completed", mode_path[-1], mode_path, route.to_json(), tb.to_json(), text, claims,
                           abstained, notices, forced, escalations, rounds, draft.model if draft else model_id(self.model),
                           self._build_id(), alerts=alerts,
                           signals=signals + [e for e in escalations if "Deep" in e], plan=deep.trace() if deep else None)

    def _deep_signals(self, hits, claims: list[ClaimOut], draft: Draft | None) -> list[str]:
        out = []
        if hits and self.retriever.reranker is not None and hits[0].score < self.thresholds["rerank_min_top"]:
            out.append("重排后最高分低于阈值")
        if any(c.status == "unsupported" for c in claims):
            out.append("引用检查失败")
        if draft is not None and not draft.abstain and draft.missing:
            out.append("问题中有部分要点没有证据覆盖")
        return out

    def _verified_signals(self, scope, question, tb: TimeBasis, claims: list[ClaimOut], draft: Draft | None,
                          meta: dict) -> tuple[list[str], list[str]]:
        # “所用证据”按全部结论引用的证据算，包括没通过引用检查的：它们正需要 Verified 的补检索（8.1）
        used = [Cited(x.document_id, meta[x.version_id]["doc_type"]) for c in claims for x in c.citations]
        forced = must_verify(self.conn, scope.tenant_id, question=question, time_params=tb.has_point, cited=used)
        signals = []
        if forced:
            signals.append(f"命中强制验证清单（{'、'.join(forced)}）")
        if draft is not None and draft.conflicts:
            signals.append("证据之间存在冲突")
        cited_ok = [c for c in claims if c.status != "unsupported"]
        if any(meta[v]["valid_to"] is not None for v in {x.version_id for c in cited_ok for x in c.citations}):
            signals.append("命中已被取代或已过期的版本")
        if any(x.locator.get("figure") is not None for c in cited_ok for x in c.citations):
            signals.append("结论依赖图表数值")
        return forced, signals

    def _verify(self, scope, question, claims: list[ClaimOut], hits, meta, target: Target, today, tb: TimeBasis,
                version_targets: dict[str, Target] | None = None) -> int:
        """语义检查与定向补检索；返回额外用掉的检索轮数。
        version_targets（Deep 比较不同版本）：只引用同一目标时间的版本的结论，按那个时间判断与补检索。"""
        sem = SemanticChecker(self.model, float(self.thresholds["semantic_min_confidence"]))
        by_id = {h.evidence_id: h for h in hits}
        vt = dict(version_targets or {})

        def evidence(ids):
            return [(i, evidence_block(by_id[i], meta[by_id[i].version_id])) for i in ids if i in by_id]

        def claim_target(c: ClaimOut) -> Target:
            ts = {vt.get(x.version_id) for x in c.citations}
            return ts.pop() if len(ts) == 1 and None not in ts else target

        for c in claims:
            if c.status == "unsupported":
                continue
            v = sem.judge(c.text, evidence([x.evidence_id for x in c.citations]), claim_target(c).on)
            c.semantic = v.to_json()
            c.status = _worst(c.status, v.status)
        pending = [c for c in claims if c.status == "unsupported"]
        if not pending or self.budgets["verified_repair_rounds"] < 1:
            return 0
        checker = DeterministicChecker(self.conn, self.authorizer, scope, self.steps.store)
        for c in pending:                         # 补检索一轮：每条不通过的结论各以自身内容检索一次
            ct = claim_target(c)
            own = ct is not target
            new = self.steps.retrieve(scope, c.text, today=today, as_of_valid=ct.on if own else tb.as_of_valid,
                                      as_of_system=tb.as_of_system)
            if not new:
                continue
            meta.update(self.steps.meta([h for h in new if h.version_id not in meta], target.system_t))
            by_id.update({h.evidence_id: h for h in new})
            if own:
                for h in new:
                    vt.setdefault(h.version_id, ct)
            v = sem.judge(c.text, evidence([h.evidence_id for h in new]), ct.on)
            if v.status == "unsupported" or not v.supporting_ids:
                c.semantic, c.repaired = v.to_json(), True
                continue
            r = checker.check(c.text, v.supporting_ids, retrieved=set(by_id), target=target, version_targets=vt)
            det = "supported" if r.outcome == "pass" else r.outcome
            c.status, c.semantic, c.repaired = _worst(det, v.status), v.to_json(), True
            c.citations = Steps.citations(r.evidence_ids, by_id, meta)
            c.checks = {k: {"outcome": x.outcome, "reason": x.reason} for k, x in r.checks.items()}
        return 1

    def _build_id(self) -> str:
        ix = self.retriever.index.snapshot()
        return getattr(ix, "alias", None) or type(ix).__name__


def log_query(conn: psycopg.Connection, r: QueryResult, *, scope: Scope, user_id: str, question: str,
              asked_at: datetime | None = None) -> uuid.UUID:
    """写入 query_log（10.1 第 8 步、10.4 审计）。必填字段缺一即报错，不写半条记录（AC-10-9）。"""
    authz = {"principals": sorted(scope.principals),
             "expires_at": scope.expires_at.isoformat() if scope.expires_at else None,
             "removed_evidence": [i for c in r.claims for i in c.removed], "alerts": r.alerts}
    claims = [{"text": c.text, "status": c.status, "core": c.core, "citations": [asdict(x) for x in c.citations],
               "checks": c.checks, "semantic": c.semantic, "repaired": c.repaired} for c in r.claims]
    for name, value in (("session_id", r.session_id), ("user_id", user_id), ("question", question),
                        ("mode", r.mode), ("model_id", r.model), ("index_build_id", r.index_build_id)):
        if not value:
            raise ValueError(f"query_log 缺少必填字段 {name}")
    qid = uuid.uuid4()
    conn.execute(
        "INSERT INTO query_log (id, tenant_id, session_id, user_id, asked_at, question, time_basis, mode, mode_path, "
        "status, answer, claims, evidence_ids, abstained, authz_decision, model_id, index_build_id, cost) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (qid, scope.tenant_id, uuid.UUID(r.session_id), user_id, asked_at or datetime.now(timezone.utc), question,
         Jsonb(r.time_basis), r.mode, r.mode_path, r.status, r.answer, Jsonb(claims), r.evidence_ids, r.abstained,
         Jsonb(authz), r.model, r.index_build_id,
         Jsonb({**r.cost, "retrieval_rounds": r.retrieval_rounds, "seconds": r.seconds,
                "router": r.router, "escalations": r.escalations, "signals": r.signals,
                "forced_verify": r.forced_verify, "plan": r.plan})))
    return qid
