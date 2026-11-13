"""查询执行的编排（开发文档 8.1–8.6）：一次提问是一个有预算上限的状态机。

1. 时间点：接口参数优先，其次规则解析，规则定不了的交给模型（7.5）；解析出的时间点回显在回答开头；
2. Router 决定入口模式；mode_hint 只升不降（8.1、AC-8-4）；
3. 入口为 Deep：按计划模板检索与生成（deep.py）；入口为 Fast 或 Verified：首轮检索与生成，Verified 复用这些证据；
4. 检索后再判断（8.1、8.2）：
   - Fast 转入 Deep 的信号：重排后最高分低于阈值；引用检查失败；问题中有部分要点没有证据覆盖。
     转入后按多跳模板，Fast 的检索结果是第一跳，Fast 已用的轮数计入 Deep 的预算；
   - Fast 或 Deep 转入 Verified 的信号：命中强制验证清单；证据之间存在冲突；命中已被取代或已过期的版本；
     结论依赖图表数值；引用的证据含疑似给模型的指令文字（10.3，由语义检查判断它是否真是事实陈述）；
   - Deep 不可用（deep_enabled 关闭，或模型没有结构化输出、上下文不够长）时，记录信号并说明，由 Fast 回答（12.1）；
5. Verified（8.3、9.2–9.4）：确定性检查 → 语义检查 → 不通过的结论以结论内容定向补检索一轮 → 再验证；
   Deep 比较不同版本时，各版本按自己的日期核对与补检索；
   模型不支持结构化输出时 Verified 关闭，时效性和强制验证类结论以“未验证”作答（12.1、AC-12-1）；
6. 组装（8.6）。检索轮数受预算限制（8.5：Fast 1 轮；Deep 3 轮，含转入前 Fast 已用的；Verified 补检索 1 轮），
   用尽仍有缺口则拒答。
结果写入 query_log（log_query），记录授权决定、模型标识、索引构建编号、成本与 Deep 的计划和动作（10.1 第 8 步、12.2）。

执行结构（8.4）：第 1 步 prepare（时间点与 Router）在任务建立之前完成；其余是 LangGraph 图的四个节点
first（首轮：Fast 或 Deep 的检索、生成与引用检查）→ escalate（Fast 转入 Deep）→ verify（转入 Verified 与验证）
→ assemble（组装）。状态只含 JSON 值（codec.py），接上检查点后每个节点之后写入 PostgreSQL，进程重启后从最后一个
检查点继续（异步执行见 tasks.py）；ask() 不接检查点，同步跑完整张图，供评测与测试使用。
"""
from __future__ import annotations

import operator
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Annotated, Any, TypedDict

import psycopg
from langgraph.graph import END, START, StateGraph
from psycopg.types.json import Jsonb

from eaer.adapters.model import Message, ModelClient
from eaer.adapters.online import collect_calls, summarize
from eaer.domain.identity import Scope
from eaer.domain.injection import suspicious
from eaer.orchestrator import codec
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
DEFAULT_BUDGETS = {"fast_rounds": 1, "deep_rounds": 3, "verified_repair_rounds": 1, "sync_seconds": 30}
NODES = ("first", "escalate", "verify", "assemble")
NODE_NAMES = {"first": "检索与生成", "escalate": "升级判断", "verify": "验证", "assemble": "组装回答"}


class QueryState(TypedDict, total=False):
    """图的状态（8.4 的 QueryState）。只含 JSON 值；授权范围不在其中，由执行者每次运行时传入。"""
    request: dict                      # question、today、as_of_valid、as_of_system、mode_hint
    tb: dict
    route: dict
    deep_off: str | None
    notices: list[str]
    mode_path: list[str]
    escalations: list[str]
    hits: list[dict]
    meta: dict
    draft: dict | None
    claims: list[dict]
    rounds: int
    plan: dict | None
    version_targets: dict
    unresolved: list[str]
    forced: list[str]
    signals: list[str]
    alerts: list[str]
    answer: str
    abstained: bool
    calls: Annotated[list[dict], operator.add]     # 模型调用记录，跨节点、跨进程累加
    steps: Annotated[list[dict], operator.add]     # 已完成的步骤与耗时


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
        self.scope: Scope | None = None        # 本次运行的授权范围：由调用方传入，不进入状态与检查点

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

    def prepare(self, question: str, *, today: date, as_of_valid: date | None = None,
                as_of_system: date | None = None, mode_hint: str | None = None) -> QueryState:
        """第 1 步：时间点（7.5）与 Router（8.1）。结果是图的输入，也决定提问记录的入口模式。"""
        t0 = time.monotonic()
        with collect_calls() as calls:
            tb = resolve(question, today=today, as_of_valid=as_of_valid, as_of_system=as_of_system,
                         resolver=self._resolve_with_model)
        route = decide(question, tb, mode_hint)
        deep_off = self.deep_unavailable()
        notices = tb.echo()
        start = route.mode
        if start == "deep" and deep_off:
            notices.append(f"这个问题需要 Deep 模式（{route.reason}），{deep_off}，先按 Fast 检索回答")
            start = "fast"
        return {"request": {"question": question, "today": today.isoformat(), "as_of_valid": codec.iso(as_of_valid),
                            "as_of_system": codec.iso(as_of_system), "mode_hint": mode_hint},
                "tb": tb.to_json(), "route": route.to_json(), "deep_off": deep_off, "notices": notices,
                "mode_path": [start], "escalations": [], "rounds": 0, "calls": codec.calls_out(calls),
                "steps": [{"node": "prepare", "seconds": round(time.monotonic() - t0, 3)}]}

    def graph(self, checkpointer=None):
        """四个节点的图；传入检查点时每个节点之后写入（tasks.py 用 PostgresSaver）。"""
        g = StateGraph(QueryState)
        for name in NODES:
            g.add_node(name, self._node(name, getattr(self, f"_{name}")))
        g.add_edge(START, NODES[0])
        for a, b in zip(NODES, NODES[1:]):
            g.add_edge(a, b)
        g.add_edge(NODES[-1], END)
        return g.compile(checkpointer=checkpointer)

    def ask(self, scope: Scope, question: str, *, today: date, as_of_valid: date | None = None,
            as_of_system: date | None = None, mode_hint: str | None = None,
            session_id: str | None = None) -> QueryResult:
        """同步跑完整次提问（不接检查点）。"""
        self.scope = scope
        st = self.prepare(question, today=today, as_of_valid=as_of_valid, as_of_system=as_of_system,
                          mode_hint=mode_hint)
        return self.result(self.graph().invoke(st), session_id or str(uuid.uuid4()))

    def result(self, st: dict[str, Any], session_id: str, status: str = "completed") -> QueryResult:
        draft = codec.draft_in(st.get("draft"))
        escalations = st.get("escalations", [])
        return QueryResult(
            session_id, status, st["mode_path"][-1], st["mode_path"], st["route"], st["tb"], st.get("answer"),
            codec.claims_in(st.get("claims")), bool(st.get("abstained")), st.get("notices", []), st.get("forced", []),
            escalations, st.get("rounds", 0), draft.model if draft else model_id(self.model), self._build_id(),
            cost=summarize(codec.calls_in(st.get("calls"))),
            seconds=round(sum(x["seconds"] for x in st.get("steps", [])), 3), alerts=st.get("alerts", []),
            signals=st.get("signals", []) + [e for e in escalations if "Deep" in e], plan=st.get("plan"))

    # ---------- 节点 ----------

    def _node(self, name: str, fn):
        def run(st: QueryState) -> dict:
            t0 = time.monotonic()
            with collect_calls() as calls:
                out = fn(st)
            return {**out, "calls": codec.calls_out(calls),
                    "steps": [{"node": name, "seconds": round(time.monotonic() - t0, 3)}]}
        return run

    def _ctx(self, st: QueryState):
        req = st["request"]
        today = date.fromisoformat(req["today"])
        tb = codec.tb_in(st["tb"])
        on = tb.as_of_valid or tb.as_of_system or today
        return req["question"], today, tb, on, target_of(on, tb.as_of_system)

    def _deep_out(self, d: DeepOutcome, target: Target) -> dict:
        vt = d.version_targets
        claims = self.steps.check(self.scope, d.draft.claims, d.hits, d.meta, target, vt)
        return {"hits": codec.hits_out(d.hits), "meta": codec.meta_out(d.meta), "draft": codec.draft_out(d.draft),
                "claims": codec.claims_out(claims), "rounds": d.rounds, "plan": d.trace(),
                "version_targets": codec.targets_out(vt), "unresolved": d.unresolved}

    def _first(self, st: QueryState) -> dict:
        s = self.steps
        question, today, tb, on, target = self._ctx(st)
        if st["mode_path"][0] == "deep":
            return self._deep_out(self._deep().run(self.scope, question, tb, today=today, target=target), target)
        hits = s.retrieve(self.scope, question, today=today, as_of_valid=tb.as_of_valid, as_of_system=tb.as_of_system)
        meta = s.meta(hits, target.system_t)
        draft = s.generate(question, hits, meta, on) if hits else None
        claims = s.check(self.scope, draft.claims, hits, meta, target) if draft else []
        return {"hits": codec.hits_out(hits), "meta": codec.meta_out(meta), "draft": codec.draft_out(draft),
                "claims": codec.claims_out(claims), "rounds": 1, "version_targets": {}, "unresolved": []}

    def _escalate(self, st: QueryState) -> dict:
        """Fast 检索后判断是否转入 Deep（8.2）。"""
        if st["mode_path"] != ["fast"]:
            return {}
        hits, draft = codec.hits_in(st["hits"]), codec.draft_in(st["draft"])
        signals = self._deep_signals(hits, codec.claims_in(st["claims"]), draft)
        if not signals:
            return {}
        if st.get("deep_off"):
            return {"escalations": st["escalations"] + ["应转入 Deep：" + "、".join(signals) + f"（{st['deep_off']}）"]}
        question, today, tb, _, target = self._ctx(st)
        d = self._deep().run(self.scope, question, tb, today=today, target=target, template="multihop", seed=hits,
                             seed_missing=draft.missing if draft else None, rounds_used=st["rounds"])
        return {**self._deep_out(d, target), "mode_path": st["mode_path"] + ["deep"],
                "escalations": st["escalations"] + ["转入 Deep：" + "、".join(signals)]}

    def _verify(self, st: QueryState) -> dict:
        """Fast 或 Deep 检索后判断是否转入 Verified（8.1、8.2），需要时做语义检查与定向补检索（9.3、9.4）。"""
        question, today, tb, _, target = self._ctx(st)
        hits, meta, draft = codec.hits_in(st["hits"]), dict(st["meta"]), codec.draft_in(st["draft"])
        claims = codec.claims_in(st["claims"])
        vt = codec.targets_in(st.get("version_targets"))
        forced, signals = self._verified_signals(self.scope, question, tb, claims, draft, meta, hits)
        mode_path, escalations, notices = list(st["mode_path"]), list(st["escalations"]), list(st["notices"])
        rounds = st["rounds"]
        alerts = [f"安全告警：授权核对未通过，已移除证据 {i}" for c in claims for i in c.removed]
        need = mode_path[-1] == "verified" or bool(signals)
        if need and not self.model.capabilities().structured_output:
            notices.append("验证模式不可用（所用模型不支持结构化输出），时效性和强制验证类结论标为“未验证”")
            for c in claims:
                if c.status == "supported":
                    c.status = "unverified"
        elif need:
            if mode_path[-1] != "verified":
                mode_path.append("verified")
                escalations.append("转入 Verified：" + "、".join(signals))
            rounds += self._repair(self.scope, question, claims, hits, meta, target, today, tb, vt)
        return {"claims": codec.claims_out(claims), "meta": codec.meta_out(meta), "forced": forced, "signals": signals,
                "mode_path": mode_path, "escalations": escalations, "notices": notices, "rounds": rounds,
                "alerts": alerts}

    def _recheck(self, claims: list[ClaimOut]) -> list[str]:
        """返回前再核对一次所引证据的权限与状态（10.1）：转入后台的提问可能在几分钟后才组装，期间权限可能被撤销、
        文档可能被归档。核对不通过的证据移除；结论一条证据都不剩时改为 unsupported。返回告警。"""
        ids = list(dict.fromkeys(x.evidence_id for c in claims if c.status != "unsupported" for x in c.citations))
        if not ids:
            return []
        allowed = dict(zip(ids, self.authorizer.check(self.scope, ids)))
        alerts = []
        for c in claims:
            gone = [x.evidence_id for x in c.citations if c.status != "unsupported" and not allowed[x.evidence_id]]
            if not gone:
                continue
            c.citations = [x for x in c.citations if allowed[x.evidence_id]]
            c.removed = c.removed + gone
            alerts += [f"返回前核对未通过（权限或状态已变化），已移除证据 {i}" for i in gone]
            if not c.citations:
                c.status = "unsupported"
        return alerts

    def _assemble(self, st: QueryState) -> dict:
        claims, draft = codec.claims_in(st["claims"]), codec.draft_in(st["draft"])
        alerts = st.get("alerts", []) + self._recheck(claims)
        text, abstained = Steps.assemble(claims, draft)
        notices = list(st["notices"])
        if st.get("unresolved") and not abstained:
            notices.append("以下要点没有找到证据，回答未涉及：" + "；".join(st["unresolved"]))
        if not st["hits"] and draft is None:
            text = ABSTAIN
        if notices and not abstained:
            echo = [n for n in notices if n.startswith("按")]
            text = "；".join(echo) + ("。" if echo else "") + text
        return {"answer": text, "abstained": abstained, "notices": notices, "claims": codec.claims_out(claims),
                "alerts": alerts}

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
                          meta: dict, hits=()) -> tuple[list[str], list[str]]:
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
        content = {h.evidence_id: h.content for h in hits}
        if any(suspicious(content.get(x.evidence_id, "")) for c in claims for x in c.citations):
            signals.append("引用的证据含疑似指令文字")
        return forced, signals

    def _repair(self, scope, question, claims: list[ClaimOut], hits, meta, target: Target, today, tb: TimeBasis,
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
            meta.update(codec.meta_out(self.steps.meta([h for h in new if h.version_id not in meta], target.system_t)))
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


def log_fields(r: QueryResult, scope: Scope) -> dict:
    """query_log 中由执行结果决定的字段（10.1 第 8 步、10.4 审计）。"""
    authz = {"principals": sorted(scope.principals),
             "expires_at": scope.expires_at.isoformat() if scope.expires_at else None,
             "removed_evidence": [i for c in r.claims for i in c.removed], "alerts": r.alerts}
    claims = [{"text": c.text, "status": c.status, "core": c.core, "citations": [asdict(x) for x in c.citations],
               "checks": c.checks, "semantic": c.semantic, "repaired": c.repaired} for c in r.claims]
    for name, value in (("session_id", r.session_id), ("mode", r.mode), ("model_id", r.model),
                        ("index_build_id", r.index_build_id)):
        if not value:
            raise ValueError(f"query_log 缺少必填字段 {name}")
    return {"time_basis": Jsonb(r.time_basis), "mode": r.mode, "mode_path": r.mode_path, "status": r.status,
            "answer": r.answer, "claims": Jsonb(claims), "evidence_ids": r.evidence_ids, "abstained": r.abstained,
            "authz_decision": Jsonb(authz), "model_id": r.model, "index_build_id": r.index_build_id,
            "cost": Jsonb({**r.cost, "retrieval_rounds": r.retrieval_rounds, "seconds": r.seconds,
                           "router": r.router, "escalations": r.escalations, "signals": r.signals,
                           "forced_verify": r.forced_verify, "plan": r.plan})}


def log_query(conn: psycopg.Connection, r: QueryResult, *, scope: Scope, user_id: str, question: str,
              asked_at: datetime | None = None, query_id: uuid.UUID | None = None) -> uuid.UUID:
    """写入 query_log。必填字段缺一即报错，不写半条记录（AC-10-9）。"""
    if not user_id or not question:
        raise ValueError("query_log 缺少必填字段 user_id 或 question")
    f = log_fields(r, scope)
    qid = query_id or uuid.uuid4()
    conn.execute(
        "INSERT INTO query_log (id, tenant_id, session_id, user_id, asked_at, question, "
        + ", ".join(f) + ") VALUES (%s, %s, %s, %s, %s, %s, " + ", ".join(["%s"] * len(f)) + ")",
        (qid, scope.tenant_id, uuid.UUID(r.session_id), user_id, asked_at or datetime.now(timezone.utc), question,
         *f.values()))
    return qid
