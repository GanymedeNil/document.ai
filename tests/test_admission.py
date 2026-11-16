"""AC-12-3 准入检查（开发文档 12.3）：用输出随机结果的验证模型跑准入，Verified 被阻止开启（硬性）。

- 验证器准入：标定集与评测集互不交叉（AC-9-5）；随机作答（换 40 个随机种子，置信度随机或一律很高）、一律判支持、
  一律判未验证的模型都不准入；按标签作答的模型准入，并标定出阈值；
- 部署时的开关：没有准入记录的模型 Verified、Deep 都关闭并说明原因；按最近一次准入结论开关；准入记录只追加，
  写入时清除该租户的回答缓存；EAER_ADMISSION=off 跳过；
- 关闭后的行为（12.1）：强制验证类问题标“未验证”并说明原因，需要 Deep 的问题由 Fast 回答并说明；
- 离线跑完整的准入流程（事实表语料、验证器、Deep 题、全部查询），产出能力矩阵并写入准入结论。
"""
from __future__ import annotations

import random
import re
from datetime import date, datetime, timezone

import pytest

from eaer.adapters.model import Capabilities, Completion, Usage
from eaer.evalrun.admission import (check_verifier, deep_correct, deep_items, matrix, split, verifier_items)
from eaer.orchestrator.admission import DEFAULT_CRITERIA, gate, latest, record
from factset.load import load
from test_engine import ANSWER_650, SCOPE, Scripted, eng  # noqa: F401
from test_ingestion import env, html  # noqa: F401
from test_retrieval import ix  # noqa: F401

FS = load("docs/事实表样例.xlsx")
ON = date(2026, 11, 11)
NOW = datetime(2026, 11, 13, tzinfo=timezone.utc)
# 同一句结论对不同证据可能一真一假（“一线城市住宿上限为600元/晚”对第 1 版是改数值，对第 2 版是复述），按（结论, 证据）取标签
LABEL = {(i.claim, i.evidence): i.supported for i in verifier_items(FS)}


def label_of(claim: str, prompt: str, default=None):
    m = re.search(r"\[ev_[0-9a-f]+\]\n([^\n]*)", prompt)
    unescape = str.maketrans("［］＜＞", "[]<>")
    return LABEL.get((claim.translate(unescape), m.group(1).translate(unescape) if m else ""), default)


class Judge:
    """只做语义检查的模型：verdict(claim) -> (结论, 置信度)。"""

    def __init__(self, verdict, structured=True):
        self.verdict, self.structured, self.calls = verdict, structured, 0     # verdict(claim, prompt)

    def capabilities(self):
        return Capabilities(self.structured, 32000, False, 0)

    def generate(self, messages, schema):
        self.calls += 1
        p = "\n".join(m.content for m in messages)
        claim = re.search(r"结论：(.*)", p).group(1)
        status, conf = self.verdict(claim, p)
        return Completion("{}", {"verdict": status, "confidence": conf,
                                 "supporting_ids": re.findall(r"\[(ev_[0-9a-f]+)\]", p)}, Usage(1, 1), "judge")


def random_judge(seed: int, confident: bool):
    rng = random.Random(seed)
    return Judge(lambda c, p: (rng.choice(["supported", "unsupported", "unverified"]),
                               0.99 if confident else rng.random()))


def oracle():
    return Judge(lambda c, p: ("supported" if label_of(c, p) else "unsupported", 0.95))


# ---------- 评测集 ----------

def test_items_and_independent_split():
    items = verifier_items(FS)
    calib, evaluation = split(items)
    assert {i.id for i in calib}.isdisjoint({i.id for i in evaluation})          # AC-9-5：无交叉
    assert len(calib) + len(evaluation) == len(items) == len({i.id for i in items})
    for part in (calib, evaluation):
        assert sum(i.supported for i in part) >= DEFAULT_CRITERIA["verified_min_positive"]
        assert sum(not i.supported for i in part) >= DEFAULT_CRITERIA["verified_min_negative"]
    assert {i.kind for i in items} == {"复述", "转述", "改数值", "换主体", "改单位", "改结论", "换版本"}
    assert split(items) == (calib, evaluation)                                     # 划分可复现
    qs = deep_items(FS)
    assert len(qs) >= DEFAULT_CRITERIA["deep_min_questions"] and {q.kind for q in qs} == {"修订前后", "版本号", "多对象"}
    assert all("第二季度差旅总费用" not in q.question for q in qs)                 # 图表事实不算 Deep 的题


def test_deep_scoring():
    assert deep_correct("由500元/晚调整为600元/晚。", False, ("500元/晚", "600元/晚"))
    assert not deep_correct("调整为600元/晚。", False, ("500元/晚", "600元/晚"))
    assert not deep_correct("500元/晚、600元/晚", True, ("500元/晚", "600元/晚"))
    assert deep_correct("上限为100,000元", False, ("100000元",))


# ---------- AC-12-3（硬性）：随机验证模型被阻止 ----------

@pytest.mark.parametrize("confident", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_random_verifier_is_never_admitted(seed, confident):
    r = check_verifier(random_judge(seed, confident), FS, DEFAULT_CRITERIA, ON)
    assert r["ok"] is False and ("误通过率" in r["reason"] or "误拒率" in r["reason"])


@pytest.mark.parametrize("verdict, why", [(("supported", 0.99), "误通过率"), (("unverified", 0.99), "误拒率"),
                                          (("unsupported", 0.99), "误拒率")])
def test_degenerate_verifiers_are_not_admitted(verdict, why):
    r = check_verifier(Judge(lambda c, p: verdict), FS, DEFAULT_CRITERIA, ON)
    assert r["ok"] is False and why in r["reason"]


def test_a_correct_verifier_is_admitted_and_calibrated():
    m = oracle()
    r = check_verifier(m, FS, DEFAULT_CRITERIA, ON)
    assert r["ok"] and r["threshold"] == 0.5 and r["evaluation"]["false_pass"] == 0 and r["evaluation"]["false_reject"] == 0
    assert m.calls == len(verifier_items(FS))                                      # 每条只问一次
    assert check_verifier(Judge(lambda c, p: ("supported", 0.9), structured=False), FS, DEFAULT_CRITERIA, ON)["reason"] \
        == "所用模型不支持结构化输出"


# ---------- 部署时的开关 ----------

def report(v_ok, d_ok):
    return {"verified": {"ok": v_ok, "reason": "误通过率 20% 超过 5%" if not v_ok else "达标"},
            "deep": {"ok": d_ok, "reason": "正确率 40% 低于 70%" if not d_ok else "达标"}}


def test_gate_follows_the_latest_admission(conn):
    g = gate(conn, "t1", "m1")
    assert not g.verified and not g.deep and "尚未通过准入检查" in g.reasons["verified"]
    assert gate(conn, "t1", "m1", required=False).verified
    record(conn, "t1", "m1", verified_ok=False, deep_ok=True, report=report(False, True), now=NOW,
           thresholds={"semantic_min_confidence": 0.8})
    g = gate(conn, "t1", "m1")
    assert not g.verified and g.deep and "误通过率 20%" in g.reasons["verified"] and "deep" not in g.reasons
    record(conn, "t1", "m1", verified_ok=True, deep_ok=True, report=report(True, True),
           now=datetime(2026, 11, 14, tzinfo=timezone.utc))
    assert gate(conn, "t1", "m1").verified and latest(conn, "t1", "m1")["verified_ok"]
    assert not gate(conn, "t1", "m2").verified and not gate(conn, "t2", "m1").verified   # 按模型、按租户
    th = conn.execute("SELECT thresholds FROM tenant_config WHERE tenant_id = 't1'").fetchone()["thresholds"]
    assert th["semantic_min_confidence"] == 0.8


def test_recording_an_admission_clears_the_tenant_cache(conn):
    conn.execute("INSERT INTO query_cache (key, tenant_id, scope_hash, index_build_id, result, created_at, expires_at) "
                 "VALUES ('k', 't1', 'h', 'b', '{}', now(), now() + interval '1 hour')")
    record(conn, "t1", "m1", verified_ok=True, deep_ok=True, report=report(True, True), now=NOW)
    assert conn.execute("SELECT count(*) AS n FROM query_cache").fetchone()["n"] == 0


def test_runtime_factory_applies_the_gate(conn, tmp_path):
    from eaer.runtime import engine_factory
    from eaer.storage import LocalFileStore

    env_ = {"EAER_OPENSEARCH_URL": "http://127.0.0.1:1", "EAER_GEN_BASE_URL": "http://127.0.0.1:1/v1",
            "EAER_GEN_API_KEY": "k", "EAER_GEN_MODEL": "gen-x", "EAER_EMBED_BASE_URL": "http://127.0.0.1:1/v1",
            "EAER_EMBED_API_KEY": "k", "EAER_EMBED_MODEL": "e", "EAER_EMBED_DIMS": "8",
            "EAER_RERANK_BASE_URL": "http://127.0.0.1:1/v1", "EAER_RERANK_API_KEY": "k", "EAER_RERANK_MODEL": "r"}
    make = engine_factory(env_, {}, LocalFileStore(tmp_path))
    e = make(conn, "t1")
    assert not e.verified_enabled and not e.deep_enabled and "gen-x 尚未通过准入检查" in e.closed_reasons["verified"]
    record(conn, "t1", "gen-x", verified_ok=True, deep_ok=False, report=report(True, False), now=NOW)
    e = make(conn, "t1")
    assert e.verified_enabled and not e.deep_enabled and "未通过 Deep 准入" in e.deep_unavailable()
    e = engine_factory({**env_, "EAER_ADMISSION": "off"}, {}, LocalFileStore(tmp_path))(conn, "t2")
    assert e.verified_enabled and e.deep_enabled


# ---------- 关闭后的行为（12.1） ----------

def test_closed_modes_degrade_with_an_explanation(eng):
    why = {"verified": "所用模型 m 尚未通过准入检查", "deep": "所用模型 m 尚未通过准入检查"}
    m = Scripted(ANSWER_650)
    r = eng.make(m, verified_enabled=False, deep_enabled=False, closed_reasons=why).ask(
        SCOPE, "一线城市住宿上限是多少", today=date(2026, 11, 11))
    assert r.mode_path == ["fast"] and r.answer.endswith("（未验证）") and "judge" not in m.calls
    assert any("验证模式不可用（所用模型 m 尚未通过准入检查）" in n for n in r.notices)
    r = eng.make(m, verified_enabled=False, deep_enabled=False, closed_reasons=why).ask(
        SCOPE, "2026 年 7 月修订前后住宿上限有什么变化", today=date(2026, 11, 11))
    assert r.router["decided"] == "deep" and r.mode_path == ["fast"]
    assert any("所用模型 m 尚未通过准入检查，先按 Fast 检索回答" in n for n in r.notices)


# ---------- 离线跑完整的准入流程 ----------

class Echo(Scripted):
    """生成时把看到的证据原文逐条当结论（数值都出自证据），语义检查按标签作答。"""

    def __init__(self, abstain=False):
        def draft(p):
            if abstain:
                return {"abstain": True, "missing": "无", "claims": []}
            pairs = re.findall(r"\[(ev_[0-9a-f]+)\][^\n]*\n([^\n]*)", p)[:6]
            return {"abstain": not pairs, "claims": [{"text": c, "evidence_ids": [i], "core": n == 0}
                                                     for n, (i, c) in enumerate(pairs)]}

        def judge(claim, prompt):
            ok = label_of(claim, prompt, default=True)
            return {"verdict": "supported" if ok else "unsupported", "confidence": 0.95,
                    "supporting_ids": re.findall(r"\[(ev_[0-9a-f]+)\]", prompt) if ok else []}

        super().__init__(draft, judge)


def run_offline(db_url, tmp_path, monkeypatch, model):
    import psycopg
    from psycopg.rows import dict_row

    from eaer.evalrun.admission import run_admission
    from eaer.retrieval.embed import HashingEmbedder
    from eaer.retrieval.index import MemoryIndex
    from eaer.storage import LocalFileStore
    from factset.render import Renderer

    monkeypatch.setenv("EAER_ENV", "eval")
    manifest = Renderer(FS).render(tmp_path / "corpus", "html")
    with psycopg.connect(db_url, autocommit=True, row_factory=dict_row) as c:
        return run_admission(c, FS, tmp_path / "corpus", manifest, model=model, store=LocalFileStore(tmp_path / "f"),
                             index=MemoryIndex(), embedder=HashingEmbedder())


def test_full_admission_run_produces_a_capability_matrix(db_url, make_db, tmp_path, monkeypatch, conn):
    rep = run_offline(db_url, tmp_path, monkeypatch, Echo())
    assert rep["verified"]["ok"] and rep["deep"]["questions"] == len(deep_items(FS))
    assert rep["queries"]["scored"] == 15 and rep["queries"]["by_path"]
    text = matrix(rep)
    assert text.startswith("# 能力矩阵") and "| Verified | 是 |" in text and "| Deep |" in text
    aid = record(conn, "t1", rep["model"]["generation"], verified_ok=rep["verified"]["ok"], deep_ok=rep["deep"]["ok"],
                 report=rep, now=NOW, thresholds={"semantic_min_confidence": rep["verified"]["threshold"]})
    assert latest(conn, "t1", "Echo")["id"] == aid


def test_a_model_that_cannot_answer_is_not_admitted_to_deep(make_db, tmp_path, monkeypatch):
    rep = run_offline(make_db(), tmp_path, monkeypatch, Echo(abstain=True))
    assert rep["deep"]["ok"] is False and rep["deep"]["accuracy"] == 0 and "正确率 0%" in rep["deep"]["reason"]
    assert "| Deep | 否 |" in matrix(rep)
