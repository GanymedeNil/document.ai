"""AC-10-2（硬性）关系泄露：无权文档的存在不被透露（开发文档 10.2“派生关系”：冲突、取代提示暴露无权文档的存在
→ 关系两端都通过授权才返回；拒答文本不提及无权文档）。

做法是“两个世界”对照。同一份可见语料放进两个租户：
- t1 另有三份只给 R_FIN 的文件：Q09 那样只有它能回答的《采购审批权限规定》；跨文档取代了可见文件的
  《差旅住宿标准调整通知（财务内部）》（可见文件因此在 7 月 1 日作废）；与可见文件登记了冲突关系的《差旅费用口径说明》。
- t3 没有这些文件；可见文件的作废日由审核人按原文录入，与 t1 中被取代推出的作废日相同。
R_STAFF 用户在两个世界问同样的问题（Q09、直接点名无权文件、问已被无权文件取代的规定、问有冲突关系的规定、
Deep 版本比较、探测冲突），用户看到的一切——回答、结论、引用、冲突与升级信号、提示、拒答文本、Deep 计划——
以及模型看到的一切证据，在两个世界中必须完全相同：用户无从分辨无权文件是否存在。

模型取最坏情况：把看到的每条证据的标题、版本、有效期和原文都写进结论，列出全部证据互相“冲突”，
规划时专找无权文件。它的输出只取决于看到的证据集合（与顺序无关），所以两个世界的差别只可能来自它看到的东西。
另在 t1 中核对：无权文件里独有的编号与取值从未出现在发给模型的内容和回答里。

比较用内存索引：它的关键词打分只在过滤后的候选上统计。OpenSearch 的 BM25 统计量按分片上的全部文档计算
（含用户无权查看的），分数会有细微差别、但不返回任何无权内容，见设计变更记录 2026-11-14。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import date

import pytest

from eaer.adapters.model import Capabilities, Completion, Usage
from eaer.retrieval.service import Scope
from test_ingestion import env, html  # noqa: F401
from test_retrieval import ix  # noqa: F401

TODAY = date(2026, 11, 11)
MARKS = ("PX-7731", "SX-4402", "CX-9158")          # 只在无权文件里出现
SECRET_VALUES = ("50,000元", "720元", "680元")
WITH_SECRET, WITHOUT = "t1", "t3"

N = html("差旅报销管理办法", "第一条　为规范差旅报销，制定本办法。", "第二条　一线城市住宿上限为每晚650元。",
         "第三条　本办法自2026年1月1日起施行。")
M = html("出差伙食补贴标准", "第一条　出差伙食补贴每天80元。", "第二条　本标准自2025年2月1日起施行。")
P = html("采购审批权限规定", "第一条　部门经理单笔采购审批上限为50,000元。", f"第二条　内部编号 {MARKS[0]}。",
         "第三条　本规定自2026年3月1日起施行。")
S = html("差旅住宿标准调整通知（财务内部）", "一、一线城市住宿上限调整为每晚720元。", f"二、内部编号 {MARKS[1]}。",
         "三、本通知自2026年7月1日起施行。")
C = html("差旅费用口径说明（财务内部）", "一、一线城市住宿上限按每晚680元执行。", f"二、内部编号 {MARKS[2]}。",
         "三、本说明自2026年1月1日起施行。")

QUESTIONS = [
    "部门经理单笔采购的审批上限现在是多少？",                     # Q09：答案只在无权文件里
    "采购审批权限规定是哪天开始施行的？",                         # 直接点名无权文件
    "一线城市住宿上限现在是多少？",                               # 可见文件已被无权文件取代
    "2026 年 3 月 15 日一线城市住宿上限是多少？",                  # 取代之前；可见文件另与无权文件有冲突关系
    "差旅报销管理办法修订前后一线城市住宿上限有什么变化？",           # Deep 比较：只取有权查看的版本
    "一线城市住宿上限有没有其他文件规定了不同的标准？",              # 探测冲突
    "出差伙食补贴每天多少钱？",                                   # 对照：与无权文件无关
]

BLOCK = re.compile(r"<<<证据开始 (\w+)>>>\n\[(ev_[0-9a-f]{16})\] ([^\n]*)\n(.*?)\n<<<证据结束 \1>>>", re.S)
ANY_ID = re.compile(r"ev_[0-9a-f]{16}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class Echo:
    """最坏情况的模型：看到的证据全部写进结论、全部列为互相冲突，规划与补检索专找无权文件。
    输出只取决于提示里的证据集合（按内容排序，与检索顺序无关）。"""

    def __init__(self):
        self.prompts: list[str] = []

    def capabilities(self):
        return Capabilities(True, 128000, False, 0)

    def generate(self, messages, schema):
        p = "\n".join(m.content for m in messages)
        self.prompts.append(p)
        seen = sorted({(m.group(3), m.group(4), m.group(2)) for m in BLOCK.finditer(p)})
        ids = [eid for _, _, eid in seen]
        props = (schema or {}).get("properties", {})

        def out(parsed):
            return Completion(json.dumps(parsed, ensure_ascii=False), parsed, Usage(10, 10), "echo")

        if schema is None:
            return Completion("；".join(f"{h} {c}[{i}]" for h, c, i in seen) or "没有资料。", None, Usage(10, 10), "echo")
        if "verdict" in props:
            return out({"verdict": "supported", "confidence": 1.0, "supporting_ids": ids,
                        "reason": "；".join(h for h, _, _ in seen)})
        if "as_of_valid" in props:
            return out({"as_of_valid": ""})
        if "field" in props:
            return out({"field": "一线城市住宿上限", "objects": [
                {"label": "采购审批权限规定", "query": "采购审批权限规定 部门经理单笔采购审批上限"},
                {"label": "财务内部通知", "query": "差旅住宿标准调整通知 财务内部 一线城市住宿上限"}]})
        if "first_hop" in props:
            return out({"first_hop": "采购审批权限规定 财务内部"})
        if "phenomenon" in props:
            return out({"phenomenon": "一线城市住宿上限", "change_query": "差旅住宿标准调整通知 财务内部"})
        if "closed" in props:
            return out({"closed": False, "next_query": "差旅费用口径说明 财务内部 一线城市住宿上限", "missing": "财务内部文件"})
        if "queries" in props:
            return out({"queries": [{"label": lb, "query": f"财务内部 {lb}"}
                                    for lb in re.findall(r"- (.*?)：已用检索词", p)]})
        if not seen:
            return out({"abstain": True, "claims": [], "conflicts": [], "missing": "没有看到证据"})
        return out({"abstain": False, "conflicts": [ids] if len(ids) > 1 else [],
                    "missing": "；".join(h for h, _, _ in seen),
                    "claims": [{"text": f"{h}：{c}", "evidence_ids": [i], "core": k == 0}
                               for k, (h, c, i) in enumerate(seen)]})


def publish(e, tenant: str, acl, data: bytes, name: str, ref: str, **corrections):
    r = e.ing.receive(tenant, data, name, acl_policy_id=acl, source_ref=ref)
    e.worker.run_until_idle()
    task = str(e.conn.execute("SELECT id FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()["id"])
    res = e.review.decide(tenant, [{"task_id": task, "decision": "approve",
                                    "corrections": {"doc_type": "normative", **corrections}}], "alice")
    assert res[0]["ok"], res
    e.worker.run_until_idle()
    return r


@pytest.fixture
def worlds(ix):
    """t1：可见文件 + 三份 R_FIN 文件（含跨文档取代与冲突关系）；t3：只有可见文件。"""
    n1 = publish(ix, WITH_SECRET, ix.acl, N, "n.html", "N")
    publish(ix, WITH_SECRET, ix.acl, M, "m.html", "M")
    fin = ix.lc.create_acl_policy(WITH_SECRET, ["R_FIN"])
    publish(ix, WITH_SECRET, fin, P, "p.html", "P")
    publish(ix, WITH_SECRET, fin, S, "s.html", "S", relations=[
        {"relation": "supersedes", "to_document_id": str(n1.document_id), "to_version_no": 1}])
    publish(ix, WITH_SECRET, fin, C, "c.html", "C", relations=[
        {"relation": "conflicts", "to_document_id": str(n1.document_id), "to_version_no": 1}])

    ix.lc.ensure_tenant(WITHOUT)
    staff = ix.lc.create_acl_policy(WITHOUT, ["R_STAFF"])
    publish(ix, WITHOUT, staff, N, "n.html", "N", valid_to={"value": "2026-07-01"})   # 原文写明的作废日，与 t1 推出的相同
    publish(ix, WITHOUT, staff, M, "m.html", "M")

    def validity(tenant):
        return ix.conn.execute(
            "SELECT val.valid_from, val.valid_from_grade, val.valid_to, val.valid_to_grade FROM version_validity val "
            "JOIN document_version v ON v.id = val.version_id JOIN document d ON d.id = v.document_id "
            "WHERE d.tenant_id = %s AND d.source_ref = 'N' AND val.recorded_at IS NOT NULL AND val.replaced_at IS NULL",
            (tenant,)).fetchone()
    a, b = validity(WITH_SECRET), validity(WITHOUT)
    assert a == b and a["valid_to"] == date(2026, 7, 1), (a, b)                     # 两个世界里可见文件的有效期相同
    assert ix.conn.execute("SELECT count(*) AS n FROM version_relation r JOIN document_version v ON v.id = r.to_version_id "
                           "JOIN document d ON d.id = v.document_id WHERE d.source_ref = 'N' AND d.tenant_id = %s "
                           "AND r.recorded_at IS NOT NULL", (WITH_SECRET,)).fetchone()["n"] == 2
    return ix


def names(conn, tenant: str) -> dict[str, str]:
    """证据、版本、文档 id → 按内容命名，使两个世界可比（两边的 id 是各自随机生成的）。"""
    out = {}
    for r in conn.execute(
            "SELECT e.id, e.content, v.id AS vid, v.version_no, d.id AS did, d.title FROM evidence e "
            "JOIN document_version v ON v.id = e.version_id JOIN document d ON d.id = v.document_id "
            "WHERE d.tenant_id = %s", (tenant,)):
        out[r["id"]] = f"证据<{r['title']} 第{r['version_no']}版 {r['content']}>"
        out[str(r["vid"])] = f"版本<{r['title']} 第{r['version_no']}版>"
        out[str(r["did"])] = f"文档<{r['title']}>"
    return out


def canonical(obj, named: dict[str, str]):
    """id 换成内容名；列表排序（检索得分并列时按 id 排序，两边 id 不同，顺序不是存在性的信号）；去掉耗时等易变字段。"""
    if isinstance(obj, dict):
        return {k: canonical(v, named) for k, v in obj.items()
                if k not in {"seconds", "steps", "session_id", "cost", "calls"}}
    if isinstance(obj, list):
        return sorted((canonical(v, named) for v in obj), key=lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True))
    if isinstance(obj, str):
        return ANY_ID.sub(lambda m: named.get(m.group(0), "<其他 id>"), obj)
    return obj


def seen_by_model(prompts: list[str], named: dict[str, str]) -> list:
    """模型看到的内容：每个提示里的证据集合，加去掉证据区后的其余文字（证据区边界编号随机，不参与比较）。"""
    out = []
    for p in prompts:
        blocks = sorted(f"{named.get(m.group(2), '<其他 id>')} {m.group(3)}\n{m.group(4)}" for m in BLOCK.finditer(p))
        rest = canonical(BLOCK.sub("<证据>", p), named)
        out.append((rest, blocks))
    return sorted(out)


def ask(worlds, tenant: str, question: str):
    model = Echo()
    r = worlds.make(model).ask(Scope(tenant, frozenset({"R_STAFF"})), question, today=TODAY)
    named = names(worlds.conn, tenant)
    return model, r, canonical({**asdict(r), "response": r.response()}, named), seen_by_model(model.prompts, named)


@pytest.fixture
def make(worlds):
    from eaer.adapters.authorizer import LocalAuthorizer
    from eaer.orchestrator.engine import Engine

    worlds.make = lambda model: Engine(worlds.conn, worlds.retriever, model, LocalAuthorizer(worlds.conn, {}),
                                       store=worlds.store)
    return worlds


@pytest.mark.parametrize("question", QUESTIONS)
def test_user_cannot_tell_whether_restricted_documents_exist(make, question):
    model_a, ra, view_a, seen_a = ask(make, WITH_SECRET, question)
    model_b, rb, view_b, seen_b = ask(make, WITHOUT, question)
    assert model_a.prompts, "模型没有被调用，比较没有意义"
    assert seen_a == seen_b, "两个世界里模型看到的内容不同"
    assert view_a == view_b, "两个世界里用户看到的回答不同"
    # 独立于对照的直接检查：无权文件独有的编号与取值不在任何提示与回答里
    text = json.dumps({**asdict(ra), "response": ra.response()}, ensure_ascii=False, default=str) + "".join(model_a.prompts)
    for secret in MARKS + SECRET_VALUES:
        assert secret not in text, secret


def test_the_comparison_detects_a_leak(make, monkeypatch):
    """对照本身要能发现泄露：让检索在 t1 漏掉权限过滤，比较必须失败。"""
    real = make.retriever.search

    def leaky(scope, *a, **kw):
        return real(Scope(scope.tenant_id, scope.principals | {"R_FIN"}), *a, **kw)

    monkeypatch.setattr(make.retriever, "search", leaky)
    _, _, view_a, seen_a = ask(make, WITH_SECRET, QUESTIONS[0])
    _, _, view_b, seen_b = ask(make, WITHOUT, QUESTIONS[0])
    assert (seen_a, view_a) != (seen_b, view_b)
