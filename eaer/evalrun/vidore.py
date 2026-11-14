"""ViDoRe V3 评测（开发文档 14.1：可对外，CC BY 4.0；AC-14-6 图表与表格问答正确率）。

    python -m eaer.evalrun.datasets fetch vidore-v3-hr        # HR 子集：14 份欧盟劳动力市场报告（1,110 页）与问题
    EAER_ENV=eval EAER_GEN_VISION=1 python -m eaer.evalrun.vidore --admin-url postgresql://… \\
        --opensearch http://localhost:9200 [--sample 30] [--out 报告.md]

- 文档：14 份 PDF 按生产流程入库（Docling 解析；生成服务能看图时深度解析读取图表），审核通过为记录类；
- 问题：英文问题中依据含图表（Chart）或表格（Table）的，按内容类型各抽 --sample 个，经完整编排器回答；
- 判定：参考答案是长句，用生成模型作评判（correct / partial / incorrect），提示里给问题、参考答案与系统回答，
  核心数值与事实一致才算 correct；系统拒答算 incorrect。评判模型与回答模型同一家，可能偏宽，数字仅作方向参考；
- 同时统计回答里引用了看图读出的证据（source=vision）的比例。
数据来源按下载文件的 SHA-256 判断（license.DATASET_FILES）；报告经 license.emit 加数据集标记与署名。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import uuid
from datetime import date
from pathlib import Path

DATASET_DIR = Path(os.environ.get("EAER_DATASETS_DIR", Path.home() / ".cache" / "eaer" / "datasets")) / "vidore-v3-hr"
TODAY = date(2026, 1, 31)
ROLE = "R_STAFF"
JUDGE_SCHEMA = {"type": "object", "required": ["verdict", "reason"], "properties": {
    "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]}, "reason": {"type": "string"}}}
JUDGE_SYSTEM = """你是严格的评卷人。根据参考答案判断系统回答是否答对了问题：
- correct：系统回答包含参考答案的核心事实与数值（数值必须一致，允许不同的措辞、单位写法与合理的四舍五入），且没有与之矛盾的内容；
- partial：只答对了一部分核心事实或数值，或者大体正确但有个别数值不一致；
- incorrect：核心事实或数值错误、与参考答案矛盾、答非所问，或者系统拒答。
只比较事实，不评价文风与长短。"""


def load(root: Path = DATASET_DIR):
    from eaer.evalrun.license import provenance_of_files
    meta = json.loads((root / "documents_metadata.json").read_text(encoding="utf-8"))
    queries = json.loads((root / "queries.json").read_text(encoding="utf-8"))
    pdfs = sorted((root / "pdfs").glob("*.pdf"))
    return meta, queries, pdfs, [provenance_of_files([*pdfs, root / "queries.json", root / "documents_metadata.json"])]


def pick(queries: list[dict], n: int, seed: int = 7) -> list[dict]:
    """英文问题中依据含图表、含表格的各抽 n 个（两者都含的算图表）。"""
    rnd = random.Random(seed)
    en = [q for q in queries if q["language"] == "english"]
    chart = [q for q in en if "Chart" in q["content_type"]]
    table = [q for q in en if "Table" in q["content_type"] and "Chart" not in q["content_type"]]
    return [dict(q, kind="chart") for q in rnd.sample(chart, min(n, len(chart)))] + \
           [dict(q, kind="table") for q in rnd.sample(table, min(n, len(table)))]


def ingest(conn, store, index, embedder, pdfs: list[Path], *, figure_reader=None, tenant: str = "vd", log=lambda s: None):
    from eaer.ingestion.indexer import Indexer
    from eaer.ingestion.pipeline import Ingestion, Worker
    from eaer.lifecycle.service import Lifecycle
    from eaer.review.service import Review

    ing = Ingestion(conn, store, figure_reader=figure_reader)
    lc = Lifecycle(conn)
    lc.ensure_tenant(tenant)
    acl = lc.create_acl_policy(tenant, [ROLE])
    worker = Worker(ing, Indexer(conn, index, embedder, "vd"))
    for p in pdfs:
        ing.receive(tenant, p.read_bytes(), p.name, acl_policy_id=acl, source_ref=p.stem)
    log(f"接收 {len(pdfs)} 份，开始解析")
    worker.run_until_idle()
    tasks = conn.execute("SELECT id FROM review_task WHERE status = 'open'").fetchall()
    Review(conn, lc, store).decide(tenant, [{"task_id": str(t["id"]), "decision": "approve",
                                             "corrections": {"doc_type": "record"}} for t in tasks], "vd")
    log("审核通过，开始写入索引与深度解析（读取图表）")
    worker.run_until_idle()
    return {r["modality"]: r["n"] for r in conn.execute("SELECT modality, count(*) AS n FROM evidence GROUP BY modality")}


def judge(model, question: str, gold: str, answer: str) -> dict:
    from eaer.adapters.model import Message
    out = model.generate([Message("system", JUDGE_SYSTEM), Message(
        "user", f"问题：{question}\n\n参考答案：{gold}\n\n系统回答：{answer or '（拒答）'}")], JUDGE_SCHEMA)
    return out.parsed or {"verdict": "incorrect", "reason": "评判没有给出结果"}


def run_questions(conn, retriever, model, picked: list[dict], tenant: str = "vd", log=lambda s: None) -> list[dict]:
    from eaer.adapters.authorizer import LocalAuthorizer
    from eaer.orchestrator.engine import Engine
    from eaer.retrieval.service import Scope

    scope = Scope(tenant, frozenset({ROLE}))
    rows = []
    for k, q in enumerate(picked):
        try:
            r = Engine(conn, retriever, model, LocalAuthorizer(conn, {})).ask(scope, q["query"], today=TODAY)
            answer, abstained, path = r.answer or "", r.abstained, r.mode_path
            vision = any(c.locator.get("source") == "vision" for x in r.claims for c in x.citations)
        except Exception as e:  # noqa: BLE001
            answer, abstained, path, vision = f"{type(e).__name__}: {e}", None, [], False
        v = judge(model, q["query"], str(q["answer"]), "" if abstained else answer)
        rows.append({"query_id": q["query_id"], "kind": q["kind"], "query": q["query"], "gold": q["answer"],
                     "answer": answer, "abstained": abstained, "path": path, "cited_vision": vision,
                     "verdict": v["verdict"], "reason": v.get("reason", "")})
        if (k + 1) % 10 == 0:
            log(f"已回答 {k + 1} / {len(picked)}")
    return rows


def summary(rows: list[dict]) -> dict:
    out = {}
    for kind in ("chart", "table", "all"):
        rs = [r for r in rows if kind == "all" or r["kind"] == kind]
        n = max(1, len(rs))
        out[kind] = {"n": len(rs), "correct": sum(r["verdict"] == "correct" for r in rs) / n,
                     "partial": sum(r["verdict"] == "partial" for r in rs) / n,
                     "abstained": sum(bool(r["abstained"]) for r in rs) / n,
                     "cited_vision": sum(r["cited_vision"] for r in rs) / n}
    return out


def report(s: dict, evidence: dict, *, model: str, embedder: str, vision: bool) -> str:
    lines = [f"ViDoRe V3（HR 子集）：14 份英文报告，1,110 页；生成 {model}，Embedding {embedder}，看图 {'开' if vision else '关'}；"
             f"证据 {evidence}", "",
             "| 依据类型 | 问题数 | 答对 | 部分答对 | 拒答 | 回答引用了看图证据 |", "| --- | --- | --- | --- | --- | --- |"]
    for k, title in (("chart", "图表"), ("table", "表格（不含图表）"), ("all", "合计")):
        v = s[k]
        lines.append(f"| {title} | {v['n']} | {v['correct']:.1%} | {v['partial']:.1%} | {v['abstained']:.1%} | {v['cited_vision']:.1%} |")
    lines += ["", "说明：参考答案是长句，由生成模型按参考答案评判（与回答模型同一家，可能偏宽）；样本小，数字只作方向参考。"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from eaer.adapters.online import OnlineModelClient
    from eaer.evalrun.analyzers import scratch_databases
    from eaer.evalrun.license import add_external_flag, emit, refuse_early
    from eaer.ingestion.indexer import fresh_index
    from eaer.retrieval.opensearch import OpenSearchIndex, connect
    from eaer.retrieval.service import Retriever
    from eaer.storage import LocalFileStore

    p = argparse.ArgumentParser(prog="eaer.evalrun.vidore", description="ViDoRe V3：图表与表格问答正确率")
    p.add_argument("--admin-url", required=True)
    p.add_argument("--opensearch", required=True)
    p.add_argument("--data", default=str(DATASET_DIR))
    p.add_argument("--sample", type=int, default=30, help="图表、表格类问题各抽多少个")
    p.add_argument("--out")
    add_external_flag(p)
    args = p.parse_args(argv)
    if os.environ.get("EAER_ENV") != "eval":
        print("只在评测环境运行（EAER_ENV=eval）", file=sys.stderr)
        return 2
    meta, queries, pdfs, prov = load(Path(args.data))
    if (code := refuse_early(prov, external=args.external)) is not None:
        return code
    model = OnlineModelClient.from_env()
    if model.missing_for_fast():
        print(f"需要真实的模型服务：缺 {'、'.join(model.missing_for_fast())}", file=sys.stderr)
        return 2
    reader = None
    if model.vision:
        from eaer.ingestion.figures import FigureReader
        reader = FigureReader(model)
    embedder = model.embedder()
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    client, alias = connect(args.opensearch), f"eaer-vd-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as tmp, scratch_databases(args.admin_url) as new_db, \
            fresh_index(client, alias, build_id="vd", dims=model.embedding_dims):
        import psycopg
        from psycopg.rows import dict_row
        conn = psycopg.connect(new_db(), autocommit=True, row_factory=dict_row)
        index = OpenSearchIndex(client, alias=alias)
        evidence = ingest(conn, LocalFileStore(Path(tmp)), index, embedder, pdfs, figure_reader=reader, log=log)
        rows = run_questions(conn, Retriever(conn, index, embedder, model), model, pick(queries, args.sample), log=log)
        conn.close()
    if args.out:
        Path(args.out).with_suffix(".answers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    text = report(summary(rows), evidence, model=model.gen.model, embedder=embedder.name, vision=reader is not None)
    return emit(text, prov, external=args.external, out=args.out)


if __name__ == "__main__":
    sys.exit(main())
