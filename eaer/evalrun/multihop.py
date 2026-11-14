"""MultiHop-RAG 评测（开发文档 14.1：可对外，ODC-BY）：英文检索召回（AC-5-4）与拒答（AC-8-8）。

    python -m eaer.evalrun.datasets fetch multihop-rag            # 下载（固定提交号，核对 SHA-256）
    EAER_ENV=eval python -m eaer.evalrun.multihop --admin-url postgresql://… --opensearch http://localhost:9200 \\
        [--answer-sample 50] [--out 报告.md]

- 语料：609 篇英文新闻，按生产流程入库（接收、文本解析、审核通过为记录类、写入索引；真实 OpenSearch 与 Embedding）。
  每篇写成 HTML：标题、来源与日期一行，正文按空行分段。
- 检索召回（Hybrid Recall@10，AC-5-4 的英文部分）：对 2,255 个有答案的问题，依据文章出现在前 10 条结果里的比例
  （按依据计）与全部依据都命中的问题比例（按问题计）；混合检索不重排、重排后各算一遍。
- 拒答（AC-8-8）：经完整编排器（DeepSeek）回答抽样的无答案问题与有答案问题：无答案的应拒答，有答案的不应拒答，
  答对按标准答案（多为人名、Yes/No 等短答案）是否出现在回答里算；报告拒答的准确率与召回率。
数据来源按下载文件的 SHA-256 判断（license.DATASET_FILES）；报告经 license.emit 加数据集标记与署名。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path

DATASET_DIR = Path(os.environ.get("EAER_DATASETS_DIR", Path.home() / ".cache" / "eaer" / "datasets")) / "multihop-rag"
TODAY = date(2024, 1, 31)                  # 语料的最晚日期之后
ROLE = "R_STAFF"


def load(root: Path = DATASET_DIR) -> tuple[list[dict], list[dict], list]:
    from eaer.evalrun.license import provenance_of_files
    corpus = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((root / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return corpus, queries, [provenance_of_files([root / "corpus.json", root / "MultiHopRAG.json"])]


def article_html(a: dict) -> bytes:
    paras = [p.strip() for p in re.split(r"\n\s*\n", a["body"]) if p.strip()]
    head = f"<h1>{escape(a['title'])}</h1><p>{escape(a['source'])}, {escape(a['published_at'][:10])}</p>"
    return (head + "".join(f"<p>{escape(' '.join(p.split()))}</p>" for p in paras)).encode("utf-8")


def ingest(conn, store, index, embedder, corpus: list[dict], tenant: str = "mh", log=lambda s: None) -> dict[str, str]:
    """全部文章入库并发布；返回 标题 → document_id。"""
    from eaer.ingestion.indexer import Indexer
    from eaer.ingestion.pipeline import Ingestion, Worker
    from eaer.lifecycle.service import Lifecycle
    from eaer.review.service import Review

    ing = Ingestion(conn, store)
    lc = Lifecycle(conn)
    lc.ensure_tenant(tenant)
    acl = lc.create_acl_policy(tenant, [ROLE])
    worker = Worker(ing, Indexer(conn, index, embedder, "mh"))
    docs = {}
    for k, a in enumerate(corpus):
        r = ing.receive(tenant, article_html(a), f"article-{k:04d}.html", acl_policy_id=acl, source_ref=f"mh-{k:04d}",
                        title=a["title"])
        docs[a["title"]] = str(r.document_id)
    worker.run_until_idle()
    tasks = conn.execute("SELECT id FROM review_task WHERE status = 'open'").fetchall()
    Review(conn, lc, store).decide(tenant, [{"task_id": str(t["id"]), "decision": "approve",
                                             "corrections": {"doc_type": "record"}} for t in tasks], "mh")
    log(f"入库：{len(corpus)} 篇，开始写入索引")
    worker.run_until_idle()
    return docs


@dataclass
class RecallRow:
    query_id: int
    qtype: str
    needed: int
    found: int


def recall(conn, retriever, queries: list[dict], docs: dict[str, str], tenant: str = "mh") -> list[RecallRow]:
    from eaer.retrieval.service import Scope

    scope = Scope(tenant, frozenset({ROLE}))
    rows = []
    for i, q in enumerate(queries):
        if q["question_type"] == "null_query":
            continue
        want = {docs[e["title"]] for e in q["evidence_list"]}
        got = {h.document_id for h in retriever.search(scope, q["query"], today=TODAY)}
        rows.append(RecallRow(i, q["question_type"], len(want), len(want & got)))
    return rows


def recall_summary(rows: list[RecallRow]) -> dict:
    by: dict[str, list[RecallRow]] = {}
    for r in rows:
        by.setdefault(r.qtype, []).append(r)

    def s(rs):
        return {"queries": len(rs), "evidence_recall": round(sum(r.found for r in rs) / max(1, sum(r.needed for r in rs)), 4),
                "all_found": round(sum(r.found == r.needed for r in rs) / max(1, len(rs)), 4)}
    return {"all": s(rows), **{k: s(v) for k, v in sorted(by.items())}}


def answer_sample(conn, retriever, model, queries: list[dict], n: int, *, seed: int = 7, tenant: str = "mh") -> list[dict]:
    """抽样：无答案问题 n 个、有答案问题 n 个（四类均分），经完整编排器回答。"""
    from eaer.adapters.authorizer import LocalAuthorizer
    from eaer.orchestrator.engine import Engine
    from eaer.retrieval.service import Scope

    rnd = random.Random(seed)
    null = [q for q in queries if q["question_type"] == "null_query"]
    others = [q for q in queries if q["question_type"] != "null_query"]
    pick = rnd.sample(null, min(n, len(null))) + rnd.sample(others, min(n, len(others)))
    scope = Scope(tenant, frozenset({ROLE}))
    out = []
    for q in pick:
        try:
            r = Engine(conn, retriever, model, LocalAuthorizer(conn, {})).ask(scope, q["query"], today=TODAY)
            answer, abstained, path = r.answer or "", r.abstained, r.mode_path
        except Exception as e:  # noqa: BLE001 — 记下错误继续
            answer, abstained, path = f"{type(e).__name__}: {e}", None, []
        gold = q["answer"]
        correct = (abstained is True) if q["question_type"] == "null_query" else (
            abstained is False and gold.lower().rstrip(".") in answer.lower())
        out.append({"type": q["question_type"], "query": q["query"], "gold": gold, "answer": answer, "abstained": abstained,
                    "correct": correct, "path": path})
    return out


def answer_summary(rows: list[dict]) -> dict:
    null = [r for r in rows if r["type"] == "null_query"]
    ans = [r for r in rows if r["type"] != "null_query"]
    refused = [r for r in rows if r["abstained"]]
    return {"null": len(null), "answerable": len(ans),
            "refusal_recall": round(sum(bool(r["abstained"]) for r in null) / max(1, len(null)), 4),
            "refusal_precision": round(sum(r["type"] == "null_query" for r in refused) / max(1, len(refused)), 4),
            "false_refusal": round(sum(bool(r["abstained"]) for r in ans) / max(1, len(ans)), 4),
            "answer_accuracy": round(sum(r["correct"] for r in ans) / max(1, len(ans)), 4),
            "errors": sum(r["abstained"] is None for r in rows)}


def report(rec_hybrid: dict, rec_rerank: dict | None, ans: dict | None, *, embedder: str, model: str | None) -> str:
    lines = [f"MultiHop-RAG 评测：609 篇英文新闻；Embedding {embedder}；分词 standard；真实 OpenSearch", "",
             "检索召回（前 10 条，按依据计 / 全部依据都命中的问题比例）：", "",
             "| 问题类型 | 问题数 | 混合检索 | 混合检索 + 重排 |", "| --- | --- | --- | --- |"]
    for k, v in rec_hybrid.items():
        r = rec_rerank.get(k) if rec_rerank else None
        fmt = lambda x: f"{x['evidence_recall']:.1%} / {x['all_found']:.1%}"  # noqa: E731
        lines.append(f"| {'全部' if k == 'all' else k} | {v['queries']} | {fmt(v)} | {fmt(r) if r else '—'} |")
    if ans:
        lines += ["", f"拒答与回答（经完整编排器，生成 {model}；抽样 无答案 {ans['null']} 个、有答案 {ans['answerable']} 个）：",
                  f"- 拒答召回（无答案的拒答了）{ans['refusal_recall']:.1%}，拒答准确率（拒答的确实无答案）{ans['refusal_precision']:.1%}；",
                  f"- 有答案却拒答 {ans['false_refusal']:.1%}；有答案的答对（标准答案出现在回答里）{ans['answer_accuracy']:.1%}；出错 {ans['errors']} 次。"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from eaer.adapters.online import OnlineModelClient
    from eaer.evalrun.analyzers import scratch_databases
    from eaer.evalrun.license import add_external_flag, emit, refuse_early
    from eaer.ingestion.indexer import fresh_index
    from eaer.retrieval.opensearch import OpenSearchIndex, connect
    from eaer.retrieval.service import Retriever
    from eaer.storage import LocalFileStore

    p = argparse.ArgumentParser(prog="eaer.evalrun.multihop", description="MultiHop-RAG：英文检索召回与拒答")
    p.add_argument("--admin-url", required=True)
    p.add_argument("--opensearch", required=True)
    p.add_argument("--data", default=str(DATASET_DIR))
    p.add_argument("--answer-sample", type=int, default=0, help="无答案、有答案各抽这么多个经编排器回答（调用生成服务）")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--out")
    add_external_flag(p)
    args = p.parse_args(argv)
    if os.environ.get("EAER_ENV") != "eval":
        print("只在评测环境运行（EAER_ENV=eval）", file=sys.stderr)
        return 2
    corpus, queries, prov = load(Path(args.data))
    if (code := refuse_early(prov, external=args.external)) is not None:
        return code
    model = OnlineModelClient.from_env()
    if model.emb is None:
        print("需要 Embedding 服务（EAER_EMBED_*）", file=sys.stderr)
        return 2
    embedder = model.embedder()
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    client, alias = connect(args.opensearch), f"eaer-mh-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as tmp, scratch_databases(args.admin_url) as new_db, \
            fresh_index(client, alias, build_id="mh", dims=model.embedding_dims):
        import psycopg
        from psycopg.rows import dict_row
        conn = psycopg.connect(new_db(), autocommit=True, row_factory=dict_row)
        index = OpenSearchIndex(client, alias=alias)
        docs = ingest(conn, LocalFileStore(Path(tmp)), index, embedder, corpus, log=log)
        log("检索召回：混合检索")
        hybrid = recall_summary(recall(conn, Retriever(conn, index, embedder), queries, docs))
        rerank = None
        if not args.no_rerank and model.rr is not None:
            log("检索召回：混合检索 + 重排")
            rerank = recall_summary(recall(conn, Retriever(conn, index, embedder, model), queries, docs))
        ans = None
        if args.answer_sample:
            log(f"拒答与回答：各抽 {args.answer_sample} 个")
            rows = answer_sample(conn, Retriever(conn, index, embedder, model if model.rr else None), model, queries,
                                 args.answer_sample)
            ans = answer_summary(rows)
            Path(tmp, "answers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            if args.out:
                Path(args.out).with_suffix(".answers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                                                      encoding="utf-8")
        conn.close()
    text = report(hybrid, rerank, ans, embedder=embedder.name, model=model.gen.model if model.gen else None)
    return emit(text, prov, external=args.external, out=args.out)


if __name__ == "__main__":
    sys.exit(main())
