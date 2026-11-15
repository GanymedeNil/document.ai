"""MultiHop-RAG 评测（开发文档 14.1：可对外，ODC-BY）：英文检索召回（AC-5-4）与拒答（AC-8-8）。

    python -m eaer.evalrun.datasets fetch multihop-rag            # 下载（固定提交号，核对 SHA-256）
    EAER_ENV=eval python -m eaer.evalrun.multihop --admin-url postgresql://… --opensearch http://localhost:9200 \\
        [--answer-sample 50] [--out 报告.md]

- 语料：609 篇英文新闻，按生产流程入库（接收、文本解析、审核通过为记录类、写入索引；真实 OpenSearch 与 Embedding）。
  每篇写成 HTML：标题、来源与日期一行，正文按空行分段；来源与发布日期另作为数据源给出的出处与文档日期
  （receive 的 publisher、doc_date）随文档接收，与生产中数据源给出元数据走同一条路，证据标注据此给出“文档日期”“出处”。
  问题常按“某来源某日的报道”指认文章（比较、时间两类），2026-11-15 之前没有传，这两类大多拒答。
- 检索召回（Hybrid Recall@10，AC-5-4 的英文部分）：对 2,255 个有答案的问题，依据文章出现在前 10 条结果里的比例
  （按依据计）与全部依据都命中的问题比例（按问题计）；混合检索不重排、重排后各算一遍。
- 拒答（AC-8-8）：经完整编排器（DeepSeek）回答抽样的无答案问题与有答案问题：无答案的应拒答，有答案的不应拒答，
  报告拒答的准确率与召回率。答对有两种算法：标准答案原文出现在回答里（只适用于人名、机构名这类短答案）；
  生成模型按问题与标准答案评判（与 ViDoRe 相同的评判提示）——比较、时间两类的标准答案多是 Yes/No，系统给的是带引用的
  陈述而不是“是/否”，只能这样判。评判模型与回答模型同一家，可能偏宽。
  `rejudge 答案.json` 可对已有的回答重新评判，不必重新入库。
- 重排：重排调用失败（超时等）的问题记为失败并跳过，不中断整个阶段；`--rerank-sample N` 只对抽样的 N 个问题重排。
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
                        title=a["title"], doc_date=date.fromisoformat(a["published_at"][:10]), publisher=a["source"])
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


def recall(conn, retriever, queries: list[dict], docs: dict[str, str], tenant: str = "mh", *,
           sample: int = 0, failed: list | None = None) -> list[RecallRow]:
    """sample > 0 时只取固定种子抽样的这么多个问题；单个问题检索出错（重排超时等）记入 failed 并跳过。"""
    from eaer.retrieval.service import Scope

    scope = Scope(tenant, frozenset({ROLE}))
    items = [(i, q) for i, q in enumerate(queries) if q["question_type"] != "null_query"]
    if sample:
        items = sorted(random.Random(11).sample(items, min(sample, len(items))), key=lambda x: x[0])
    rows = []
    for i, q in items:
        want = {docs[e["title"]] for e in q["evidence_list"]}
        try:
            got = {h.document_id for h in retriever.search(scope, q["query"], today=TODAY)}
        except Exception as e:  # noqa: BLE001 — 记下失败继续
            if failed is None:
                raise
            failed.append((i, f"{type(e).__name__}: {e}"[:200]))
            continue
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


def rejudge(model, rows: list[dict], log=lambda s: None) -> list[dict]:
    """有答案的问题交给生成模型按标准答案评判（judged 为 correct / partial / incorrect；拒答与出错直接算 incorrect）。"""
    from eaer.evalrun.vidore import judge
    out = []
    for k, r in enumerate(rows):
        r = dict(r)
        if r["type"] != "null_query":
            if r["abstained"] is False:
                v = judge(model, r["query"], str(r["gold"]), r["answer"])
                r["judged"], r["judge_reason"] = v["verdict"], v.get("reason", "")
            else:
                r["judged"], r["judge_reason"] = "incorrect", "拒答或出错"
        out.append(r)
        if (k + 1) % 20 == 0:
            log(f"已评判 {k + 1} / {len(rows)}")
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
            "judged_accuracy": round(sum(r.get("judged") == "correct" for r in ans) / max(1, len(ans)), 4)
            if any("judged" in r for r in ans) else None,
            "by_type": {t: {"n": len(rs), "refused": sum(bool(r["abstained"]) for r in rs),
                            "substring": sum(r["correct"] for r in rs),
                            "judged": sum(r.get("judged") == "correct" for r in rs)}
                        for t in sorted({r["type"] for r in ans}) for rs in [[r for r in ans if r["type"] == t]]},
            "errors": sum(r["abstained"] is None for r in rows)}


def report(rec_hybrid: dict, rec_rerank: dict | None, ans: dict | None, *, embedder: str, model: str | None,
           rerank_note: str = "") -> str:
    lines = [f"MultiHop-RAG 评测：609 篇英文新闻；Embedding {embedder}；分词 standard；真实 OpenSearch"]
    if rec_hybrid:
        lines += ["", "检索召回（前 10 条，按依据计 / 全部依据都命中的问题比例）：", "",
                  "| 问题类型 | 问题数 | 混合检索 | 混合检索 + 重排 |", "| --- | --- | --- | --- |"]
    for k, v in (rec_hybrid or {}).items():
        r = rec_rerank.get(k) if rec_rerank else None
        fmt = lambda x: f"{x['evidence_recall']:.1%} / {x['all_found']:.1%}"  # noqa: E731
        lines.append(f"| {'全部' if k == 'all' else k} | {v['queries']} | {fmt(v)} | {fmt(r) if r else '—'} |")
    if rerank_note:
        lines += ["", rerank_note]
    if ans:
        lines += ["", f"拒答与回答（经完整编排器，生成 {model}；抽样 无答案 {ans['null']} 个、有答案 {ans['answerable']} 个）：",
                  f"- 拒答召回（无答案的拒答了）{ans['refusal_recall']:.1%}，拒答准确率（拒答的确实无答案）{ans['refusal_precision']:.1%}；",
                  f"- 有答案却拒答 {ans['false_refusal']:.1%}；有答案的答对：标准答案原文出现在回答里 {ans['answer_accuracy']:.1%}"
                  + (f"，生成模型按标准答案评判 {ans['judged_accuracy']:.1%}" if ans.get("judged_accuracy") is not None else "")
                  + f"；出错 {ans['errors']} 次。", "",
                  "| 问题类型 | 问题数 | 拒答 | 答对（原文出现） | 答对（评判） |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| {t} | {v['n']} | {v['refused']} | {v['substring']} | {v['judged']} |" for t, v in ans["by_type"].items()]
        lines += ["", "比较、时间两类的标准答案多是 Yes/No，原文出现的算法对它们不适用，以评判为准；评判模型与回答模型同一家，可能偏宽。"]
    return "\n".join(lines)


def rejudge_main(argv: list[str]) -> int:
    """对已有的回答文件（运行时 --out 旁边的 .answers.json）重新评判，结果写回该文件，报告只含拒答与回答部分。"""
    from eaer.adapters.online import OnlineModelClient
    from eaer.evalrun.license import add_external_flag, emit, refuse_early

    p = argparse.ArgumentParser(prog="eaer.evalrun.multihop rejudge", description="MultiHop-RAG：对已有的回答重新评判")
    p.add_argument("answers")
    p.add_argument("--data", default=str(DATASET_DIR))
    p.add_argument("--out")
    add_external_flag(p)
    args = p.parse_args(argv)
    if os.environ.get("EAER_ENV") != "eval":
        print("只在评测环境运行（EAER_ENV=eval）", file=sys.stderr)
        return 2
    _, _, prov = load(Path(args.data))
    if (code := refuse_early(prov, external=args.external)) is not None:
        return code
    model = OnlineModelClient.from_env()
    if model.gen is None:
        print("需要生成服务（EAER_GEN_*）", file=sys.stderr)
        return 2
    path = Path(args.answers)
    rows = rejudge(model, json.loads(path.read_text(encoding="utf-8")),
                   log=lambda s: print(s, file=sys.stderr, flush=True))
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    text = report({}, None, answer_summary(rows), embedder="（沿用原运行）", model=model.gen.model)
    return emit(text, prov, external=args.external, out=args.out)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "rejudge":
        return rejudge_main(argv[1:])
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
    p.add_argument("--rerank-sample", type=int, default=0, help="只对抽样的这么多个问题算重排后的召回（0 为全部）")
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
        log("混合检索：" + json.dumps(hybrid["all"], ensure_ascii=False))
        rerank, rerank_note = None, ""
        if not args.no_rerank and model.rr is not None:
            log("检索召回：混合检索 + 重排")
            failed: list = []
            rows_rr = recall(conn, Retriever(conn, index, embedder, model), queries, docs,
                             sample=args.rerank_sample, failed=failed)
            rerank = recall_summary(rows_rr)
            if args.rerank_sample:
                hybrid_same = recall_summary(recall(conn, Retriever(conn, index, embedder), queries, docs,
                                                    sample=args.rerank_sample))
                rerank_note = (f"重排只对抽样的 {args.rerank_sample} 个问题计算；同一批问题不重排时全部 "
                               f"{hybrid_same['all']['evidence_recall']:.1%} / {hybrid_same['all']['all_found']:.1%}。")
            if failed:
                rerank_note += f"重排调用失败、未计入的问题 {len(failed)} 个（如 {failed[0][1]}）。"
                log(rerank_note)
        ans = None
        if args.answer_sample:
            log(f"拒答与回答：各抽 {args.answer_sample} 个")
            rows = answer_sample(conn, Retriever(conn, index, embedder, model if model.rr else None), model, queries,
                                 args.answer_sample)
            rows = rejudge(model, rows, log=log)
            ans = answer_summary(rows)
            Path(tmp, "answers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            if args.out:
                Path(args.out).with_suffix(".answers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                                                      encoding="utf-8")
        conn.close()
    text = report(hybrid, rerank, ans, embedder=embedder.name, model=model.gen.model if model.gen else None,
                  rerank_note=rerank_note)
    return emit(text, prov, external=args.external, out=args.out)


if __name__ == "__main__":
    sys.exit(main())
