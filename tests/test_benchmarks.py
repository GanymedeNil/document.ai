"""公开评测集的运行器（eaer/evalrun/datasets.py、multihop.py、vidore.py）与来源登记（license.provenance_of_files）。
离线：不下载、不调用模型服务；真实数据上的结果记入 CLAUDE.md。

- 来源由下载文件的 SHA-256 决定：登记过的文件属于对应数据集，改过一个字节或混了别的数据集的文件就是未登记来源；
- MultiHop-RAG：文章写成 HTML 的格式；三篇文章的小语料端到端（入库、发布、检索召回按依据与按问题统计）；
  拒答的准确率、召回率与答对率的算法；
- ViDoRe：按依据类型抽样（含图表的算图表、只含表格的算表格、只有英文）；评判结果的统计；
- 下载清单的提交号都是完整的 40 位。
"""
from __future__ import annotations

import hashlib
import shutil

from eaer.evalrun import datasets, license as lic, multihop, vidore
from eaer.evalrun.license import UNREGISTERED, provenance_of_files
from test_ingestion import env  # noqa: F401
from test_retrieval import ix  # noqa: F401


def test_downloaded_files_are_identified_by_their_checksum(tmp_path, monkeypatch):
    a, b, c = tmp_path / "corpus.json", tmp_path / "q.json", tmp_path / "other.json"
    a.write_bytes(b"[1]")
    b.write_bytes(b"[2]")
    c.write_bytes(b"[3]")
    monkeypatch.setattr(lic, "DATASET_FILES", {
        hashlib.sha256(b"[1]").hexdigest(): ("MultiHop-RAG", "corpus"), hashlib.sha256(b"[2]").hexdigest(): ("MultiHop-RAG", "q"),
        hashlib.sha256(b"[3]").hexdigest(): ("ViDoRe V3", "x")})
    assert provenance_of_files([a, b]).dataset == "MultiHop-RAG"
    assert provenance_of_files([a, c]).dataset == UNREGISTERED                     # 混了别的数据集
    edited = tmp_path / "edited.json"
    shutil.copy(a, edited)
    edited.write_bytes(b"[1] ")
    assert provenance_of_files([edited, b]).dataset == UNREGISTERED                # 改过一个字节
    assert all(n in lic.DATASETS for n, _ in lic.DATASET_FILES.values())


def test_pinned_revisions():
    assert all(len(s["revision"]) == 40 for s in datasets.SPECS.values())
    assert datasets.SPECS["llm-aggrefact"].get("gated")


CORPUS = [
    {"title": "Chip maker posts record revenue", "source": "Example News", "published_at": "2023-10-02T08:00:00+00:00",
     "body": "Acme Semiconductors reported record quarterly revenue of $4.2 billion.\n\nThe company credited demand for AI chips."},
    {"title": "Acme CEO steps down", "source": "Daily Tech", "published_at": "2023-11-05T08:00:00+00:00",
     "body": "Jane Doe, chief executive of Acme Semiconductors, announced she will step down in January.\n\n"
             "The board named John Roe as interim CEO."},
    {"title": "Local bakery wins award", "source": "Town Paper", "published_at": "2023-12-01T08:00:00+00:00",
     "body": "A family bakery won the regional bread award for its sourdough."},
]
QUERIES = [
    {"query": "Which company that reported record quarterly revenue of $4.2 billion also saw its chief executive step down?",
     "answer": "Acme Semiconductors", "question_type": "inference_query",
     "evidence_list": [{"title": CORPUS[0]["title"]}, {"title": CORPUS[1]["title"]}]},
    {"query": "What award did the family bakery win for its sourdough?", "answer": "regional bread award",
     "question_type": "inference_query", "evidence_list": [{"title": CORPUS[2]["title"]}]},
    {"query": "What did the mayor say about the stadium?", "answer": "Insufficient information.",
     "question_type": "null_query", "evidence_list": []},
]


def test_article_html():
    h = multihop.article_html(CORPUS[1]).decode()
    assert h.startswith("<h1>Acme CEO steps down</h1><p>Daily Tech, 2023-11-05</p>") and h.count("<p>") == 3


def test_multihop_end_to_end_on_a_tiny_corpus(ix):
    docs = multihop.ingest(ix.conn, ix.store, ix.index, ix.emb, CORPUS)
    assert len(docs) == 3
    rows = multihop.recall(ix.conn, ix.retriever, QUERIES, docs)
    assert len(rows) == 2 and all(r.needed and r.found == r.needed for r in rows)      # 无答案的不计召回
    s = multihop.recall_summary(rows)
    assert s["all"] == {"queries": 2, "evidence_recall": 1.0, "all_found": 1.0}


def test_answer_and_refusal_arithmetic():
    rows = [{"type": "null_query", "abstained": True, "correct": True}, {"type": "null_query", "abstained": False, "correct": False},
            {"type": "inference_query", "abstained": True, "correct": False},
            {"type": "comparison_query", "abstained": False, "correct": True},
            {"type": "temporal_query", "abstained": None, "correct": False}]
    s = multihop.answer_summary(rows)
    assert s["refusal_recall"] == 0.5 and s["refusal_precision"] == 0.5
    assert s["false_refusal"] == round(1 / 3, 4) and s["answer_accuracy"] == round(1 / 3, 4) and s["errors"] == 1


def test_vidore_picks_chart_and_table_questions_in_english():
    qs = [{"query_id": i, "language": lang, "content_type": ct, "query": f"q{i}", "answer": "a"}
          for i, (lang, ct) in enumerate([("english", ["Chart", "Text"]), ("english", ["Table", "Text"]),
                                          ("english", ["Chart", "Table"]), ("english", ["Text"]),
                                          ("french", ["Chart", "Text"])])]
    picked = vidore.pick(qs, 5)
    assert {(q["query_id"], q["kind"]) for q in picked} == {(0, "chart"), (2, "chart"), (1, "table")}
    rows = [{"kind": "chart", "verdict": "correct", "abstained": False, "cited_vision": True},
            {"kind": "chart", "verdict": "partial", "abstained": False, "cited_vision": False},
            {"kind": "table", "verdict": "incorrect", "abstained": True, "cited_vision": False}]
    s = vidore.summary(rows)
    assert s["chart"]["correct"] == 0.5 and s["chart"]["cited_vision"] == 0.5 and s["table"]["abstained"] == 1.0
    assert s["all"]["n"] == 3 and abs(s["all"]["correct"] - 1 / 3) < 1e-9
