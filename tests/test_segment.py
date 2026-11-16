"""应用侧中文分词（jieba 候选，开发文档 3.3“中文分词”）与 OpenSearch 预切词写入、查询。"""
from __future__ import annotations

import random
from dataclasses import replace
from datetime import date

from eaer.retrieval.index import Filters
from eaer.retrieval.opensearch import PRE_SEGMENTED, OpenSearchIndex
from eaer.retrieval.segment import JiebaSegmenter
from test_retrieval import FakeClient, random_doc

SEG = JiebaSegmenter()


def test_domain_words_stay_whole_and_search_mode_adds_parts():
    words = SEG.segment("关于差旅住宿标准的补充通知")
    assert {"差旅", "住宿", "标准", "补充", "通知"} <= set(words)
    assert {"部门", "经理", "部门经理"} <= set(SEG.segment("部门经理单笔采购审批上限"))


def test_punctuation_dropped_and_lowercased():
    words = SEG.segment("一线城市住宿上限为 650 元/晚，Hotel")
    assert "/" not in words and "，" not in words and " " not in words
    assert "650" in words and "hotel" in words


def test_user_dictionary(tmp_path):
    d = tmp_path / "terms.txt"
    d.write_text("差旅费用分析 10 n\n", encoding="utf-8")
    seg = JiebaSegmenter(d)
    assert "差旅费用分析" in seg.segment("2026 年第二季度差旅费用分析报告")
    assert seg.name.endswith("+terms.txt")


def test_opensearch_index_writes_and_queries_pre_segmented_text():
    client = FakeClient()
    ix = OpenSearchIndex(client, segmenter=SEG)
    ix.create("b1", dims=4, analyzer="smartcn")          # 应用侧切词时忽略引擎分词器
    body = client.indices.calls[0][1]["body"]["mappings"]
    assert body["properties"]["text"]["analyzer"] == PRE_SEGMENTED
    assert body["_meta"]["text_analysis"] == SEG.name

    doc = replace(random_doc(random.Random(1)), text="关于差旅住宿标准的补充通知")
    ix.upsert([doc])
    assert client.bulks[0][1]["text"] == " ".join(SEG.segment(doc.text))

    f = Filters("t1", frozenset({"A"}), date(2026, 1, 1))
    ix.search_text("差旅住宿标准", f, 10)
    assert client.searches[0]["query"]["bool"]["must"] == [{"match": {"text": "差旅 住宿 标准"}}]
    assert ix.search_text("，。", f, 10) == [] and len(client.searches) == 1   # 切完为空不发请求
