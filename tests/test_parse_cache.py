"""解析结果的暂存（迁移 0011，6.2 第 2、6 阶段）：文本解析的结果留给深度解析复用，扫描件与图片不再做两遍 OCR。

- 一份带表格与图表的文档从接收到深度解析完成只解析一次；深度解析得到的证据与重新解析时完全相同；
- 深度解析成功后删除暂存；有图片读取失败时保留，重试照样不再解析；
- 解析器版本不同（升级后）时重新解析；
- 删除待审核版本时暂存随版本删除，不留内容（4.4、6.5）。
"""
from __future__ import annotations

import base64

from eaer.ingestion.figures import FigureReader
from eaer.ingestion.parsers import HtmlParser
from test_figures import CHART, Vision, png
from test_ingestion import env  # noqa: F401
from test_retrieval import ix  # noqa: F401


class Counting(HtmlParser):
    def __init__(self, name: str = HtmlParser.name):
        self.name, self.calls = name, 0

    def parse(self, data, filename):
        self.calls += 1
        return super().parse(data, filename)


BODY = ("<h1>差旅费用分析报告</h1><p>报告日期：2026年7月10日</p>"
        "<table><tr><th>项目</th><th>金额</th></tr><tr><td>住宿</td><td>58万元</td></tr></table>"
        f'<figure><img src="data:image/png;base64,{base64.b64encode(png()).decode()}" alt="图1"></figure>').encode()


def publish(ix, data=BODY, name="r.html"):
    """接收、解析、审核通过；不执行发布后的任务（写入索引、深度解析），留给用例自己调用。"""
    r = ix.ing.receive("t1", data, name, acl_policy_id=ix.acl)
    ix.worker.run_until_idle()
    task = str(ix.conn.execute("SELECT id FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()["id"])
    assert ix.review.decide("t1", [{"task_id": task, "decision": "approve", "corrections": {"doc_type": "record"}}],
                            "rv")[0]["ok"]
    return r


def deep_evidence(ix, vid):
    return sorted((r["modality"], r["content"]) for r in ix.conn.execute(
        "SELECT modality, content FROM evidence WHERE version_id = %s AND modality <> 'text'", (vid,)))


def cache_rows(ix, vid):
    return (ix.conn.execute("SELECT count(*) AS n FROM parse_cache WHERE version_id = %s", (vid,)).fetchone()["n"],
            ix.conn.execute("SELECT count(*) AS n FROM parse_cache_image WHERE version_id = %s", (vid,)).fetchone()["n"])


def test_one_parse_from_receipt_to_deep_parse(ix):
    p = Counting()
    ix.ing.parsers, ix.ing.figure_reader = [p], FigureReader(Vision(CHART))
    r = ix.ing.receive("t1", BODY, "r.html", acl_policy_id=ix.acl)
    ix.worker.run_until_idle()
    assert p.calls == 1 and cache_rows(ix, r.version_id) == (1, 1)               # 块与图片都暂存了
    task = str(ix.conn.execute("SELECT id FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()["id"])
    ix.review.decide("t1", [{"task_id": task, "decision": "approve", "corrections": {"doc_type": "record"}}], "rv")
    ix.worker.run_until_idle()
    assert p.calls == 1                                                           # 深度解析没有再解析
    cached = deep_evidence(ix, r.version_id)
    assert ("table", "住宿｜金额：58万元") in cached and any(m == "figure" for m, _ in cached)
    assert cache_rows(ix, r.version_id) == (0, 0)                                 # 完成后删除
    ix.ing.figure_reader = FigureReader(Vision(CHART))
    ix.ing.parse_deep(r.version_id)                                               # 没有暂存：重新解析，结果相同
    assert p.calls == 2 and deep_evidence(ix, r.version_id) == cached


def test_cache_survives_a_failed_figure_and_a_new_parser_reparses(ix):
    p = Counting()
    ix.ing.parsers, ix.ing.figure_reader = [p], FigureReader(Vision(TimeoutError("看图超时"), CHART))
    r = publish(ix)
    try:                                                                          # 第一次：图片读取失败，暂存保留
        ix.ing.parse_deep(r.version_id)
    except Exception as e:  # noqa: BLE001
        assert type(e).__name__ == "FigureReadError"
    assert p.calls == 1 and cache_rows(ix, r.version_id) == (1, 1)
    ix.ing.parse_deep(r.version_id)                                               # 重试：用暂存，读到图，删除暂存
    assert p.calls == 1 and cache_rows(ix, r.version_id) == (0, 0)
    assert any(m == "figure" for m, _ in deep_evidence(ix, r.version_id))
    q = Counting("native-html-2")                                                 # 解析器升级：暂存作废
    r2 = publish(ix, BODY.replace(b"58", b"61"), "r2.html")
    ix.ing.parsers, ix.ing.figure_reader = [q], None
    ix.ing.parse_deep(r2.version_id)
    assert q.calls == 1


def test_deleting_a_pending_version_removes_its_cache(ix):
    r = ix.ing.receive("t1", BODY, "p.html", acl_policy_id=ix.acl)
    ix.worker.run_until_idle()
    assert cache_rows(ix, r.version_id) == (1, 1)
    ix.lc.delete_pending(r.version_id, "rv")
    assert cache_rows(ix, r.version_id) == (0, 0)
