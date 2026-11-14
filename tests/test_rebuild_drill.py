"""AC-13-3 索引重建（开发文档 13.4、5.3）：删除索引后从数据库与原件重建，重建后评测结果与重建前一致。

开发文档把它列为人工演练（部署说明第 6 节）；这里把演练的判定做成用例，真实 OpenSearch，未设置
EAER_TEST_OPENSEARCH_URL 时跳过：
- 数据覆盖各种状态：多份制度、同一制度的两个版本（新版取代旧版，有效期衔接）、记录类文档、归档的文档、
  只给 R_FIN 的文档、权限变更过的文档；
- 一组检索（不同权限、当前、某天实际生效、截至某天系统所知）记下结果（证据 id 与顺序），再加上对账；
- 整个删除索引（别名与具体索引都没了，模拟数据卷损坏），用部署时的 ensure-index 重建；
- 重建后同一组检索结果完全相同、对账差异为 0；重建不调用 Embedding 服务（向量取自数据库缓存）。
"""
from __future__ import annotations

import os
import uuid
from datetime import date

import psycopg
import pytest
from psycopg.rows import dict_row
from stubmodel import StubModelServer

from eaer.adapters.online import OnlineModelClient
from eaer.ingestion.__main__ import main as ingestion_main
from eaer.ingestion.indexer import Indexer, fresh_index
from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.retrieval.opensearch import OpenSearchIndex, connect
from eaer.retrieval.reconcile import reconcile
from eaer.retrieval.service import Retriever, Scope
from eaer.review.service import Review
from eaer.storage import LocalFileStore

URL = os.environ.get("EAER_TEST_OPENSEARCH_URL")
pytestmark = pytest.mark.skipif(not URL, reason="未设置 EAER_TEST_OPENSEARCH_URL")
TODAY = date(2026, 11, 14)


def doc(title, *paras):
    return (f"<h1>{title}</h1>" + "".join(f"<p>{p}</p>" for p in paras)).encode()


CORPUS = [
    ("travel", "R_STAFF", "normative", doc("差旅报销管理办法", "第一条　一线城市住宿上限为每晚600元。", "第二条　本办法自2026年1月1日起施行。")),
    ("travel", "R_STAFF", "normative", doc("差旅报销管理办法", "第一条　一线城市住宿上限为每晚680元。",
                                            "第二条　本办法自2026年7月1日起施行，原办法同时废止。")),
    ("meeting", "R_STAFF", "normative", doc("会议管理办法", "第一条　会议室预订须提前一个工作日提交申请。", "第二条　本办法自2026年1月1日起施行。")),
    ("report", "R_STAFF", "record", doc("2026年第二季度差旅费用分析报告", "报告日期：2026年7月10日", "第二季度差旅总费用为128万元。")),
    ("salary", "R_FIN", "normative", doc("高管薪酬管理办法", "第一条　高管年薪上限为300万元。", "第二条　本办法自2026年1月1日起施行。")),
    ("vehicle", "R_STAFF", "normative", doc("车辆使用管理办法", "第一条　公务用车须提前半天登记。", "第二条　本办法自2026年1月1日起施行。")),
    ("seal", "R_STAFF", "normative", doc("印章管理办法", "第一条　用印须经部门负责人审批。", "第二条　本办法自2026年1月1日起施行。")),
]
QUERIES = [
    (frozenset({"R_STAFF"}), "一线城市住宿上限是多少", {}),
    (frozenset({"R_STAFF"}), "一线城市住宿上限是多少", {"as_of_valid": date(2026, 3, 1)}),
    (frozenset({"R_STAFF"}), "一线城市住宿上限是多少", {"as_of_system": date(2026, 11, 13)}),
    (frozenset({"R_STAFF"}), "第二季度差旅总费用", {}),
    (frozenset({"R_STAFF"}), "会议室预订提前多久", {}),
    (frozenset({"R_STAFF"}), "高管年薪上限", {}),
    (frozenset({"R_FIN"}), "高管年薪上限", {}),
    (frozenset({"R_STAFF", "R_FIN"}), "公务用车登记", {}),
    (frozenset({"R_STAFF"}), "用印审批", {}),
]


def test_index_rebuilt_from_the_database_gives_the_same_results(db_url, tmp_path, monkeypatch):
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    ing = Ingestion(conn, LocalFileStore(tmp_path / "files"))
    ing.lc.ensure_tenant("t1")
    review = Review(conn, ing.lc, ing.store)
    alias = f"eaer-drill-{uuid.uuid4().hex[:8]}"
    client = connect(URL)
    with StubModelServer() as stub:
        embedder = OnlineModelClient.from_env(stub.env()).embedder()   # 与部署相同：经 Embedding 服务（替身）
        with fresh_index(client, alias, build_id="b1", dims=stub.dims):
            worker = Worker(ing, Indexer(conn, OpenSearchIndex(client, alias=alias), embedder, "b1"))
            docs = {}
            for ref, role, kind, body in CORPUS:
                r = ing.receive("t1", body, f"{ref}.html", acl_policy_id=ing.lc.create_acl_policy("t1", [role]),
                                source_ref=ref)
                worker.run_until_idle()
                task = conn.execute("SELECT id FROM review_task WHERE version_id = %s", (r.version_id,)).fetchone()["id"]
                assert review.decide("t1", [{"task_id": str(task), "decision": "approve",
                                             "corrections": {"doc_type": kind}}], "rv")[0]["ok"]
                worker.run_until_idle()
                docs[ref] = r.document_id
            ing.lc.archive_document(docs["seal"], "admin")
            ing.lc.change_acl(docs["vehicle"], ["R_FIN"], "admin")
            worker.run_until_idle()

            def results():
                ix = OpenSearchIndex(client, alias=alias)
                r = Retriever(conn, ix, embedder)
                out = [[h.evidence_id for h in r.search(Scope("t1", roles), q, today=TODAY, **kw)]
                       for roles, q, kw in QUERIES]
                return out, reconcile(conn, ix, "t1")

            before, diff = results()
            assert not any(diff.values())
            assert before[0] and before[1] and before[1] != before[0]          # 当前与 3 月 1 日取到不同版本
            def evidence_of(ref):
                return {r["id"] for r in conn.execute(
                    "SELECT e.id FROM evidence e JOIN document_version v ON v.id = e.version_id "
                    "WHERE v.document_id = %s", (docs[ref],))}
            salary, seal = evidence_of("salary"), evidence_of("seal")
            assert not salary & set(before[5]) and salary & set(before[6])      # 无权的查不到，有权的查得到
            assert not seal & {i for r in before for i in r}                   # 已归档的谁都查不到

            for name in OpenSearchIndex(client, alias=alias).current():        # 模拟数据卷损坏：索引整个没了
                client.indices.delete(index=name)
            assert not OpenSearchIndex(client, alias=alias).current()
            calls = dict(stub.calls)
            for k, v in {"EAER_OPENSEARCH_URL": URL, "EAER_INDEX_ALIAS": alias, "EAER_DATABASE_URL": db_url,
                         **stub.env()}.items():
                monkeypatch.setenv(k, v)
            assert ingestion_main(["ensure-index", "--build-id", "b2"]) == 0    # 部署时的命令
            assert stub.calls == calls                                         # 向量全部取自数据库缓存
            assert ingestion_main(["ensure-index"]) == 0                       # 已存在：不动
            assert sorted(OpenSearchIndex(client, alias=alias).current()) == [f"{alias}-b2"]

            after, diff = results()
            assert not any(diff.values())
            assert after == before
            OpenSearchIndex(client, alias=alias).drop_all()
    conn.close()
