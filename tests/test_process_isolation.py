"""AC-3-2 进程隔离（开发文档 3.4、13.1）：后台任务进程停止时，查询照常可用，入库任务排队；恢复后续跑。

在进程层面验证：接口进程与后台任务进程各自作为子进程，按部署时的环境变量启动（tests/livestack.py），连真实的
PostgreSQL 与 OpenSearch，模型服务用 OpenAI 兼容的替身（tests/stubmodel.py，走与线上模型相同的客户端代码）。

1. 两个进程都在：上传两份制度，后台解析；审核通过其中一份，后台写入索引，提问能答出并引用它；
2. 停掉后台任务进程：再审核通过第二份、上传第三份——任务留在队列里（queued），没有人执行；
   这期间连续提问，成功率 100%（接口进程自己完成检索与生成，不依赖后台进程）；第三份未解析，审核通过被明确拒绝；
3. 重新启动后台任务进程：排队的任务全部完成、没有失败；第二份、第三份随后都能检索到并答出。
需要 EAER_TEST_OPENSEARCH_URL；未设置时跳过。
"""
from __future__ import annotations

import os
import uuid

import pytest

from eaer.ingestion.indexer import fresh_index
from eaer.retrieval.opensearch import connect
from livestack import Stack, approve, ask, open_tasks, upload, wait_until
from stubmodel import StubModelServer

OS_URL = os.environ.get("EAER_TEST_OPENSEARCH_URL")
pytestmark = pytest.mark.skipif(not OS_URL, reason="未设置 EAER_TEST_OPENSEARCH_URL")

TRAVEL = ("差旅报销管理办法", "第一条　为规范差旅报销，制定本办法。", "第二条　一线城市住宿上限为每晚650元。",
          "第三条　本办法自2026年1月1日起施行。")
BUYING = ("采购管理办法", "第一条　为规范采购行为，制定本办法。", "第二条　部门经理单笔采购审批上限为5万元。",
          "第三条　本办法自2026年1月1日起施行。")
MEETING = ("会议管理办法", "第一条　为规范会议组织，制定本办法。", "第二条　会议室预订须提前一个工作日提交申请。",
           "第三条　本办法自2026年1月1日起施行。")
Q_TRAVEL, Q_BUYING, Q_MEETING = "一线城市住宿上限是多少", "部门经理单笔采购审批上限是多少", "会议室预订须提前多久提交申请"


@pytest.fixture
def stack(db_url, tmp_path):
    alias = f"eaer-iso-{uuid.uuid4().hex[:8]}"
    with StubModelServer() as stub, fresh_index(connect(OS_URL), alias, build_id="b1", dims=stub.dims):
        s = Stack(tmp_path, db_url, OS_URL, alias, stub.env())
        try:
            yield s
        finally:
            s.close()


def unfinished(conn) -> list[dict]:
    return conn.execute("SELECT stage, status, count(*) AS n FROM ingest_job WHERE status <> 'done' "
                        "GROUP BY stage, status ORDER BY stage, status").fetchall()


def answer_cites(api: str, question: str, value: str) -> bool:
    r = ask(api, question)
    assert r.status_code == 200, r.text
    b = r.json()
    return b["status"] == "completed" and not b["abstained"] and value in b["answer"] and bool(b["claims"][0]["citations"])


def test_queries_survive_a_stopped_worker_and_jobs_resume(stack, conn):
    api = stack.start_api()
    stack.start_worker()

    # 1. 两个进程都在
    travel, buying = upload(api, *TRAVEL), upload(api, *BUYING)
    wait_until(lambda: {travel["document_id"], buying["document_id"]} <= set(open_tasks(api))
               and not unfinished(conn), what="两份文档解析完成", stack=stack, procs=["worker"])
    tasks = open_tasks(api)
    assert approve(api, tasks[travel["document_id"]]["task_id"])["ok"]
    wait_until(lambda: not unfinished(conn), what="第一份写入索引", stack=stack, procs=["worker"])
    assert answer_cites(api, Q_TRAVEL, "650元")

    # 2. 停掉后台任务进程：任务排队，提问照常
    stack.stop("worker")
    assert approve(api, tasks[buying["document_id"]]["task_id"])["ok"]       # 审核本身在接口进程内完成
    meeting = upload(api, *MEETING)
    queued = {(r["stage"], r["status"]): r["n"] for r in unfinished(conn)}
    assert queued == {("parse_text", "queued"): 1, ("publish_index", "queued"): 1, ("parse_deep", "queued"): 1}, queued
    refused = approve(api, open_tasks(api)[meeting["document_id"]]["task_id"])
    assert not refused["ok"] and "抽取尚未完成" in refused["error"]

    replies = [ask(api, q) for q in [Q_TRAVEL] * 10 + [Q_BUYING] * 5]
    assert [r.status_code for r in replies] == [200] * 15                  # 成功率 100%
    assert all(r.json()["status"] == "completed" for r in replies)
    assert all("650元" in r.json()["answer"] for r in replies[:10])
    assert all("5万元" not in r.json()["answer"] for r in replies[10:])     # 第二份还没进索引，查不到也不编造
    assert {(r["stage"], r["status"]): r["n"] for r in unfinished(conn)} == queued   # 没有人执行，也没有丢

    # 3. 重新启动：排队的任务全部完成
    stack.start_worker("worker-2")
    wait_until(lambda: not unfinished(conn) and meeting["document_id"] in open_tasks(api),
               what="排队的任务完成", stack=stack, procs=["worker-2"])
    assert approve(api, open_tasks(api)[meeting["document_id"]]["task_id"])["ok"]
    wait_until(lambda: not unfinished(conn), what="第三份写入索引", stack=stack, procs=["worker-2"])
    assert conn.execute("SELECT count(*) AS n FROM ingest_job WHERE status = 'failed'").fetchone()["n"] == 0
    assert answer_cites(api, Q_BUYING, "5万元")
    assert answer_cites(api, Q_MEETING, "一个工作日")
    stack.check_alive("api")
