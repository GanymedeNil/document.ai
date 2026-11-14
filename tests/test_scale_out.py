"""AC-13-2 横向扩展（开发文档 13.1、13.5）：接口进程扩到 2 个实例，后台进程扩到 3 个——冒烟通过；会话与异步任务不丢失。

真实进程（tests/livestack.py）：2 个接口进程、3 个后台任务进程，共用 PostgreSQL（任务队列、会话检查点）、
OpenSearch 与文件存储，模型服务用替身（生成调用可以调慢）。
- 冒烟测试集分别经两个接口实例跑通（上传、解析、审核、写入索引由哪个后台进程做都可以）；
- 会话：在实例 A 提问，实例 B 能看到这个会话并接着问，A 再看时两轮都在；
- 异步任务：同步时限调到很短、生成调慢，同时经两个实例发起多个提问——都先返回 running，由后台进程接着执行；
  全部完成，每个恰好完成一次（query_log 一问一条、状态 completed），两个实例看到的会话一致；
  执行中杀掉一个后台进程，任务也不丢（租约到期后由其他后台进程接手）。
集群版的 Temporal 任务编排与 S3 兼容文件存储尚未实现（见 CLAUDE.md），这里的扩展用的是精简版的组件。
需要 EAER_TEST_OPENSEARCH_URL。
"""
from __future__ import annotations

import os
import threading
import time
import uuid

import httpx
import pytest

from eaer.ingestion.indexer import fresh_index
from eaer.retrieval.opensearch import connect
from eaer.smoke import Api, run
from livestack import H, Stack, approve, ask, open_tasks, upload, wait_until
from stubmodel import StubModelServer

OS_URL = os.environ.get("EAER_TEST_OPENSEARCH_URL")
pytestmark = pytest.mark.skipif(not OS_URL, reason="未设置 EAER_TEST_OPENSEARCH_URL")
Q = "一线城市住宿上限是多少"


def session(api, sid):
    r = httpx.get(f"{api}/v1/sessions/{sid}", headers=H("tok-staff"), timeout=30)
    assert r.status_code == 200, r.text
    return r.json()["items"]


def test_two_api_instances_and_three_workers(db_url, conn, tmp_path):
    alias = f"eaer-scale-{uuid.uuid4().hex[:8]}"
    with StubModelServer() as stub, fresh_index(connect(OS_URL), alias, build_id="b1", dims=stub.dims):
        stack = Stack(tmp_path, db_url, OS_URL, alias, stub.env())
        try:
            a, b = stack.start_api("api-a"), stack.start_api("api-b")
            for k in (1, 2, 3):
                stack.start_worker(f"worker-{k}")

            # 冒烟：两个实例各跑一遍
            for api in (a, b):
                steps = run(Api(api), editor="tok-ed", reviewer="tok-rv", user="tok-staff", timeout=60)
                assert len(steps) == 9 and all(s["ok"] for s in steps), (api, steps)

            # 会话跨实例
            doc = upload(a, "差旅报销管理办法", "第一条　为规范差旅报销，制定本办法。", "第二条　一线城市住宿上限为每晚650元。",
                         "第三条　本办法自2026年1月1日起施行。")
            wait_until(lambda: doc["document_id"] in open_tasks(b), what="解析完成", stack=stack, procs=["worker-1"])
            wait_until(lambda: approve(b, open_tasks(b)[doc["document_id"]]["task_id"])["ok"], what="审核通过")
            wait_until(lambda: "650元" in (ask(a, Q).json().get("answer") or ""), what="写入索引")
            first = ask(a, Q).json()
            follow = ask(b, "二线城市呢", session_id=first["session_id"])
            assert follow.status_code == 200, follow.text
            assert len(session(b, first["session_id"])) == 2 and len(session(a, first["session_id"])) == 2

            # 异步任务：同步时限很短、生成调慢，同时经两个实例提问
            conn.execute("INSERT INTO tenant_config (tenant_id, budgets, updated_at) VALUES ('t1', %s, now()) "
                         "ON CONFLICT (tenant_id) DO UPDATE SET budgets = EXCLUDED.budgets",
                         ('{"sync_seconds": 0.5, "cache_ttl_seconds": 0}',))
            stub.generate_delay = 1.5
            replies, lock = [], threading.Lock()

            def fire(api, k):
                r = ask(api, f"{Q}（第{k}问）")
                with lock:
                    replies.append((api, r))

            threads = [threading.Thread(target=fire, args=((a, b)[k % 2], k)) for k in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(60)
            assert len(replies) == 6 and all(r.status_code == 200 for _, r in replies)
            assert all(r.json()["status"] == "running" for _, r in replies)     # 都转入了后台
            time.sleep(1.0)
            stack.stop("worker-1", kill=True)                                   # 执行中杀掉一个后台进程
            sids = [r.json()["session_id"] for _, r in replies]

            def all_done():
                rows = conn.execute("SELECT session_id, status FROM query_log WHERE session_id = ANY(%s::uuid[])",
                                    (sids,)).fetchall()
                return len(rows) == 6 and all(r["status"] == "completed" for r in rows)

            wait_until(all_done, timeout=150, what="异步任务全部完成（含租约到期后接手）", stack=stack,
                       procs=["worker-2", "worker-3"])
            for sid in sids:                                                    # 每问恰好一条，两个实例看到的一致
                items_a, items_b = session(a, sid), session(b, sid)
                assert len(items_a) == 1 and items_a == items_b
                assert items_a[0]["status"] == "completed" and "650元" in (items_a[0]["answer"] or "")
            tasks = conn.execute("SELECT status, count(*) AS n FROM query_task WHERE session_id = ANY(%s::uuid[]) "
                                 "GROUP BY status", (sids,)).fetchall()
            assert {r["status"]: r["n"] for r in tasks} == {"completed": 6}
            for name in ("api-a", "api-b", "worker-2", "worker-3"):
                stack.check_alive(name)
        finally:
            stack.close()
