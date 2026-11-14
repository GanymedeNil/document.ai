"""冒烟测试集（eaer/smoke.py）本身的验收：对真实的接口进程与后台任务进程（tests/livestack.py，模型服务用替身）
跑一遍，九步全部通过。部署演练（AC-13-1、13-4）与集群扩展（AC-13-2）用的就是它。需要 EAER_TEST_OPENSEARCH_URL。"""
from __future__ import annotations

import os
import uuid

import pytest

from eaer.ingestion.indexer import fresh_index
from eaer.retrieval.opensearch import connect
from eaer.smoke import Api, diff, run, seed, snapshot
from livestack import Stack
from stubmodel import StubModelServer

OS_URL = os.environ.get("EAER_TEST_OPENSEARCH_URL")
pytestmark = pytest.mark.skipif(not OS_URL, reason="未设置 EAER_TEST_OPENSEARCH_URL")


def test_smoke_set_passes_on_a_live_stack(db_url, tmp_path):
    alias = f"eaer-smoke-{uuid.uuid4().hex[:8]}"
    with StubModelServer() as stub, fresh_index(connect(OS_URL), alias, build_id="b1", dims=stub.dims):
        stack = Stack(tmp_path, db_url, OS_URL, alias, stub.env())
        try:
            api = stack.start_api()
            stack.start_worker()
            steps = run(Api(api), editor="tok-ed", reviewer="tok-rv", user="tok-staff", timeout=60)
            assert len(steps) == 9 and all(s["ok"] for s in steps), steps
            # 备份恢复与索引重建演练用的回溯快照（AC-13-3、13-4）
            seed(Api(api), editor="tok-ed", reviewer="tok-rv", timeout=60)
            import time
            deadline = time.monotonic() + 60
            while True:
                snap = snapshot(Api(api), user="tok-staff")
                if [r["versions"] for r in snap] == [[2], [1], [2], []] or time.monotonic() > deadline:
                    break
                time.sleep(1)
            assert [r["versions"] for r in snap] == [[2], [1], [2], []], snap   # 当前、3 月 1 日、8 月 1 日、1 月 15 日系统所知
            assert snap[3]["abstained"] and not snap[0]["abstained"]
            assert diff(snap, snapshot(Api(api), user="tok-staff")) == []
            changed = [dict(r) for r in snap]
            changed[1]["versions"] = [2]
            assert len(diff(snap, changed)) == 1
        finally:
            stack.close()
