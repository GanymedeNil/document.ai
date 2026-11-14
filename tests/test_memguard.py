"""本地模型服务的内存上限（modelserver/memguard.py、modelserver/watchdog.py）：默认 24 GB，不让模型服务把机器拖进
内存压力、被系统随机杀进程。不加载模型。

- 大小的写法；本进程的物理内存占用读得到；
- 加载前按缓存里的权重文件估算（按加载精度折算、加运行余量），本进程已占用加上估算超过预算就不加载；
- 看门狗：合计超过上限时从占用最大的进程开始终止，只终止到回到上限以内；接近上限只记警告；--dry-run 只记录；
  能找到命令行含 modelserver 的进程、跳过自己；
- 模型服务：内存不足时这一次请求返回 503，进程照常服务下一个请求；/health 报告内存占用。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from modelserver.memguard import (GiB, HEADROOM, MemoryBudgetExceeded, MemoryGuard, estimate_bytes, footprint,
                                  is_out_of_memory, parse_size)
from modelserver.watchdog import Proc, check, scan, victims


def test_sizes():
    assert parse_size("24G") == 24 * GiB and parse_size("24GB") == parse_size("24GiB") == 24 * GiB
    assert parse_size("512M") == 512 * 1024 ** 2 and parse_size("1.5G") == int(1.5 * GiB) and parse_size("100") == 100
    with pytest.raises(ValueError):
        parse_size("lots")


def test_own_footprint_is_readable():
    assert 10 * 1024 ** 2 < footprint(os.getpid()) < 64 * GiB
    assert footprint(2 ** 22 + 12345) == 0                         # 不存在的进程


def fake_model(tmp_path, monkeypatch, *, size: int, dtype: str = "bfloat16", rev: str = "a" * 40):
    snap = tmp_path / "hub" / "models--org--m" / "snapshots" / rev
    snap.mkdir(parents=True)
    with open(snap / "model.safetensors", "wb") as f:
        f.truncate(size)                                           # 稀疏文件，不真的占磁盘
    (snap / "config.json").write_text(json.dumps({"dtype": dtype}), encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    return rev


def test_estimate_from_cached_weights(tmp_path, monkeypatch):
    rev = fake_model(tmp_path, monkeypatch, size=2 * GiB)
    assert estimate_bytes("org/m", rev, "bfloat16") == int(2 * GiB * HEADROOM)
    assert estimate_bytes("org/m", rev, "float32") == int(4 * GiB * HEADROOM)     # bf16 存、fp32 加载：翻倍
    assert estimate_bytes("org/m", "b" * 40, "float32") is None                   # 这个提交没下载
    assert estimate_bytes("org/absent", None, "float32") is None


def test_guard_refuses_a_model_that_does_not_fit():
    used = footprint(os.getpid())
    g = MemoryGuard(used + 2 * GiB, "cpu")
    g.admit("small", 1 * GiB)
    g.admit("unknown", None)                                       # 估不出来时交给硬上限
    with pytest.raises(MemoryBudgetExceeded, match="超过内存预算"):
        g.admit("big", 3 * GiB)
    assert g.stats()["limit_bytes"] == used + 2 * GiB and g.relieve() is False


def test_watchdog_kills_the_largest_until_under_the_limit():
    procs = [Proc(1, "embed", 10 * GiB), Proc(2, "rerank", 9 * GiB), Proc(3, "small", 6 * GiB)]
    assert [p.pid for p in victims(procs, 24 * GiB)] == [1]                         # 25 → 15
    assert [p.pid for p in victims(procs, 5 * GiB)] == [1, 2, 3]
    assert victims(procs, 25 * GiB) == []
    killed, logs = [], []
    assert check(procs, 24 * GiB, kill=killed.append, log=logs.append) == [1] and killed == [1]
    assert "超过上限" in logs[0] and "终止 1" in logs[0]
    logs.clear()
    assert check(procs[1:], 16 * GiB, kill=killed.append, log=logs.append) == [] and "接近上限" in logs[0]
    logs.clear()
    assert check(procs, 24 * GiB, kill=None, log=logs.append) == [1] and killed == [1]   # 只记录
    logs.clear()
    assert check(procs[2:], 24 * GiB, kill=killed.append, log=logs.append) == [] and logs == []


def test_watchdog_finds_model_server_processes():
    child = subprocess.Popen([sys.executable, "-c", "import time, sys; sys.argv.append('modelserver-test'); time.sleep(30)",
                              "modelserver-test"])
    try:
        found = {p.pid: p for p in scan(["modelserver-test"])}
        assert child.pid in found and found[child.pid].bytes > 0 and os.getpid() not in found
    finally:
        child.kill()
        child.wait()


class Boom:
    dims = 3

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("MPS backend out of memory (MPS allocated: 23.9 GB, max allowed: 24.0 GB)")
        from modelserver.backends import Encoded
        return Encoded([[1.0, 0.0, 0.0] for _ in texts], 3)


def test_out_of_memory_is_a_503_not_a_crash():
    from modelserver.app import create_app

    assert is_out_of_memory(RuntimeError("CUDA out of memory")) and not is_out_of_memory(ValueError("x"))
    guard = MemoryGuard(24 * GiB, "cpu")
    c = TestClient(create_app(Boom(), None, embed_name="e", guard=guard))
    r = c.post("/v1/embeddings", json={"input": ["一"]})
    assert r.status_code == 503 and "内存不足" in r.json()["error"]["message"]
    assert c.post("/v1/embeddings", json={"input": ["一"]}).status_code == 200      # 进程照常服务
    assert c.get("/health").json()["memory"]["limit_bytes"] == 24 * GiB
