"""本地模型服务的内存上限（modelserver/memguard.py、modelserver/watchdog.py）：默认 24 GB，不让模型服务把机器拖进
内存压力、被系统随机杀进程。不加载模型。

- 大小的写法；本进程的物理内存占用读得到；
- 加载前按缓存里的权重文件估算（按加载精度折算、加运行余量），本进程已占用加上估算超过预算就不加载；
- 看门狗：合计超过上限时从占用最大的进程开始终止，只终止到回到上限以内；接近上限只记警告；--dry-run 只记录；
  能找到命令行含 modelserver 的进程、跳过自己；读不到命令行的进程跳过，一轮检查出错时记下、下一轮照常查，不退出；
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


class FakeProc:
    def __init__(self, pid, cmd):
        self.pid, self._cmd = pid, cmd

    def cmdline(self):
        if isinstance(self._cmd, BaseException):
            raise self._cmd
        return self._cmd


def test_watchdog_skips_processes_it_cannot_read(monkeypatch):
    """读不到命令行的进程跳过，不让看门狗退出。2026-11-15：macOS 上 psutil 读取正在退出的进程抛 SystemError，
    看门狗因此退出，之后就没人兜底了。"""
    import psutil

    procs = [FakeProc(101, SystemError("<built-in function proc_cmdline> returned a result with an exception set")),
             FakeProc(102, psutil.NoSuchProcess(102)), FakeProc(103, psutil.AccessDenied(103)),
             FakeProc(104, ["python", "-m", "modelserver", "--port", "8100"]), FakeProc(105, ["bash"])]
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **k: iter(procs))
    monkeypatch.setattr("modelserver.watchdog.footprint", lambda pid: GiB)
    assert scan(["modelserver"], me=1) == [Proc(104, "python -m modelserver --port 8100", GiB)]


def test_watchdog_loop_survives_a_failed_round(monkeypatch, tmp_path):
    """一轮检查出了意外的错误：记一条、下一轮照常检查，而不是退出。"""
    import modelserver.watchdog as wd

    rounds = iter([RuntimeError("读取进程表失败"), [Proc(7, "modelserver", 30 * GiB)]])

    def fake_scan(patterns):
        r = next(rounds)
        if isinstance(r, Exception):
            raise r
        return r

    class Stop(Exception):
        pass

    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) == 2:
            raise Stop

    killed = []
    monkeypatch.setattr(wd, "scan", fake_scan)
    monkeypatch.setattr(wd, "terminate", killed.append)
    monkeypatch.setattr(wd.time, "sleep", fake_sleep)
    log = tmp_path / "wd.log"
    with pytest.raises(Stop):
        wd.main(["--limit", "24G", "--log", str(log)])
    text = log.read_text(encoding="utf-8")
    assert "这一轮检查失败" in text and "RuntimeError: 读取进程表失败" in text
    assert "超过上限" in text and killed == [7]                           # 第二轮照常检查、照常终止


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


def test_batches_are_split_when_memory_runs_out():
    """一批内存不足时对半拆开重试，结果顺序不变；拆到单条仍不够才把内存不足抛出去。"""
    from modelserver.backends import split_on_oom

    calls = []

    def run(items):
        calls.append(len(items))
        if len(items) > 2:
            raise RuntimeError("MPS backend out of memory")
        return [x * 10 for x in items]
    assert split_on_oom(run, [1, 2, 3, 4, 5, 6, 7, 8], "cpu") == [10, 20, 30, 40, 50, 60, 70, 80]
    assert calls[0] == 8 and max(calls[1:]) <= 4

    def always(items):
        raise RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        split_on_oom(always, [1, 2], "cpu")

    def other(items):
        raise ValueError("bad input")
    with pytest.raises(ValueError):                                                  # 不是内存不足：原样抛出，不重试
        split_on_oom(other, [1, 2, 3], "cpu")
