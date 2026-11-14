"""模型服务的内存看门狗（进程外）：几个模型服务加起来不超过给定的内存上限（默认 24 GB）。

    python -m modelserver.watchdog --limit 24G [--match modelserver] [--interval 2] [--log 文件] [--dry-run]

每隔 interval 秒，把命令行里含 --match 任一片段的进程（默认 “modelserver”，看门狗自己除外）的物理内存占用加起来：
- macOS：phys_footprint（与活动监视器一致，含 MPS 显存）；
- Linux：有 nvidia-smi 时取各进程占用的显存，否则取 RSS。
总量超过上限的 90% 记一条警告；超过上限时，按占用从大到小终止进程（先 SIGTERM，等 5 秒还在就 SIGKILL），
直到剩下的回到上限以内——宁可停掉一个模型服务，也不让系统在内存压力下随机杀进程。--dry-run 只记录不终止。
进程内的预算与硬上限见 memguard.py；看门狗兜底的是“几个进程各自没超、加起来超了”的情况。
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

from modelserver.memguard import fmt, footprint, parse_size

WARN_AT = 0.9


@dataclass(frozen=True)
class Proc:
    pid: int
    name: str
    bytes: int


def gpu_memory_by_pid() -> dict[int, int]:
    """Linux 上 nvidia-smi 报告的各进程显存；没有 nvidia-smi 时为空。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    usage: dict[int, int] = {}
    for line in out.splitlines():
        pid, _, mib = line.partition(",")
        if pid.strip().isdigit() and mib.strip().isdigit():
            usage[int(pid)] = usage.get(int(pid), 0) + int(mib) * 1024 * 1024
    return usage


def scan(patterns: list[str], me: int | None = None) -> list[Proc]:
    import psutil

    me = os.getpid() if me is None else me
    gpu = gpu_memory_by_pid() if sys.platform.startswith("linux") else {}
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(p.info.get("cmdline") or [])
        if p.info["pid"] == me or "modelserver.watchdog" in cmd or not any(s in cmd for s in patterns):
            continue
        used = gpu.get(p.info["pid"]) if gpu else footprint(p.info["pid"])
        out.append(Proc(p.info["pid"], cmd[:120], used or 0))
    return out


def victims(procs: list[Proc], limit: int) -> list[Proc]:
    """要终止的进程：从占用最大的开始，直到剩下的总和不超过上限。"""
    total, out = sum(p.bytes for p in procs), []
    for p in sorted(procs, key=lambda p: -p.bytes):
        if total <= limit:
            break
        out.append(p)
        total -= p.bytes
    return out


def terminate(pid: int, grace: float = 5.0) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def check(procs: list[Proc], limit: int, *, kill: Callable[[int], None] | None, log: Callable[[str], None]) -> list[int]:
    """检查一次；返回终止了的进程。"""
    total = sum(p.bytes for p in procs)
    detail = "，".join(f"{p.pid} {fmt(p.bytes)}" for p in sorted(procs, key=lambda p: -p.bytes))
    if total > limit:
        out = []
        for v in victims(procs, limit):
            log(f"超过上限：合计 {fmt(total)} > {fmt(limit)}（{detail}）；终止 {v.pid}（{fmt(v.bytes)}）：{v.name}")
            if kill is not None:
                kill(v.pid)
            out.append(v.pid)
        return out
    if total > WARN_AT * limit:
        log(f"接近上限：合计 {fmt(total)}，上限 {fmt(limit)}（{detail}）")
    return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="modelserver.watchdog", description="模型服务的内存看门狗")
    p.add_argument("--limit", default="24G")
    p.add_argument("--match", action="append", help="命令行包含这个片段的进程计入（可重复；默认 modelserver）")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--log", help="另把记录追加到这个文件")
    p.add_argument("--dry-run", action="store_true", help="只记录，不终止进程")
    p.add_argument("--once", action="store_true", help="只检查一次（打印当前占用）")
    args = p.parse_args(argv)
    limit, patterns = parse_size(args.limit), args.match or ["modelserver"]

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, file=sys.stderr, flush=True)
        if args.log:
            with open(args.log, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    log(f"看门狗启动：上限 {fmt(limit)}，计入命令行含 {patterns} 的进程，每 {args.interval:g} 秒检查"
        + ("（只记录）" if args.dry_run else ""))
    while True:
        procs = scan(patterns)
        if args.once:
            log(f"当前合计 {fmt(sum(x.bytes for x in procs))}：" + ("；".join(f"{x.pid} {fmt(x.bytes)} {x.name}" for x in procs)
                                                              or "没有匹配的进程"))
            return 0
        check(procs, limit, kill=None if args.dry_run else terminate, log=log)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
