"""显存（统一内存）上限：本地模型服务不超过给定的内存预算（默认 24 GB）。

Mac 的 GPU 与 CPU 共用内存，模型服务占多了，系统会在内存压力下直接杀进程（连同别的程序）。这里分两道：

1. 进程内（MemoryGuard，模型服务启动时装上，MODELSERVER_MEMORY_LIMIT，默认 24G）：
   - 加载前估算：模型权重的大小（Hugging Face 缓存里 safetensors 文件的总和，按加载精度折算）乘以运行余量，
     加上本进程已占用的，超过预算就不加载、启动报错——不在加载途中被系统杀掉；
   - 给 PyTorch 设硬上限：MPS 用 torch.mps.set_per_process_memory_fraction（相对 Metal 建议的最大工作集），
     CUDA 用 torch.cuda.set_per_process_memory_fraction（相对显卡总显存）；超过时 PyTorch 抛内存不足，
     这一次请求返回 503，进程不死；
   - 每次请求后，GPU 上已分配的超过预算的 80% 时释放缓存（empty_cache）；/health 报告占用。
2. 进程外（watchdog.py，python -m modelserver.watchdog --limit 24G）：多个模型服务一起跑时看总量，见该文件。

物理内存占用：macOS 取 phys_footprint（与活动监视器的“内存”一列一致，含 MPS 分配的显存）；
其他系统取 RSS，CUDA 的显存另由 nvidia-smi 给出（watchdog.py）。
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

GiB = 1024 ** 3
HEADROOM = 1.3            # 权重之外的运行占用（激活、KV、分词器、框架本身）按权重的 30% 估
RELIEVE_AT = 0.8          # GPU 已分配超过预算的这个比例时释放缓存
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


class MemoryBudgetExceeded(RuntimeError):
    pass


def parse_size(text: str) -> int:
    """“24G”“512M”“1.5G”“25769803776” → 字节数。"""
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)i?B?\s*", str(text), re.I)
    if not m:
        raise ValueError(f"无法识别的大小：{text}（例如 24G、512M）")
    return int(float(m.group(1)) * 1024 ** "_KMGT".index(m.group(2).upper() or "_"))


def fmt(n: int | float) -> str:
    return f"{n / GiB:.2f} GB"


# ---------- 物理内存占用 ----------

class _RusageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(f, ctypes.c_uint64) for f in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime",
        "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
        "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten")]


_libproc = None


def footprint(pid: int) -> int:
    """进程的物理内存占用（字节）；进程不存在或无权读取时为 0。"""
    global _libproc
    if sys.platform == "darwin":
        if _libproc is None:
            _libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib", use_errno=True)
        info = _RusageInfoV2()
        if _libproc.proc_pid_rusage(int(pid), 2, ctypes.byref(info)) != 0:      # RUSAGE_INFO_V2
            return 0
        return int(info.ri_phys_footprint)
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


# ---------- 加载前估算 ----------

def snapshot_dir(repo: str, revision: str | None) -> Path | None:
    """Hugging Face 缓存中这个模型这次提交的目录；没下载时为 None。"""
    root = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
                / "hub")
    snaps = root / f"models--{repo.replace('/', '--')}" / "snapshots"
    if revision and (snaps / revision).is_dir():
        return snaps / revision
    found = sorted(snaps.iterdir()) if snaps.is_dir() else []
    return found[-1] if found and not revision else None


def estimate_bytes(repo: str, revision: str | None, load_dtype: str) -> int | None:
    """按缓存里的权重文件估算加载后的占用；没下载（要先下载）时为 None，交给加载后的实测与硬上限。"""
    d = snapshot_dir(repo, revision)
    if d is None:
        return None
    weights = sum(p.stat().st_size for p in d.glob("*.safetensors"))
    try:
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        stored = cfg.get("dtype") or cfg.get("torch_dtype") or (cfg.get("text_config") or {}).get("dtype") or "bfloat16"
    except (OSError, ValueError):
        stored = "bfloat16"
    ratio = DTYPE_BYTES.get(load_dtype, 4) / DTYPE_BYTES.get(str(stored), 2)
    return int(weights * ratio * HEADROOM)


# ---------- 进程内的预算 ----------

@dataclass
class MemoryGuard:
    limit: int
    device: str
    pid: int = field(default_factory=os.getpid)

    def admit(self, name: str, estimate: int | None) -> None:
        """加载一个模型之前调用：本进程已占用加上估算超过预算就报错，不加载。"""
        used = footprint(self.pid)
        if estimate is not None and used + estimate > self.limit:
            raise MemoryBudgetExceeded(
                f"加载 {name} 预计需要 {fmt(estimate)}，本进程已占用 {fmt(used)}，超过内存预算 {fmt(self.limit)}"
                "（MODELSERVER_MEMORY_LIMIT）。可以只加载其中一个模型（--no-embedding / --no-rerank），"
                "或换小一档的模型")

    def apply(self) -> None:
        """给 PyTorch 设硬上限：之后的 GPU 分配超过预算时抛内存不足，而不是让系统杀进程。"""
        import torch
        if self.device == "mps":
            rec = torch.mps.recommended_max_memory()
            torch.mps.set_per_process_memory_fraction(min(2.0, max(0.01, self.limit / rec)))
        elif self.device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, self.limit / total))

    def gpu_allocated(self) -> int:
        import torch
        if self.device == "mps":
            return int(torch.mps.driver_allocated_memory())
        if self.device == "cuda":
            return int(torch.cuda.memory_reserved())
        return 0

    def relieve(self) -> bool:
        """请求结束后调用：GPU 已分配超过预算的 80% 时释放缓存；返回是否释放了。"""
        if self.gpu_allocated() < RELIEVE_AT * self.limit:
            return False
        import torch
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        return True

    def stats(self) -> dict:
        return {"limit_bytes": self.limit, "process_bytes": footprint(self.pid),
                "gpu_allocated_bytes": self.gpu_allocated(), "device": self.device}


def is_out_of_memory(e: BaseException) -> bool:
    text = str(e).lower()
    return isinstance(e, MemoryError) or "out of memory" in text or "outofmemory" in type(e).__name__.lower()
