"""真实进程的测试环境：接口进程（uvicorn eaer.api.main:app）与后台任务进程（python -m eaer.ingestion）各自作为
子进程启动，按部署时的环境变量配置，连真实的 PostgreSQL、OpenSearch，模型服务用 tests/stubmodel.py 的替身。

用于进程隔离（AC-3-2）与横向扩展（AC-13-2）这类必须在进程层面验证的验收项。
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
TOKENS = {"tok-ed": {"user": "ed", "tenant": "t1", "roles": ["editor"]},
          "tok-rv": {"user": "rv", "tenant": "t1", "roles": ["reviewer"]},
          "tok-staff": {"user": "staff", "tenant": "t1", "roles": [], "principals": ["R_STAFF"]}}


def H(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def page(title: str, *paras: str) -> bytes:
    return (f"<h1>{title}</h1>" + "".join(f"<p>{p}</p>" for p in paras)).encode("utf-8")


class Stack:
    def __init__(self, tmp: Path, db_url: str, opensearch_url: str, alias: str, model_env: dict[str, str],
                 extra: dict[str, str] | None = None):
        tmp.mkdir(parents=True, exist_ok=True)
        tokens = tmp / "tokens.json"
        tokens.write_text(json.dumps(TOKENS), encoding="utf-8")
        base = {k: v for k, v in os.environ.items() if not k.startswith("EAER_")}   # 不继承本机的模型配置与密钥
        self.env = {**base, "EAER_DATABASE_URL": db_url, "EAER_STORAGE_DIR": str(tmp / "files"),
                    "EAER_TOKENS_FILE": str(tokens), "EAER_OPENSEARCH_URL": opensearch_url, "EAER_INDEX_ALIAS": alias,
                    "EAER_ADMISSION": "off", "EAER_POLL_SECONDS": "0.2", "PYTHONUNBUFFERED": "1",
                    **model_env, **(extra or {})}
        self.logs = tmp / "logs"
        self.logs.mkdir(exist_ok=True)
        self.procs: dict[str, subprocess.Popen] = {}

    def _spawn(self, name: str, args: list[str]) -> subprocess.Popen:
        log = open(self.logs / f"{name}.log", "ab")
        p = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT)
        self.procs[name] = p
        return p

    def start_api(self, name: str = "api") -> str:
        port = free_port()
        self._spawn(name, ["-m", "uvicorn", "eaer.api.main:app", "--host", "127.0.0.1", "--port", str(port),
                           "--log-level", "warning"])
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            self.check_alive(name)
            try:
                if httpx.get(url + "/v1/review/tasks", headers=H("tok-rv"), timeout=2).status_code == 200:
                    return url
            except httpx.TransportError:
                pass
            time.sleep(0.2)
        raise AssertionError(f"接口进程 {name} 30 秒内没有就绪：\n{self.log(name)}")

    def start_worker(self, name: str = "worker") -> subprocess.Popen:
        p = self._spawn(name, ["-m", "eaer.ingestion"])
        time.sleep(0.5)
        self.check_alive(name)
        return p

    def stop(self, name: str, *, kill: bool = False) -> None:
        p = self.procs.pop(name)
        p.send_signal(signal.SIGKILL if kill else signal.SIGTERM)
        p.wait(10)

    def check_alive(self, name: str) -> None:
        p = self.procs[name]
        if p.poll() is not None:
            raise AssertionError(f"进程 {name} 已退出（{p.returncode}）：\n{self.log(name)}")

    def log(self, name: str) -> str:
        f = self.logs / f"{name}.log"
        return f.read_text(encoding="utf-8", errors="replace")[-4000:] if f.exists() else ""

    def close(self) -> None:
        for name in list(self.procs):
            p = self.procs.pop(name)
            if p.poll() is None:
                p.kill()
                p.wait(10)


def upload(api: str, title: str, *paras: str, principals: str = "R_STAFF") -> dict:
    r = httpx.post(api + "/v1/documents", headers=H("tok-ed"), data={"principals": principals},
                   files={"file": (f"{title}.html", page(title, *paras), "text/html")}, timeout=30)
    assert r.status_code == 201, r.text
    return r.json()


def open_tasks(api: str) -> dict[str, dict]:
    """document_id → 待审核任务。"""
    r = httpx.get(api + "/v1/review/tasks", headers=H("tok-rv"), params={"limit": 200}, timeout=30)
    assert r.status_code == 200, r.text
    return {t["document_id"]: t for t in r.json()["items"]}


def approve(api: str, task_id: str, doc_type: str = "normative") -> dict:
    r = httpx.post(api + "/v1/review/decisions", headers=H("tok-rv"), timeout=30, json={"decisions": [
        {"task_id": task_id, "decision": "approve", "corrections": {"doc_type": doc_type}}]})
    assert r.status_code == 200, r.text
    return r.json()["results"][0]


def ask(api: str, question: str, token: str = "tok-staff", **body) -> httpx.Response:
    return httpx.post(api + "/v1/query", headers=H(token), json={"question": question, **body}, timeout=60)


def wait_until(cond, *, timeout: float = 60, what: str = "条件", stack: Stack | None = None, procs=()) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        for n in procs:
            stack.check_alive(n)
        time.sleep(0.2)
    raise AssertionError(f"{timeout} 秒内没有等到：{what}")
