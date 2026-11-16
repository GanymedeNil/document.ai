"""冒烟测试集（开发文档 13.5 AC-13-1、13-2、13-4，3.4 AC-3-3）：对一个已部署的环境走一遍主流程，全部通过才算部署成功。

    EAER_SMOKE_EDITOR=… EAER_SMOKE_REVIEWER=… EAER_SMOKE_USER=… python -m eaer.smoke --api http://localhost:8000 [--timeout 180]

令牌从环境变量读（也可用 --editor 等参数，但参数会出现在进程列表里，且以“-”开头的令牌会被当成选项）。

只经对外接口（11.2），不碰数据库，可以在任何能访问接口的机器上运行；只用标准库。
1. 上传一份带本次运行编号的制度（内容每次不同，不会被去重），等待后台解析后审核通过；
2. 提问，直到回答引用了这份制度并答出其中的取值（等待后台写入索引）；
3. 点开引用（证据接口）与会话接口；
4. 归档这份制度，再问——不再引用它；
5. 取消归档，再问——又能引用；最后再归档，不在部署环境里留下可检索的测试数据。
令牌对应的账号：editor（上传、归档）、reviewer（审核）、user（文档权限角色含 --principal，默认 R_STAFF）。
每一步的结果与耗时打印成 JSON；全部通过返回 0，否则返回 1。

备份恢复与索引重建演练（AC-13-3、13-4）另用三个子命令：
    python -m eaer.smoke seed --api … --editor … --reviewer …         上传演练数据：同一制度的两个版本（2026-01-01、2026-07-01 生效）
    python -m eaer.smoke snapshot --api … --out 恢复前.json            对演练数据按当前、某天实际生效、截至某天系统所知提问，
                                                                     记下是否拒答、引用的版本号、回答中的取值与引用的证据 id
    python -m eaer.smoke diff 恢复前.json 恢复后.json                 两次快照一致返回 0
比较的是由数据决定的部分：是否拒答、引用哪个版本、答出的取值。引用了哪几条证据由生成模型决定（同一版本里多引一条
“施行日期”也是对的），温度为 0 也会变，只记下、不比较；检索层面逐条相同由 tests/test_rebuild_drill.py 验证。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid


class Api:
    def __init__(self, base: str, timeout: float = 120.0):
        self.base, self.timeout = base.rstrip("/"), timeout

    def call(self, method: str, path: str, token: str, *, body=None, data: bytes | None = None,
             content_type: str | None = None) -> tuple[int, dict]:
        headers = {"Authorization": f"Bearer {token}"}
        if body is not None:
            data, content_type = json.dumps(body).encode("utf-8"), "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw or b"{}")
            except ValueError:
                return e.code, {"raw": raw.decode("utf-8", "replace")[:200]}

    def upload(self, token: str, filename: str, content: bytes, fields: dict[str, str]) -> tuple[int, dict]:
        boundary = uuid.uuid4().hex
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8")
                 for k, v in fields.items()]
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                     f"Content-Type: text/html\r\n\r\n".encode("utf-8") + content + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        return self.call("POST", "/v1/documents", token, data=b"".join(parts),
                         content_type=f"multipart/form-data; boundary={boundary}")


class SmokeFailed(Exception):
    pass


def run(api: Api, *, editor: str, reviewer: str, user: str, principal: str = "R_STAFF", timeout: float = 180.0,
        log=lambda s: None) -> list[dict]:
    run_id = uuid.uuid4().hex[:6]
    days = 10 + int(run_id, 16) % 80
    title = f"冒烟测试{run_id}号设备借用办法"
    body = (f"<h1>{title}</h1><p>第一条　为规范冒烟测试{run_id}号设备的借用，制定本办法。</p>"
            f"<p>第二条　冒烟测试{run_id}号设备的借用期限为{days}天。</p><p>第三条　本办法自2026年1月1日起施行。</p>")
    question = f"冒烟测试{run_id}号设备的借用期限是多少天"
    steps: list[dict] = []

    def step(name: str, fn):
        t0 = time.monotonic()
        try:
            detail = fn()
            steps.append({"step": name, "ok": True, "seconds": round(time.monotonic() - t0, 2), "detail": detail})
            log(f"通过：{name}（{time.monotonic() - t0:.1f} 秒）")
        except Exception as e:  # noqa: BLE001 — 记下失败的一步并停止
            steps.append({"step": name, "ok": False, "seconds": round(time.monotonic() - t0, 2),
                          "detail": f"{type(e).__name__}: {e}"})
            log(f"失败：{name}：{e}")
            raise SmokeFailed(name)

    def wait(what: str, cond, limit: float = timeout):
        deadline = time.monotonic() + limit
        last = None
        while time.monotonic() < deadline:
            ok, last = cond()
            if ok:
                return last
            time.sleep(2)
        raise SmokeFailed(f"{limit:.0f} 秒内没有等到：{what}（最后一次：{str(last)[:200]}）")

    state: dict = {}

    def upload():
        code, b = api.upload(editor, f"smoke-{run_id}.html", body.encode("utf-8"),
                             {"principals": principal, "source_ref": f"smoke-{run_id}"})
        if code != 201:
            raise SmokeFailed(f"上传返回 {code}：{b}")
        state["doc"] = b["document_id"]
        return {"document_id": b["document_id"]}

    def approve():
        def try_approve():
            code, b = api.call("GET", "/v1/review/tasks?limit=200", reviewer)
            if code != 200:
                return False, f"任务列表 {code}"
            task = next((t for t in b["items"] if t["document_id"] == state["doc"]), None)
            if task is None:
                return False, "任务还没出现"
            code, b = api.call("POST", "/v1/review/decisions", reviewer, body={"decisions": [
                {"task_id": task["task_id"], "decision": "approve", "corrections": {"doc_type": "normative"}}]})
            r = b.get("results", [{}])[0]
            return bool(r.get("ok")), r.get("error") or b
        return wait("后台解析完成并审核通过", try_approve)

    def ask(expect_cited: bool):
        def once():
            code, b = api.call("POST", "/v1/query", user, body={"question": question})
            if code != 200 or b.get("status") != "completed":
                return False, f"{code} {b.get('status')} {b.get('code')}"
            docs = {c["document_id"] for k in b["claims"] for c in k["citations"]}
            cited = state["doc"] in docs and str(days) in (b["answer"] or "")
            state["last"] = b
            return (cited if expect_cited else state["doc"] not in docs), {
                "mode_path": b["mode_path"], "abstained": b["abstained"], "cited": state["doc"] in docs}
        return once

    def evidence_and_session():
        b = state["last"]
        cite = next(c for k in b["claims"] for c in k["citations"] if c["document_id"] == state["doc"])
        code, ev = api.call("GET", f"/v1/evidence/{cite['evidence_id']}", user)
        if code != 200:
            raise SmokeFailed(f"证据接口返回 {code}")
        code, s = api.call("GET", f"/v1/sessions/{b['session_id']}", user)
        if code != 200:
            raise SmokeFailed(f"会话接口返回 {code}")
        return {"evidence_id": cite["evidence_id"], "session_items": len(s.get("items", []))}

    def archive(on: bool):
        code, b = api.call("POST", f"/v1/documents/{state['doc']}/{'archive' if on else 'unarchive'}", editor)
        if code != 200:
            raise SmokeFailed(f"{'归档' if on else '取消归档'}返回 {code}：{b}")
        return b

    try:
        step("上传", upload)
        step("后台解析后审核通过", approve)
        step("提问并引用新文档（等待写入索引）", lambda: wait("回答引用新文档", ask(True)))
        step("证据接口与会话接口", evidence_and_session)
        step("归档", lambda: archive(True))
        step("归档后提问不再引用", lambda: wait("不再引用", ask(False)))
        step("取消归档", lambda: archive(False))
        step("取消归档后又能引用", lambda: wait("重新引用", ask(True)))
        step("清理：再次归档", lambda: archive(True))
    except SmokeFailed:
        pass
    return steps


DRILL_REF = "drill-travel"
DRILL_V1 = ("<h1>演练差旅住宿标准</h1><p>第一条　为规范演练差旅住宿，制定本标准。</p><p>第二条　演练一线城市住宿上限为每晚600元。</p>"
            "<p>第三条　本标准自2026年1月1日起施行。</p>")
DRILL_V2 = ("<h1>演练差旅住宿标准</h1><p>第一条　为规范演练差旅住宿，制定本标准。</p><p>第二条　演练一线城市住宿上限为每晚680元。</p>"
            "<p>第三条　本标准自2026年7月1日起施行，原标准同时废止。</p>")
DRILL_QUESTIONS = [
    {"question": "演练一线城市住宿上限是多少"},
    {"question": "演练一线城市住宿上限是多少", "as_of_valid": "2026-03-01"},
    {"question": "演练一线城市住宿上限是多少", "as_of_valid": "2026-08-01"},
    {"question": "演练一线城市住宿上限是多少", "as_of_system": "2026-01-15"},
]


def seed(api: Api, *, editor: str, reviewer: str, principal: str = "R_STAFF", timeout: float = 180.0) -> list[str]:
    """上传并审核通过演练数据（两个版本，依次发布）；已存在（内容相同被去重）也照常返回。"""
    out = []
    for k, body in enumerate((DRILL_V1, DRILL_V2), 1):
        code, b = api.upload(editor, f"{DRILL_REF}-v{k}.html", body.encode("utf-8"),
                             {"principals": principal, "source_ref": DRILL_REF})
        if code not in (200, 201):
            raise SmokeFailed(f"上传第 {k} 版返回 {code}：{b}")
        out.append(b["version_id"])
        deadline = time.monotonic() + timeout
        while True:
            _, tasks = api.call("GET", "/v1/review/tasks?limit=200", reviewer)
            task = next((t for t in tasks.get("items", []) if t.get("version_id") == b["version_id"]), None)
            if task is None and b.get("duplicate"):
                break                                            # 之前已审核通过
            if task is not None:
                _, r = api.call("POST", "/v1/review/decisions", reviewer, body={"decisions": [
                    {"task_id": task["task_id"], "decision": "approve", "corrections": {"doc_type": "normative"}}]})
                if r.get("results", [{}])[0].get("ok"):
                    break
            if time.monotonic() > deadline:
                raise SmokeFailed(f"第 {k} 版 {timeout:.0f} 秒内没能审核通过")
            time.sleep(2)
    return out


def snapshot(api: Api, *, user: str) -> list[dict]:
    rows = []
    for q in DRILL_QUESTIONS:
        code, b = api.call("POST", "/v1/query", user, body=q)
        cites = sorted({(c["evidence_id"], c.get("version_no")) for k in b.get("claims", []) for c in k["citations"]})
        rows.append({**q, "status": code, "abstained": b.get("abstained"),
                     "versions": sorted({v for _, v in cites if v is not None}),
                     "values": sorted(set(re.findall(r"\d+元", b.get("answer") or ""))),
                     "evidence": [e for e, _ in cites]})
    return rows


def diff(a: list[dict], b: list[dict]) -> list[str]:
    keys = ("status", "abstained", "versions", "values")
    return [f"{x['question']}（{x.get('as_of_valid') or x.get('as_of_system') or '当前'}）：{ {k: x[k] for k in keys} } → "
            f"{ {k: y[k] for k in keys} }" for x, y in zip(a, b) if any(x[k] != y[k] for k in keys)] + (
        ["两次快照的问题数不同"] if len(a) != len(b) else [])


def _token_args(p, *roles: str) -> None:
    import os
    for role in roles:
        env = f"EAER_SMOKE_{role.upper()}"
        p.add_argument(f"--{role}", default=os.environ.get(env), required=not os.environ.get(env),
                       help=f"{role} 的令牌；默认取环境变量 {env}")


def drill_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="eaer.smoke")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("seed")
    s.add_argument("--api", required=True)
    _token_args(s, "editor", "reviewer")
    s.add_argument("--principal", default="R_STAFF")
    n = sub.add_parser("snapshot")
    n.add_argument("--api", required=True)
    _token_args(n, "user")
    n.add_argument("--out", required=True)
    d = sub.add_parser("diff")
    d.add_argument("first")
    d.add_argument("second")
    args = p.parse_args(argv)
    if args.command == "seed":
        print(json.dumps({"versions": seed(Api(args.api), editor=args.editor, reviewer=args.reviewer,
                                           principal=args.principal)}, ensure_ascii=False))
        return 0
    if args.command == "snapshot":
        rows = snapshot(Api(args.api), user=args.user)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    with open(args.first, encoding="utf-8") as f1, open(args.second, encoding="utf-8") as f2:
        problems = diff(json.load(f1), json.load(f2))
    print("两次快照一致" if not problems else "不一致：\n" + "\n".join(problems))
    return 0 if not problems else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("seed", "snapshot", "diff"):
        return drill_main(argv)
    p = argparse.ArgumentParser(prog="eaer.smoke", description="部署后的冒烟测试集")
    p.add_argument("--api", required=True)
    _token_args(p, "editor", "reviewer", "user")
    p.add_argument("--principal", default="R_STAFF")
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args(argv)
    t0 = time.monotonic()
    steps = run(Api(args.api), editor=args.editor, reviewer=args.reviewer, user=args.user, principal=args.principal,
                timeout=args.timeout, log=lambda s: print(s, file=sys.stderr))
    ok = len(steps) == 9 and all(s["ok"] for s in steps)
    print(json.dumps({"ok": ok, "seconds": round(time.monotonic() - t0, 1), "steps": steps}, ensure_ascii=False,
                     indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
