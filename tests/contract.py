"""按 OpenAPI 文件（eaer/api/openapi.yaml）检查实际的请求与响应（开发文档 11、AC-11-1）。

Contract.check(response) 对每个响应检查：
- 请求的方法与路径是文件中声明的某个操作；
- 状态码在该操作中有声明，媒体类型在该状态码下有声明，响应带 X-Request-Id；
- 响应体符合声明的 JSON Schema（格式 uuid、date、date-time 也检查）；错误响应的 request_id 与响应头相同；
  流式响应逐个事件按 x-events 中该事件的 schema 检查；
- 成功（2xx）的请求本身也符合文件：路径参数、查询参数与请求体都通过校验——实现接受了文件不允许的请求，
  就是两者不一致。
检查过的（操作, 状态码, 媒体类型）记在 seen 里，供覆盖检查使用。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from email.parser import BytesParser
from email.policy import HTTP
from typing import Any, Iterator
from urllib.parse import parse_qsl, unquote, urlsplit

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from eaer.api.app import openapi_spec

BASE = "urn:eaer:openapi"
METHODS = ("get", "put", "post", "delete", "patch", "head", "options")
REQUEST_ID = re.compile(r"req_[0-9a-f]{16}")

FORMATS = FormatChecker()
if "date-time" not in FORMATS.checkers:          # 没装 rfc3339-validator 时 jsonschema 不检查 date-time，这里补上
    @FORMATS.checks("date-time", raises=ValueError)
    def _date_time(value: Any) -> bool:
        if not isinstance(value, str):
            return True
        return "T" in value and datetime.fromisoformat(value).tzinfo is not None


def pointer(*parts: str) -> str:
    return "".join("/" + p.replace("~", "~0").replace("/", "~1") for p in parts)


@dataclass
class Operation:
    id: str
    method: str
    template: str
    regex: re.Pattern
    spec: dict
    ptr: str

    def path_params(self, path: str) -> dict[str, str]:
        return {k: unquote(v) for k, v in self.regex.fullmatch(path).groupdict().items()}


def parse_sse(text: str) -> list[tuple[str, Any]]:
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def parse_multipart(content_type: str, body: bytes) -> dict[str, Any]:
    """multipart/form-data → {字段名: 值}；文件字段的值记为字符串占位。"""
    msg = BytesParser(policy=HTTP).parsebytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
    out: dict[str, Any] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if part.get_param("filename", header="content-disposition") is not None:
            out[name] = "<file>"
        else:
            out[name] = part.get_payload(decode=True).decode("utf-8")
    return out


class Contract:
    def __init__(self, spec: dict | None = None):
        self.spec = spec or openapi_spec()
        self.registry = Registry().with_resource(
            BASE, Resource.from_contents(self.spec, default_specification=DRAFT202012))
        self.operations: list[Operation] = []
        for path, item in self.spec["paths"].items():
            rx = re.compile(re.sub(r"\\\{(\w+)\\\}", r"(?P<\1>[^/]+)", re.escape(path)))
            for m in METHODS:
                if m in item:
                    op = item[m]
                    self.operations.append(Operation(op["operationId"], m.upper(), path, rx, op, pointer("paths", path, m)))
        self.seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self._validators: dict[str, Draft202012Validator] = {}

    # ---------- 文件内的查找 ----------

    def at(self, ptr: str) -> Any:
        node: Any = self.spec
        for part in ptr.lstrip("/").split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            node = node[int(part)] if isinstance(node, list) else node[part]
        return node

    def deref(self, obj: dict, ptr: str) -> tuple[dict, str]:
        """解开 OpenAPI 层面的 $ref（响应、参数等），返回对象及其在文件中的位置。"""
        while "$ref" in obj:
            ptr = obj["$ref"].removeprefix("#")
            obj = self.at(ptr)
        return obj, ptr

    def errors(self, ptr: str, instance: Any) -> list[str]:
        v = self._validators.get(ptr)
        if v is None:
            v = self._validators[ptr] = Draft202012Validator({"$ref": f"{BASE}#{ptr}"}, registry=self.registry,
                                                             format_checker=FORMATS)
        return [f"{'/'.join(map(str, e.absolute_path)) or '（根）'}：{e.message}" for e in v.iter_errors(instance)]

    def find(self, method: str, path: str) -> Operation | None:
        hits = [o for o in self.operations if o.method == method and o.regex.fullmatch(path)]
        return min(hits, key=lambda o: o.template.count("{")) if hits else None

    def by_id(self, op_id: str) -> Operation:
        return next(o for o in self.operations if o.id == op_id)

    def parameters(self, op: Operation) -> Iterator[tuple[dict, str]]:
        for i, p in enumerate(op.spec.get("parameters", [])):
            yield self.deref(p, f"{op.ptr}/parameters/{i}")

    def declared(self, op: Operation) -> set[tuple[str, str]]:
        out = set()
        for status, r in op.spec["responses"].items():
            r, _ = self.deref(r, "")
            out |= {(status, media) for media in r.get("content", {})}
        return out

    # ---------- 请求 ----------

    def request_errors(self, op: Operation, method: str, url: str, headers, body: bytes) -> list[str]:
        """请求按文件检查出的问题；空表示文件允许这个请求。"""
        parts = urlsplit(url)
        problems: list[str] = []
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        path_values = op.path_params(parts.path)
        for p, ptr in self.parameters(op):
            src = path_values if p["in"] == "path" else query if p["in"] == "query" else {}
            if p["name"] not in src:
                if p.get("required"):
                    problems.append(f"缺少参数 {p['name']}")
                continue
            value: Any = src[p["name"]]
            if p["schema"].get("type") == "integer":
                try:
                    value = int(value)
                except ValueError:
                    pass
            problems += [f"参数 {p['name']}：{e}" for e in self.errors(f"{ptr}/schema", value)]
        rb = op.spec.get("requestBody")
        if rb is None:
            return problems
        media = (headers.get("content-type") or "").split(";")[0].strip()
        if media not in rb["content"]:
            return problems + [f"请求体的媒体类型 {media or '（无）'} 未声明"]
        if media == "application/json":
            try:
                instance = json.loads(body)
            except ValueError:
                return problems + ["请求体不是 JSON"]
        else:
            instance = parse_multipart(headers["content-type"], body)
        schema_ptr = f"{op.ptr}/requestBody/content/{pointer(media)[1:]}/schema"
        return problems + [f"请求体 {e}" for e in self.errors(schema_ptr, instance)]

    # ---------- 响应 ----------

    def check(self, resp) -> Operation:
        req = resp.request
        url = str(req.url)
        path = urlsplit(url).path
        op = self.find(req.method, path)
        assert op is not None, f"{req.method} {path} 不在 OpenAPI 文件中"
        status = str(resp.status_code)
        where = f"{op.id} {req.method} {path} → {status}"
        assert status in op.spec["responses"], f"{where}：状态码未在文件中声明。响应：{resp.text[:300]}"
        r, rptr = self.deref(op.spec["responses"][status], f"{op.ptr}/responses/{status}")
        rid = resp.headers.get("X-Request-Id", "")
        assert REQUEST_ID.fullmatch(rid), f"{where}：X-Request-Id 缺失或格式不对：{rid!r}"
        assert "X-Request-Id" in r.get("headers", {}), f"{where}：文件中该响应没有声明 X-Request-Id"
        media = resp.headers.get("content-type", "").split(";")[0].strip()
        assert media in r.get("content", {}), f"{where}：媒体类型 {media} 未在文件中声明"
        schema_ptr = f"{rptr}/content/{pointer(media)[1:]}/schema"
        if media == "application/json":
            body = resp.json()
            errs = self.errors(schema_ptr, body)
            assert not errs, f"{where}：响应体不符合文件：{errs[:5]}\n{json.dumps(body, ensure_ascii=False)[:600]}"
            if resp.status_code >= 400:
                assert body["request_id"] == rid, f"{where}：错误响应体的 request_id 与响应头不同"
        elif media == "text/event-stream":
            events = parse_sse(resp.text)
            declared = r["content"][media]["x-events"]
            assert events and events[-1][0] in ("done", "error"), f"{where}：事件流没有以 done 或 error 结束"
            for name, data in events:
                assert name in declared, f"{where}：事件 {name} 未在 x-events 中声明"
                errs = self.errors(f"{rptr}/content/{pointer(media)[1:]}/x-events/{name}", data)
                assert not errs, f"{where}：事件 {name} 的 data 不符合文件：{errs[:5]}"
        if resp.status_code < 300:
            problems = self.request_errors(op, req.method, url, req.headers, req.content)
            assert not problems, f"{where}：实现接受了文件不允许的请求：{problems}"
        self.seen[op.id].add((status, media))
        return op


class Checked:
    """包一层 TestClient：每个响应都经 Contract.check。"""

    def __init__(self, client, contract: Contract):
        self.client, self.contract = client, contract

    def request(self, method: str, url: str, **kw):
        resp = self.client.request(method, url, **kw)
        self.contract.check(resp)
        return resp

    def get(self, url: str, **kw):
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw):
        return self.request("POST", url, **kw)
