"""数据源接口与参考实现（开发文档 11.3）。

参考实现 LocalFolderSource：本地文件夹，按修改时间与大小发现变化；每个目录下的 _acl.yaml 声明可访问的角色，
对该目录及其子目录生效，子目录的 _acl.yaml 覆盖上级。没有适用的 _acl.yaml、或角色为空的文件一律不列出
（默认拒绝）；权限被撤销到空时该文件以“删除”出现。

游标是不透明字符串。参考实现把已同步的快照压缩进游标，不需要额外的状态存储；文件很多时游标会变大，
生产数据源应把状态放在服务端。
"""
from __future__ import annotations

import base64
import json
import os
import re
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

ACL_FILE = "_acl.yaml"


class SourceNotFound(Exception):
    """数据源中没有这份文档（或调用方无权知道它的存在）。"""


class AclFormatError(Exception):
    pass


@dataclass(frozen=True)
class Change:
    kind: Literal["upsert", "delete", "acl"]     # 新增或内容更新、删除、仅权限变更
    source_ref: str
    modified_at: datetime | None = None


@dataclass(frozen=True)
class ChangePage:
    changes: list[Change]
    cursor: str               # 下一次调用传入；处理完本页再保存，处理失败可用旧游标重来
    has_more: bool


@dataclass(frozen=True)
class RawDocument:
    source_ref: str
    filename: str
    data: bytes
    created_at: datetime | None       # 源系统创建时间；拿不到时为 None
    modified_at: datetime | None


@dataclass(frozen=True)
class AclPolicy:
    principals: tuple[str, ...]


class DataSource(Protocol):
    max_revocation_delay_seconds: int                     # 撤权生效的最长延迟
    def list_changes(self, cursor: str | None) -> ChangePage: ...
    def fetch(self, source_ref: str) -> RawDocument: ...
    def read_acl(self, source_ref: str) -> AclPolicy: ...


# ---------- 参考实现 ----------

_FLOW = re.compile(r"^principals\s*:\s*\[(?P<items>.*)\]\s*$")


def parse_acl(text: str) -> tuple[str, ...]:
    """只接受一个 principals 键，流式 [a, b] 或块式列表；其余写法报错，避免把写错的权限当成空权限。"""
    lines = [ln.split("#", 1)[0].rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        raise AclFormatError("_acl.yaml 为空")
    m = _FLOW.match(lines[0].strip())
    if m and len(lines) == 1:
        items = [i.strip().strip("'\"") for i in m.group("items").split(",")]
        return tuple(sorted({i for i in items if i}))
    if lines[0].strip() == "principals:" and all(ln.strip().startswith("- ") for ln in lines[1:]):
        return tuple(sorted({ln.strip()[2:].strip().strip("'\"") for ln in lines[1:]} - {""}))
    raise AclFormatError(f"无法解析 _acl.yaml：{lines[0]!r}；只支持 principals: [角色, …] 或块式列表")


def _encode(snapshot: dict) -> str:
    return base64.urlsafe_b64encode(zlib.compress(json.dumps(snapshot, sort_keys=True).encode())).decode()


def _decode(cursor: str | None) -> dict:
    if not cursor:
        return {}
    try:
        return json.loads(zlib.decompress(base64.urlsafe_b64decode(cursor.encode())))
    except Exception as e:  # noqa: BLE001
        raise ValueError("游标无效") from e


class LocalFolderSource:
    def __init__(self, root: str | Path, *, page_size: int = 100, max_revocation_delay_seconds: int = 60):
        self.root = Path(root).resolve()
        self.page_size = page_size
        self.max_revocation_delay_seconds = max_revocation_delay_seconds

    # 文件名以 _ 或 . 开头的不是文档（_acl.yaml、隐藏文件）
    def _scan(self) -> dict[str, list]:
        out: dict[str, list] = {}
        acl_cache: dict[Path, tuple[str, ...] | None] = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "_")))
            for name in sorted(filenames):
                if name.startswith((".", "_")):
                    continue
                p = Path(dirpath) / name
                principals = self._acl_for(p.parent, acl_cache)
                if not principals:
                    continue
                st = p.stat()
                out[p.relative_to(self.root).as_posix()] = [st.st_mtime_ns, st.st_size, list(principals)]
        return out

    def _acl_for(self, directory: Path, cache: dict) -> tuple[str, ...] | None:
        if directory in cache:
            return cache[directory]
        f = directory / ACL_FILE
        if f.is_file():
            val: tuple[str, ...] | None = parse_acl(f.read_text(encoding="utf-8"))
        elif directory == self.root:
            val = None
        else:
            val = self._acl_for(directory.parent, cache)
        cache[directory] = val
        return val

    def _path(self, source_ref: str) -> Path:
        p = (self.root / source_ref).resolve()
        if self.root not in p.parents or not p.is_file() or p.name.startswith((".", "_")):
            raise SourceNotFound(source_ref)
        return p

    def list_changes(self, cursor: str | None) -> ChangePage:
        old, cur = _decode(cursor), self._scan()
        changes: list[tuple[str, Change]] = []
        for ref, (mtime, size, principals) in cur.items():
            ts = datetime.fromtimestamp(mtime / 1e9, timezone.utc)
            prev = old.get(ref)
            if prev is None or prev[:2] != [mtime, size]:
                changes.append((ref, Change("upsert", ref, ts)))
            elif prev[2] != principals:
                changes.append((ref, Change("acl", ref, ts)))
        changes += [(ref, Change("delete", ref)) for ref in old if ref not in cur]
        changes.sort(key=lambda c: c[0])
        page, rest = changes[: self.page_size], changes[self.page_size:]
        nxt = dict(old)
        for ref, ch in page:
            if ch.kind == "delete":
                nxt.pop(ref, None)
            else:
                nxt[ref] = cur[ref]
        return ChangePage([c for _, c in page], _encode(nxt), bool(rest))

    def fetch(self, source_ref: str) -> RawDocument:
        p = self._path(source_ref)
        if not self._acl_for(p.parent, {}):
            raise SourceNotFound(source_ref)
        st = p.stat()
        birth = getattr(st, "st_birthtime", None)
        return RawDocument(source_ref, p.name, p.read_bytes(),
                           datetime.fromtimestamp(birth, timezone.utc) if birth else None,
                           datetime.fromtimestamp(st.st_mtime, timezone.utc))

    def read_acl(self, source_ref: str) -> AclPolicy:
        p = self._path(source_ref)
        principals = self._acl_for(p.parent, {})
        if not principals:
            raise SourceNotFound(source_ref)
        return AclPolicy(principals)
