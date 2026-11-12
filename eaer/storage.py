"""文件存储接口（开发文档 3.3、13.1）。精简版用本地文件系统，集群版换成 S3 兼容存储，业务代码不变。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol


class FileStore(Protocol):
    def put(self, tenant_id: str, data: bytes, filename: str) -> str: ...
    def get(self, uri: str) -> bytes: ...
    def delete(self, uri: str) -> None: ...
    def exists(self, uri: str) -> bool: ...


class LocalFileStore:
    """按租户和内容哈希存放原件：local://<租户>/<哈希>/<文件名>。"""

    def __init__(self, base: str | Path):
        self.base = Path(base)

    def _path(self, uri: str) -> Path:
        if not uri.startswith("local://"):
            raise ValueError(f"不是本地存储的地址：{uri}")
        p = (self.base / uri.removeprefix("local://")).resolve()
        if self.base.resolve() not in p.parents:
            raise ValueError(f"地址越出存储目录：{uri}")
        return p

    def put(self, tenant_id: str, data: bytes, filename: str) -> str:
        safe = re.sub(r"[^\w.\-]+", "_", Path(filename).name) or "file"
        digest = hashlib.sha256(data).hexdigest()
        uri = f"local://{re.sub(r'[^\w.-]+', '_', tenant_id)}/{digest}/{safe}"
        p = self._path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return uri

    def get(self, uri: str) -> bytes:
        return self._path(uri).read_bytes()

    def delete(self, uri: str) -> None:
        p = self._path(uri)
        if p.exists():
            p.unlink()
            try:
                p.parent.rmdir()
            except OSError:
                pass

    def exists(self, uri: str) -> bool:
        return self._path(uri).exists()
