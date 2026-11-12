"""接口进程入口：uvicorn eaer.api.main:app

环境变量：
  EAER_DATABASE_URL   数据库连接串
  EAER_STORAGE_DIR    精简版本地文件存储目录
  EAER_TOKENS_FILE    参考实现的令牌表（JSON：{"令牌": {"user": "...", "tenant": "...", "roles": ["editor"]}}）
"""
import json
import os

from eaer.api.app import Identity, TokenAuthenticator, create_app
from eaer.storage import LocalFileStore


def _tokens(path: str) -> dict[str, Identity]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {t: Identity(v["user"], v["tenant"], frozenset(v["roles"])) for t, v in raw.items()}


app = create_app(os.environ["EAER_DATABASE_URL"], LocalFileStore(os.environ["EAER_STORAGE_DIR"]),
                 TokenAuthenticator(_tokens(os.environ["EAER_TOKENS_FILE"])))
