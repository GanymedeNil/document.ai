"""接口进程入口：uvicorn eaer.api.main:app

环境变量：
  EAER_DATABASE_URL   数据库连接串
  EAER_STORAGE_DIR    精简版本地文件存储目录
  EAER_TOKENS_FILE    参考实现的令牌表（JSON）：
      {"令牌": {"user": "alice", "tenant": "t1", "roles": ["editor"], "principals": ["R_STAFF"]}}
      roles 是接口角色（editor、reviewer）；principals 是文档权限角色，供参考授权实现 LocalAuthorizer 使用，
      可省略（省略的账号无法查看证据）。客户现场分别换成 SSO 与客户的授权实现。
"""
import json
import os

from eaer.adapters.authorizer import LocalAuthorizer
from eaer.api.app import Identity, TokenAuthenticator, create_app
from eaer.storage import LocalFileStore


def _load(path: str) -> tuple[dict[str, Identity], dict[str, dict]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    tokens = {t: Identity(v["user"], v["tenant"], frozenset(v["roles"])) for t, v in raw.items()}
    accounts = {v["user"]: {"tenant": v["tenant"], "principals": v["principals"]}
                for v in raw.values() if v.get("principals")}
    return tokens, accounts


_tokens, _accounts = _load(os.environ["EAER_TOKENS_FILE"])
app = create_app(os.environ["EAER_DATABASE_URL"], LocalFileStore(os.environ["EAER_STORAGE_DIR"]),
                 TokenAuthenticator(_tokens), authorizer=lambda conn: LocalAuthorizer(conn, _accounts))
