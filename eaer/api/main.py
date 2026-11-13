"""接口进程入口：uvicorn eaer.api.main:app

环境变量：
  EAER_DATABASE_URL   数据库连接串
  EAER_STORAGE_DIR    精简版本地文件存储目录
  EAER_TOKENS_FILE    参考实现的令牌表（JSON）：
      {"令牌": {"user": "alice", "tenant": "t1", "roles": ["editor"], "principals": ["R_STAFF"]}}
      roles 是接口角色（editor、reviewer）；principals 是文档权限角色，供参考授权实现 LocalAuthorizer 使用，
      可省略（省略的账号无法查看证据）。客户现场分别换成 SSO 与客户的授权实现。
  提问接口另需：EAER_OPENSEARCH_URL、EAER_INDEX_ALIAS（默认 eaer）、EAER_TEXT_ANALYZER（默认 smartcn），以及
      生成、Embedding、重排三项模型服务（EAER_GEN_*、EAER_EMBED_*、EAER_RERANK_*，见 eaer/adapters/online.py）。
      缺任何一项都启动报错（开发文档 12.1“Fast 无法运行，部署时报错”）；只想先开放上传与审核时
      设置 EAER_DISABLE_QUERY=1，提问接口返回 503。
  EAER_DEEP_ENABLED   默认 1；设为 0 关闭 Deep（准入检查未通过时，开发文档 12.3），需要 Deep 的问题由 Fast 回答并说明。
      模型没有结构化输出或上下文长度低于 thresholds.deep_min_context_tokens 时 Deep 自动关闭（12.1）。
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


def _engine_factory():
    """按部署环境变量构造查询编排器；缺配置直接报错。"""
    if os.environ.get("EAER_DISABLE_QUERY") == "1":
        return None
    from eaer.adapters.online import OnlineModelClient
    from eaer.ingestion.__main__ import segmenter_for
    from eaer.orchestrator.engine import Engine
    from eaer.retrieval.opensearch import OpenSearchIndex, connect
    from eaer.retrieval.service import Retriever

    model = OnlineModelClient.from_env()
    missing = model.missing_for_fast()
    if missing or not os.environ.get("EAER_OPENSEARCH_URL"):
        raise RuntimeError("提问接口缺少配置：" + "、".join(missing + ([] if os.environ.get("EAER_OPENSEARCH_URL")
                                                                     else ["EAER_OPENSEARCH_URL"]))
                           + "。只开放上传与审核时设置 EAER_DISABLE_QUERY=1")
    client = connect(os.environ["EAER_OPENSEARCH_URL"])
    alias = os.environ.get("EAER_INDEX_ALIAS", "eaer")
    seg = segmenter_for(os.environ.get("EAER_TEXT_ANALYZER", "smartcn"))
    embedder = model.embedder()
    deep_enabled = os.environ.get("EAER_DEEP_ENABLED", "1") != "0"     # 准入检查（12.3）未通过时设为 0

    def make(conn, tenant_id):
        row = conn.execute("SELECT thresholds, budgets FROM tenant_config WHERE tenant_id = %s", (tenant_id,)).fetchone()
        retriever = Retriever(conn, OpenSearchIndex(client, alias=alias, segmenter=seg), embedder, model)
        return Engine(conn, retriever, model, LocalAuthorizer(conn, _accounts), store=_store,
                      thresholds=row["thresholds"] if row else None, budgets=row["budgets"] if row else None,
                      deep_enabled=deep_enabled)
    return make


_tokens, _accounts = _load(os.environ["EAER_TOKENS_FILE"])
_store = LocalFileStore(os.environ["EAER_STORAGE_DIR"])
app = create_app(os.environ["EAER_DATABASE_URL"], _store, TokenAuthenticator(_tokens),
                 authorizer=lambda conn: LocalAuthorizer(conn, _accounts), engine=_engine_factory())
