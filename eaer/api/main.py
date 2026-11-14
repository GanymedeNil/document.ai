"""接口进程入口：uvicorn eaer.api.main:app

环境变量：
  EAER_DATABASE_URL   数据库连接串
  EAER_STORAGE_DIR    精简版本地文件存储目录
  EAER_TOKENS_FILE    参考实现的令牌表（JSON）：
      {"令牌": {"user": "alice", "tenant": "t1", "roles": ["editor"], "principals": ["R_STAFF"]}}
      roles 是接口角色（editor、reviewer）；principals 是文档权限角色，供参考授权实现 LocalAuthorizer 使用，
      可省略（省略的账号无法查看证据）。客户现场分别换成 SSO 与客户的授权实现。
  提问接口另需：EAER_OPENSEARCH_URL、EAER_INDEX_ALIAS（默认 eaer）、EAER_TEXT_ANALYZER（默认 standard），以及
      生成、Embedding、重排三项模型服务（EAER_GEN_*、EAER_EMBED_*、EAER_RERANK_*，见 eaer/adapters/online.py）。
      缺任何一项都启动报错（开发文档 12.1“Fast 无法运行，部署时报错”）；只想先开放上传与审核时
      设置 EAER_DISABLE_QUERY=1，提问接口返回 503。
  EAER_DEEP_ENABLED   默认 1；设为 0 关闭 Deep（准入检查未通过时，开发文档 12.3），需要 Deep 的问题由 Fast 回答并说明。
      模型没有结构化输出或上下文长度低于 thresholds.deep_min_context_tokens 时 Deep 自动关闭（12.1）。
  EAER_QUERY_THREADS  同时执行的提问数，默认 8。提问超过同步时限（budgets.sync_seconds，默认 30 秒）转为后台任务，
      由后台任务进程（python -m eaer.ingestion）接着执行，它需要与本进程相同的查询配置（见 eaer/runtime.py）。
  EAER_TRACE_EXPORTER none（默认）/ console / otlp：每次查询一条 trace（13.3，见 eaer/observability.py）。
  EAER_RATE_QUERY_PER_MINUTE / EAER_RATE_UPLOAD_PER_MINUTE  每个用户每分钟的提问 / 上传上限，默认 30 / 120，0 不限（11.1 的 429）。
"""
import os
import socket

from eaer.api.app import TokenAuthenticator, create_app
from eaer.api.ratelimit import Limits
from eaer.observability import configure as configure_tracing
from eaer.runtime import authorizer_factory, engine_factory, load_tokens
from eaer.storage import LocalFileStore

configure_tracing(dict(os.environ), "eaer-api")
_tokens, _accounts = load_tokens(os.environ["EAER_TOKENS_FILE"])
_store = LocalFileStore(os.environ["EAER_STORAGE_DIR"])
app = create_app(os.environ["EAER_DATABASE_URL"], _store, TokenAuthenticator(_tokens),
                 authorizer=authorizer_factory(_accounts), engine=engine_factory(dict(os.environ), _accounts, _store),
                 query_threads=int(os.environ.get("EAER_QUERY_THREADS", "8")),
                 owner=f"{socket.gethostname()}:{os.getpid()}", rate_limits=Limits.from_env(dict(os.environ)))
