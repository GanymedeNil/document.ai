"""部署时的公共装配：接口进程与后台任务进程用同一套环境变量构造授权实现与查询编排器。

后台任务进程要接着执行转入后台、或执行者中断的提问（8.5），所以它与接口进程需要同样的查询配置：
EAER_OPENSEARCH_URL、EAER_INDEX_ALIAS、EAER_TEXT_ANALYZER、模型服务（EAER_GEN_*、EAER_EMBED_*、EAER_RERANK_*），
以及授权实现（参考实现读 EAER_TOKENS_FILE 中的文档权限角色）。缺任何一项都启动报错（12.1）；
不开放提问时设置 EAER_DISABLE_QUERY=1。EAER_DEEP_ENABLED=0 关闭 Deep。
准入（12.3）：Verified 与 Deep 只对通过准入检查的生成模型开启（按租户取最近一次准入结论，见 eaer/orchestrator/admission.py；
准入检查用 python -m eaer.evalrun.admission 运行）。没有准入记录的模型两者都关闭并在回答中说明；
EAER_ADMISSION=off 跳过准入（只用于本地开发）。
"""
from __future__ import annotations

import json
import os
from typing import Callable

from eaer.domain.identity import Identity


class ConfigError(Exception):
    pass


def load_tokens(path: str) -> tuple[dict[str, Identity], dict[str, dict]]:
    """参考实现的令牌表：返回 (令牌 → 身份, 用户 → 租户与文档权限角色)。"""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    tokens = {t: Identity(v["user"], v["tenant"], frozenset(v["roles"])) for t, v in raw.items()}
    accounts = {v["user"]: {"tenant": v["tenant"], "principals": v["principals"]}
                for v in raw.values() if v.get("principals")}
    return tokens, accounts


def authorizer_factory(accounts: dict[str, dict]) -> Callable:
    from eaer.adapters.authorizer import LocalAuthorizer
    return lambda conn: LocalAuthorizer(conn, accounts)


def segmenter_for(analyzer: str):
    if analyzer == "jieba":
        from eaer.retrieval.segment import JiebaSegmenter
        return JiebaSegmenter(os.environ.get("EAER_JIEBA_USER_DICT") or None)
    return None


def engine_factory(env: dict[str, str], accounts: dict[str, dict], store) -> Callable | None:
    """按环境变量构造查询编排器的工厂 (conn, tenant_id) -> Engine；EAER_DISABLE_QUERY=1 时为 None，缺配置报错。"""
    if env.get("EAER_DISABLE_QUERY") == "1":
        return None
    from eaer.adapters.authorizer import LocalAuthorizer
    from eaer.adapters.online import OnlineModelClient
    from eaer.orchestrator.engine import Engine
    from eaer.retrieval.opensearch import OpenSearchIndex, connect
    from eaer.retrieval.service import Retriever

    model = OnlineModelClient.from_env(env)
    missing = model.missing_for_fast() + ([] if env.get("EAER_OPENSEARCH_URL") else ["EAER_OPENSEARCH_URL"])
    if missing:
        raise ConfigError("提问服务缺少配置：" + "、".join(missing) + "。不开放提问时设置 EAER_DISABLE_QUERY=1")
    client = connect(env["EAER_OPENSEARCH_URL"])
    alias = env.get("EAER_INDEX_ALIAS", "eaer")
    seg = segmenter_for(env.get("EAER_TEXT_ANALYZER", "smartcn"))
    embedder = model.embedder()
    deep_enabled = env.get("EAER_DEEP_ENABLED", "1") != "0"
    admission_required = env.get("EAER_ADMISSION", "required") != "off"

    def make(conn, tenant_id):
        from eaer.orchestrator.admission import gate

        row = conn.execute("SELECT thresholds, budgets FROM tenant_config WHERE tenant_id = %s", (tenant_id,)).fetchone()
        g = gate(conn, tenant_id, model.gen.model, required=admission_required)
        retriever = Retriever(conn, OpenSearchIndex(client, alias=alias, segmenter=seg), embedder, model)
        return Engine(conn, retriever, model, LocalAuthorizer(conn, accounts), store=store,
                      thresholds=row["thresholds"] if row else None, budgets=row["budgets"] if row else None,
                      deep_enabled=deep_enabled and g.deep, verified_enabled=g.verified, closed_reasons=g.reasons)
    return make


def query_runner(env: dict[str, str], store, *, owner: str | None = None):
    """后台任务进程用的查询执行者；EAER_DISABLE_QUERY=1 时为 None，缺配置报错。"""
    if env.get("EAER_DISABLE_QUERY") == "1":
        return None
    if not env.get("EAER_TOKENS_FILE"):
        raise ConfigError("后台执行提问需要授权配置 EAER_TOKENS_FILE（与接口进程相同）；不开放提问时设置 EAER_DISABLE_QUERY=1")
    from eaer.orchestrator.tasks import TaskRunner

    _, accounts = load_tokens(env["EAER_TOKENS_FILE"])
    return TaskRunner(env["EAER_DATABASE_URL"], engine_factory(env, accounts, store), authorizer_factory(accounts),
                      owner=owner)
