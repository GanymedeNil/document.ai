"""可观测（开发文档 13.3、10.2、AC-13-5）：每次查询一条 trace，覆盖状态机的每个节点。

trace 的结构：
- eaer.query：一次提问的根（接口进程收到提问时开始；评测与测试经 Engine.ask）；
- eaer.query.execute：执行图的一段。转入后台、或执行者中断后由别的进程接着执行时，从查询状态里保存的 traceparent
  续上同一条 trace（W3C Trace Context），所以跨进程仍是一条；
- eaer.node.<节点>：prepare（时间点识别与 Router）、first、escalate、verify、assemble，记节点名、耗时、
  本节点结束时的检索候选数、模型调用次数与 token 数、模式路径；
- eaer.retrieve：一次检索各阶段的候选数（关键词、向量、融合、数据库核对后、返回）；
- eaer.model.<generate|embed|rerank>：模型调用的模型名、token 数、尝试次数、是否成功；
- eaer.ingest.<阶段>：后台任务（解析、写入索引等）的版本、尝试次数、结果。

原文不进 trace（10.2“可观测链路”）：span 的属性只接受数字、布尔值和形如标识符的 ASCII 字符串（模型名、阶段名、
UUID、证据 id），问题、证据、回答等原文一律进不来（span() 会丢弃并计数）。原文只在数据库里（query_log、会话检查点），
经接口按同一套权限访问：会话只给提问人，证据逐条授权；检查点按 trace_retention_days 清理。所以 trace 后端
（精简版可选安装，集群版接 Phoenix）不需要再做文档级的权限控制。异常只记类型名，不记异常信息（信息里可能带原文）。

配置（部署环境变量）：
  EAER_TRACE_EXPORTER   none（默认，不导出）/ console（打印到标准输出）/ otlp（OTLP/HTTP，需安装 requirements-otlp.txt，
                        接收端地址用标准的 OTEL_EXPORTER_OTLP_ENDPOINT 等变量）
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

_SAFE = re.compile(r"[A-Za-z0-9_.:/@+\-]{0,128}", re.ASCII)
_PROP = TraceContextTextMapPropagator()
REDACTED = "eaer.redacted_attributes"


class TraceConfigError(Exception):
    pass


def configure(env: dict[str, str], service: str, *, exporter=None) -> bool:
    """按 EAER_TRACE_EXPORTER 设置全局 TracerProvider；返回是否启用。exporter 参数供测试注入（如 InMemorySpanExporter）。"""
    kind = env.get("EAER_TRACE_EXPORTER", "none")
    if exporter is None and kind == "none":
        return False
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    if exporter is not None:
        processor = SimpleSpanProcessor(exporter)
    elif kind == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        processor = BatchSpanProcessor(ConsoleSpanExporter())
    elif kind == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            raise TraceConfigError("EAER_TRACE_EXPORTER=otlp 需要先安装 requirements-otlp.txt")
        processor = BatchSpanProcessor(OTLPSpanExporter())
    else:
        raise TraceConfigError(f"EAER_TRACE_EXPORTER 只能是 none、console、otlp：{kind}")
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return True


def tracer():
    return trace.get_tracer("eaer")


def safe(value: Any) -> Any:
    """可以放进 span 属性的值；原文（含非 ASCII、空白、超长的字符串）返回 None。"""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value if _SAFE.fullmatch(value) else None
    if isinstance(value, (list, tuple)):
        items = [safe(v) for v in value]
        if all(isinstance(v, str) for v in items) or all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                         for v in items):
            return items
        return None
    return None


def set_attrs(sp, attrs: dict[str, Any]) -> None:
    dropped = 0
    for k, v in attrs.items():
        if v is None:
            continue
        s = safe(v)
        if s is None:
            dropped += 1
            continue
        sp.set_attribute(k, s)
    if dropped:
        sp.set_attribute(REDACTED, dropped)


def traceparent() -> str | None:
    """当前 span 的 W3C traceparent；没有在记录的 span 时为 None。存进查询状态，供别的进程续上同一条 trace。"""
    carrier: dict[str, str] = {}
    _PROP.inject(carrier)
    return carrier.get("traceparent")


@contextmanager
def span(name: str, *, parent: str | None = None, **attrs: Any) -> Iterator[Any]:
    """开一个 span；parent 为 traceparent 字符串时以它为父（跨进程续接）。属性经 safe() 过滤。"""
    token = None
    if parent:
        token = otel_context.attach(_PROP.extract({"traceparent": parent}))
    try:
        with tracer().start_as_current_span(name, record_exception=False, set_status_on_exception=False) as sp:
            set_attrs(sp, attrs)
            try:
                yield sp
            except BaseException as e:
                sp.set_status(Status(StatusCode.ERROR, type(e).__name__))
                sp.set_attribute("error.type", type(e).__name__)
                raise
    finally:
        if token is not None:
            otel_context.detach(token)


def call_totals(calls) -> dict[str, int]:
    """一组模型调用记录（eaer.adapters.online.CallRecord）的次数与 token 合计。"""
    calls = list(calls or [])
    return {"eaer.model_calls": len(calls),
            "gen_ai.usage.input_tokens": sum(int(getattr(c, "input_tokens", 0) or 0) for c in calls),
            "gen_ai.usage.output_tokens": sum(int(getattr(c, "output_tokens", 0) or 0) for c in calls)}
