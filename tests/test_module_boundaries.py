"""AC-3-1：模块边界违规为 0（开发文档 3.2；硬性）。

对 eaer/ 与 factset/ 的源码做静态检查（CI 中随 pytest 运行），三条规则：
1. 检索索引：只有 retrieval 导入检索引擎客户端、发起检索（search_text / search_vector）；
   只有 ingestion 与 lifecycle 写索引（upsert / update_fields / delete_version / switch_alias / drop_all / drop_index）。
2. 版本状态：只有 lifecycle 执行改变 document_version 状态的 SQL（新增版本、改 status、删除版本）。
3. 模型：业务模块不导入模型厂商的 SDK，模型调用只能在 adapters（模型接口与参考实现）里。

不用 import-linter：第 2 条和索引读写要看 SQL 与方法调用，导入规则表达不了。
写索引按方法名识别，若别的类也用了这些方法名会误报，届时调整规则并写明原因。
迁移脚本（db/migrations）是表结构与触发器定义，不参与第 2 条检查。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

ENGINE_CLIENTS = {"opensearchpy", "opensearch", "elasticsearch"}
INDEX_READS = {"search_text", "search_vector"}
INDEX_WRITES = {"upsert", "update_fields", "delete_version", "switch_alias", "drop_all", "drop_index"}
VENDOR_SDKS = {
    "openai", "anthropic", "google.generativeai", "google.genai", "vertexai", "dashscope", "zhipuai", "zai",
    "qianfan", "volcenginesdkarkruntime", "sparkai", "tencentcloud.hunyuan", "cohere", "mistralai", "voyageai",
    "ollama", "litellm", "sentence_transformers", "transformers", "langchain_openai", "langchain_anthropic",
    "langchain_community", "llama_index",
}
WRITERS = {"ingestion", "lifecycle"}
_UPDATE = re.compile(r"\bUPDATE\s+document_version\s+SET\s+(?P<set>.*?)(?:\bWHERE\b|\bRETURNING\b|$)", re.I | re.S)
_CREATE_OR_DELETE = re.compile(r"\b(?:INSERT\s+INTO|DELETE\s+FROM)\s+document_version\b", re.I)


def module_of(rel: str) -> str:
    """eaer/<模块>/… → 模块名；factset、evalrun、evalload 归入 eval；eaer 根目录下的文件归入 core。"""
    parts = Path(rel).parts
    if parts[0] == "factset" or (len(parts) > 2 and parts[1] in ("evalrun", "evalload")):
        return "eval"
    return parts[1] if len(parts) > 2 else "core"


def _matches(name: str, prefixes: set[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _imported(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        return [node.module] + [f"{node.module}.{a.name}" for a in node.names]
    if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and ((isinstance(node.func, ast.Name) and node.func.id in ("__import__", "import_module"))
                 or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"))):
        return [node.args[0].value]
    return []


def _strings(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):   # f-string：只看字面部分
        return ["".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))]
    return []


def violations(rel: str, source: str) -> list[str]:
    mod = module_of(rel)
    out: list[str] = []
    for node in ast.walk(ast.parse(source, filename=rel)):
        where = f"{rel}:{getattr(node, 'lineno', '?')}"
        for name in _imported(node):
            if _matches(name, ENGINE_CLIENTS) and mod != "retrieval":
                out.append(f"{where} 规则 1：{mod} 导入了检索引擎客户端 {name}")
            if _matches(name, VENDOR_SDKS) and mod != "adapters":
                out.append(f"{where} 规则 3：{mod} 导入了模型厂商 SDK {name}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in INDEX_READS and mod != "retrieval":
                out.append(f"{where} 规则 1：{mod} 直接检索索引（{node.func.attr}）")
            if node.func.attr in INDEX_WRITES and mod not in WRITERS:
                out.append(f"{where} 规则 1：{mod} 写检索索引（{node.func.attr}）")
        if mod != "lifecycle" and "migrations" not in Path(rel).parts:
            for s in _strings(node):
                m = _UPDATE.search(s)
                if (m and re.search(r"\bstatus\s*=", m.group("set"), re.I)) or _CREATE_OR_DELETE.search(s):
                    out.append(f"{where} 规则 2：{mod} 改变版本状态：{' '.join(s.split())[:80]}")
    return out


def sources() -> list[Path]:
    return sorted(p for d in ("eaer", "factset") for p in (ROOT / d).rglob("*.py"))


def test_no_boundary_violations():
    found = [v for p in sources() for v in violations(str(p.relative_to(ROOT)), p.read_text(encoding="utf-8"))]
    assert found == [], "模块边界违规：\n" + "\n".join(found)


def test_scans_every_module():
    mods = {module_of(str(p.relative_to(ROOT))) for p in sources()}
    assert {"api", "ingestion", "lifecycle", "retrieval", "review", "eval", "core"} <= mods


@pytest.mark.parametrize("rel, src, rule", [
    ("eaer/api/x.py", "import opensearchpy", "规则 1"),
    ("eaer/evalrun/x.py", "from opensearchpy import OpenSearch", "规则 1"),
    ("eaer/review/x.py", "from opensearchpy.helpers import bulk", "规则 1"),
    ("eaer/ingestion/x.py", "self.index.search_text(q, f, 10)", "规则 1"),
    ("eaer/orchestrator/x.py", "index.search_vector(v, f, 10)", "规则 1"),
    ("eaer/retrieval/x.py", "self.index.upsert(docs)", "规则 1"),
    ("eaer/evalrun/x.py", "ix.switch_alias(name)", "规则 1"),
    ("eaer/api/x.py", "index.delete_version(v)", "规则 1"),
    ("eaer/review/x.py", 'conn.execute("UPDATE document_version SET status = %s WHERE id = %s", (s, i))', "规则 2"),
    ("eaer/ingestion/x.py", 'c.execute("UPDATE document_version "\n  "SET archived_at = now(), status = \'archived\'")', "规则 2"),
    ("eaer/api/x.py", 'c.execute(f"DELETE FROM document_version WHERE id = {i}")', "规则 2"),
    ("eaer/ingestion/x.py", 'c.execute("INSERT INTO document_version (id, status) VALUES (%s, %s)")', "规则 2"),
    ("eaer/orchestrator/x.py", "from openai import OpenAI", "规则 3"),
    ("eaer/verifier/x.py", "import anthropic", "规则 3"),
    ("eaer/retrieval/x.py", "import google.generativeai as genai", "规则 3"),
    ("eaer/ingestion/x.py", "import importlib\nimportlib.import_module('dashscope')", "规则 3"),
    ("factset/x.py", "__import__('openai')", "规则 3"),
])
def test_checker_catches_violations(rel, src, rule):
    found = violations(rel, src)
    assert found and all(rule in v for v in found), found


@pytest.mark.parametrize("rel, src", [
    ("eaer/retrieval/x.py", "from opensearchpy import OpenSearch\nself.index.search_text(q, f, 1)"),
    ("eaer/ingestion/x.py", "self.index.upsert(docs); self.index.delete_version(v)"),
    ("eaer/lifecycle/x.py", 'c.execute("UPDATE document_version SET status = %s WHERE id = %s")'),
    ("eaer/ingestion/x.py", 'c.execute("UPDATE document_version SET parser_version = %s WHERE id = %s AND status = 1")'),
    ("eaer/adapters/x.py", "import openai\nfrom anthropic import Anthropic"),
    ("eaer/db/migrations/versions/0001.py", 'op.execute("UPDATE document_version SET status = \'x\'")'),
    ("eaer/api/x.py", "import openpyxl, opensearch_lookalike"),
])
def test_checker_allows_legitimate_code(rel, src):
    assert violations(rel, src) == []
