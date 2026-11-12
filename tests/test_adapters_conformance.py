"""AC-11-4：对接接口一致性套件（开发文档 11.3；硬性）。参考实现先通过同一套测试。

ModelClient 的参考实现（线上模型通用客户端）需要用户提供接入配置，尚未实现；这里用测试替身跑通套件，
并另跑一个“不支持结构化输出”的替身，验证套件对两种能力声明都适用。
"""
from __future__ import annotations

import json

import pytest

from eaer.adapters.authorizer import AuthorizationError, LocalAuthorizer
from eaer.adapters.conformance import (AuthorizerContract, DataSourceContract, LocalAccountsDriver,
                                       LocalFolderDriver, ModelClientContract)
from eaer.adapters.datasource import AclFormatError, LocalFolderSource, SourceNotFound, parse_acl
from eaer.adapters.model import DeterministicTestModel, ModelEmbedder
from eaer.domain.identity import Identity


class TestLocalFolderSource(DataSourceContract):
    @pytest.fixture
    def source_driver(self, tmp_path):
        return LocalFolderSource(tmp_path / "src", page_size=7), LocalFolderDriver(tmp_path / "src")


class TestLocalAuthorizer(AuthorizerContract):
    @pytest.fixture
    def authz(self, conn):
        accounts: dict[str, dict] = {}
        return LocalAuthorizer(conn, accounts), LocalAccountsDriver(accounts), conn


class TestDeterministicModel(ModelClientContract):
    @pytest.fixture
    def model(self):
        return DeterministicTestModel()


class TestModelWithoutStructuredOutput(ModelClientContract):
    @pytest.fixture
    def model(self):
        return DeterministicTestModel(structured_output=False)


# ---------- 参考实现自身的细节 ----------

@pytest.mark.parametrize("text, want", [
    ("principals: [R_STAFF, R_FIN]\n", ("R_FIN", "R_STAFF")),
    ("# 财务目录\nprincipals:\n  - R_FIN\n  - 'R_AUDIT'\n", ("R_AUDIT", "R_FIN")),
    ("principals: []\n", ()),
])
def test_parse_acl(text, want):
    assert parse_acl(text) == want


@pytest.mark.parametrize("text", ["", "roles: [R_A]", "principals: R_A", "principals: [R_A]\nowner: x"])
def test_parse_acl_rejects_other_shapes(text):
    with pytest.raises(AclFormatError):
        parse_acl(text)


def test_acl_inherits_from_parent_and_default_denies(tmp_path):
    root = tmp_path / "src"
    (root / "fin" / "q3").mkdir(parents=True)
    (root / "open").mkdir()
    (root / "fin" / "_acl.yaml").write_text("principals: [R_FIN]\n", encoding="utf-8")
    (root / "fin" / "q3" / "r.txt").write_text("报告", encoding="utf-8")       # 继承上级目录
    (root / "open" / "n.txt").write_text("没有权限声明", encoding="utf-8")      # 默认拒绝
    (root / "fin" / ".hidden").write_text("x", encoding="utf-8")
    src = LocalFolderSource(root)
    page = src.list_changes(None)
    assert [c.source_ref for c in page.changes] == ["fin/q3/r.txt"]
    assert src.read_acl("fin/q3/r.txt").principals == ("R_FIN",)
    with pytest.raises(SourceNotFound):
        src.fetch("open/n.txt")


def test_revoking_to_nobody_reports_delete(tmp_path):
    drv = LocalFolderDriver(tmp_path / "src")
    src = LocalFolderSource(tmp_path / "src")
    a = drv.put("a", b"x", ["R_A"])
    cur = src.list_changes(None).cursor
    drv.set_acl(a, [])
    assert [(c.kind, c.source_ref) for c in src.list_changes(cur).changes] == [("delete", a)]


def test_bad_cursor_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="游标无效"):
        LocalFolderSource(tmp_path).list_changes("not-a-cursor")


def test_local_authorizer_from_file_and_tenant_mismatch(conn, tmp_path):
    cfg = tmp_path / "accounts.json"
    cfg.write_text(json.dumps({"accounts": {"alice": {"tenant": "t1", "principals": ["R_STAFF"]}},
                               "ttl_seconds": 60}), encoding="utf-8")
    a = LocalAuthorizer.from_file(conn, cfg)
    assert a.resolve(Identity("alice", "t1", frozenset())).principals == frozenset({"R_STAFF"})
    with pytest.raises(AuthorizationError):
        a.resolve(Identity("alice", "t2", frozenset()))       # 账号属于别的租户


def test_model_embedder_wraps_model_client():
    emb = ModelEmbedder(DeterministicTestModel(dims=32), "test-double-32")
    assert emb.dims == 32 and len(emb.embed(["住宿"])[0]) == 32
