"""对接接口一致性套件（开发文档 11.3、AC-11-4；硬性）。

任何 DataSource、Authorizer、ModelClient 实现都必须通过同一套测试：参考实现先通过（tests/test_adapters_conformance.py），
客户实现交付前必跑。用法：在客户项目的测试目录里继承对应的 *Contract 类，提供它要求的夹具，然后运行 pytest：

    from eaer.adapters.conformance import DataSourceContract

    class TestSharePointSource(DataSourceContract):
        @pytest.fixture
        def source_driver(self):
            return SharePointSource(...), SharePointTestDriver(...)   # 指向专用的测试站点，初始为空

数据源与授权的测试需要“驱动”：在被测系统里放文档、改权限、准备测试账号。参考驱动见本模块末尾。
"""
from __future__ import annotations

import math
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pytest

from eaer.adapters.authorizer import AuthorizationError, ScopeExpired
from eaer.adapters.datasource import ACL_FILE, Change, SourceNotFound
from eaer.adapters.model import Capabilities, CapabilityNotSupported, Completion, Message
from eaer.domain.identity import Identity, Scope

# ---------------------------------------------------------------- 数据源


class SourceDriver(Protocol):
    def put(self, name: str, data: bytes, principals: list[str]) -> str:
        """新增或覆盖一份文档并设定其权限，返回数据源会报告的 source_ref。"""
    def remove(self, source_ref: str) -> None: ...
    def set_acl(self, source_ref: str, principals: list[str]) -> None: ...


def drain(source, cursor: str | None, limit: int = 10_000) -> tuple[list[Change], str]:
    """翻完所有页，返回全部变化与最终游标。"""
    out: list[Change] = []
    for _ in range(limit):
        page = source.list_changes(cursor)
        assert isinstance(page.cursor, str) and page.cursor, "游标必须是非空字符串"
        out += page.changes
        cursor = page.cursor
        if not page.has_more:
            return out, cursor
    raise AssertionError("翻页不结束：游标没有前进")


class DataSourceContract:
    """子类提供夹具 source_driver，返回 (DataSource, SourceDriver)，数据源初始为空。"""

    def test_declares_revocation_delay(self, source_driver):
        src, _ = source_driver
        assert isinstance(src.max_revocation_delay_seconds, int) and src.max_revocation_delay_seconds >= 0

    def test_initial_listing_fetch_and_acl(self, source_driver):
        src, drv = source_driver
        a = drv.put("a", "甲文档".encode(), ["R_A"])
        b = drv.put("b", "乙文档".encode(), ["R_A", "R_B"])
        changes, _ = drain(src, None)
        assert {(c.kind, c.source_ref) for c in changes} == {("upsert", a), ("upsert", b)}
        doc = src.fetch(a)
        assert doc.source_ref == a and doc.data == "甲文档".encode() and doc.filename
        assert set(src.read_acl(b).principals) == {"R_A", "R_B"}

    def test_cursor_is_incremental(self, source_driver):
        src, drv = source_driver
        drv.put("a", b"x", ["R_A"])
        _, cur = drain(src, None)
        assert drain(src, cur)[0] == []

    def test_content_update_is_upsert(self, source_driver):
        src, drv = source_driver
        a = drv.put("a", b"v1", ["R_A"])
        _, cur = drain(src, None)
        drv.put("a", b"v2 longer", ["R_A"])
        changes, _ = drain(src, cur)
        assert [(c.kind, c.source_ref) for c in changes] == [("upsert", a)]
        assert src.fetch(a).data == b"v2 longer"

    def test_acl_change_is_reported(self, source_driver):
        src, drv = source_driver
        a = drv.put("a", b"x", ["R_A"])
        _, cur = drain(src, None)
        drv.set_acl(a, ["R_B"])
        changes, _ = drain(src, cur)
        assert [(c.kind, c.source_ref) for c in changes] == [("acl", a)]
        assert set(src.read_acl(a).principals) == {"R_B"}

    def test_delete_is_reported(self, source_driver):
        src, drv = source_driver
        a, b = drv.put("a", b"x", ["R_A"]), drv.put("b", b"y", ["R_A"])
        _, cur = drain(src, None)
        drv.remove(b)
        changes, _ = drain(src, cur)
        assert [(c.kind, c.source_ref) for c in changes] == [("delete", b)]
        with pytest.raises(SourceNotFound):
            src.fetch(b)
        assert src.fetch(a).data == b"x"

    def test_unknown_reference_is_not_found(self, source_driver):
        src, _ = source_driver
        for ref in ("does-not-exist", "../etc/passwd"):
            with pytest.raises(SourceNotFound):
                src.fetch(ref)
            with pytest.raises(SourceNotFound):
                src.read_acl(ref)

    def test_paging_reports_each_change_once(self, source_driver):
        src, drv = source_driver
        refs = {drv.put(f"d{i:02d}", f"文档{i}".encode(), ["R_A"]) for i in range(25)}
        changes, cur = drain(src, None)
        assert sorted(c.source_ref for c in changes) == sorted(refs)
        assert drain(src, cur)[0] == []

    def test_replaying_an_old_cursor_is_safe(self, source_driver):
        """处理失败时用旧游标重来，得到同样的变化，不丢失。"""
        src, drv = source_driver
        drv.put("a", b"x", ["R_A"])
        _, cur = drain(src, None)
        b = drv.put("b", b"y", ["R_A"])
        first, _ = drain(src, cur)
        again, _ = drain(src, cur)
        assert [(c.kind, c.source_ref) for c in first] == [(c.kind, c.source_ref) for c in again] == [("upsert", b)]


# ---------------------------------------------------------------- 授权


class AuthorizerDriver(Protocol):
    tenant_id: str
    def identity(self, principals: set[str]) -> Identity:
        """一个测试账号的身份，它在被测授权系统里恰好拥有这些文档权限角色。"""
    def unknown_identity(self) -> Identity: ...


class AuthorizerContract:
    """子类提供夹具 authz，返回 (Authorizer, AuthorizerDriver, conn)；conn 是该实现核对证据所用的数据库（已迁移、可写）。"""

    @pytest.fixture
    def corpus(self, authz):
        from eaer.lifecycle.service import Lifecycle

        _, drv, conn = authz
        lc = Lifecycle(conn)
        ev = {}

        def make(key, tenant, principals, state):
            lc.ensure_tenant(tenant)
            acl = lc.create_acl_policy(tenant, principals)
            doc = lc.create_document(tenant, f"conf-{key}-{uuid.uuid4().hex[:6]}", key, "normative")
            vid = lc.add_version(doc, content_hash=f"sha256:{uuid.uuid4().hex}", raw_object_uri=f"file://{key}",
                                 acl_policy_id=acl, parser_version="conformance")
            ev[key] = lc.add_evidence(vid, "text", {"para": 1}, f"{key} 的证据")
            if state != "pending_review":
                lc.publish(vid, "conformance")
            if state == "archived":
                lc.archive_document(doc, "conformance")
            return doc

        docs = {
            "allowed": make("allowed", drv.tenant_id, ["R_A"], "searchable"),
            "other_role": make("other_role", drv.tenant_id, ["R_B"], "searchable"),
            "pending": make("pending", drv.tenant_id, ["R_A"], "pending_review"),
            "archived": make("archived", drv.tenant_id, ["R_A"], "archived"),
            "other_tenant": make("other_tenant", drv.tenant_id + "-other", ["R_A"], "searchable"),
        }
        return ev, docs, lc

    def test_resolve_known_identity(self, authz):
        a, drv, _ = authz
        s = a.resolve(drv.identity({"R_A"}))
        assert isinstance(s, Scope) and s.tenant_id == drv.tenant_id and s.principals == frozenset({"R_A"})
        assert s.expires_at is not None and s.expires_at.tzinfo is not None
        assert s.expires_at > datetime.now(timezone.utc)

    def test_unknown_identity_is_refused(self, authz):
        a, drv, _ = authz
        with pytest.raises(AuthorizationError):
            a.resolve(drv.unknown_identity())

    def test_check_matrix(self, authz, corpus):
        a, drv, _ = authz
        ev, _, _ = corpus
        scope = a.resolve(drv.identity({"R_A"}))
        keys = ["allowed", "other_role", "pending", "archived", "other_tenant"]
        got = a.check(scope, [ev[k] for k in keys] + ["ev_" + "0" * 64])
        assert got == [True, False, False, False, False, False]

    def test_check_keeps_order_and_length(self, authz, corpus):
        a, drv, _ = authz
        ev, _, _ = corpus
        scope = a.resolve(drv.identity({"R_A"}))
        assert a.check(scope, []) == []
        assert a.check(scope, [ev["other_role"], ev["allowed"], ev["allowed"]]) == [False, True, True]

    def test_expired_scope_is_refused(self, authz, corpus):
        a, drv, _ = authz
        ev, _, _ = corpus
        stale = Scope(drv.tenant_id, frozenset({"R_A"}), datetime.now(timezone.utc) - timedelta(seconds=1))
        with pytest.raises(ScopeExpired):
            a.check(stale, [ev["allowed"]])

    def test_revocation_is_seen_by_check(self, authz, corpus):
        a, drv, _ = authz
        ev, docs, lc = corpus
        scope = a.resolve(drv.identity({"R_A"}))
        assert a.check(scope, [ev["allowed"]]) == [True]
        lc.change_acl(docs["allowed"], ["R_B"], "conformance")
        assert a.check(scope, [ev["allowed"]]) == [False]


# ---------------------------------------------------------------- 模型


class ModelClientContract:
    """子类提供夹具 model，返回 ModelClient。线上模型会产生少量调用费用。"""

    def test_capabilities_are_well_formed(self, model):
        c = model.capabilities()
        assert isinstance(c, Capabilities)
        assert isinstance(c.structured_output, bool) and isinstance(c.vision, bool)
        assert isinstance(c.context_tokens, int) and c.context_tokens > 0
        assert isinstance(c.embedding_dims, int) and c.embedding_dims > 0

    def test_embed_shapes_and_stability(self, model):
        dims = model.capabilities().embedding_dims
        assert model.embed([]) == []
        vecs = model.embed(["一线城市住宿上限", "Hotel cap in tier-1 cities", "一线城市住宿上限"])
        assert len(vecs) == 3 and all(len(v) == dims for v in vecs)
        assert all(isinstance(x, float) and math.isfinite(x) for v in vecs for x in v)
        dot = sum(a * b for a, b in zip(vecs[0], vecs[2]))
        norm = math.sqrt(sum(a * a for a in vecs[0]) * sum(b * b for b in vecs[2]))
        assert norm > 0 and dot / norm > 0.999            # 同一文本两次结果一致

    def test_rerank_scores_every_passage(self, model):
        passages = ["出差伙食补贴每天80元", "一线城市住宿上限为每晚650元", "部门经理单笔采购审批上限10万元"]
        scores = model.rerank("一线城市住宿上限是多少", passages)
        assert len(scores) == 3 and all(isinstance(s, float) and math.isfinite(s) for s in scores)
        assert max(range(3), key=scores.__getitem__) == 1
        assert model.rerank("任意问题", []) == []

    def test_generate_returns_text_and_usage(self, model):
        out = model.generate([Message("system", "用一句话回答。"), Message("user", "一加一等于几？")], None)
        assert isinstance(out, Completion) and out.text.strip() and out.parsed is None and out.model
        assert out.usage.input_tokens >= 0 and out.usage.output_tokens >= 0

    def test_structured_output_matches_declared_capability(self, model):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}, "confident": {"type": "boolean"}},
                  "required": ["answer", "confident"]}
        msgs = [Message("user", "回答“一加一等于几”，并说明是否确定。")]
        if model.capabilities().structured_output:
            out = model.generate(msgs, schema)
            assert isinstance(out.parsed, dict) and set(schema["required"]) <= set(out.parsed)
            assert isinstance(out.parsed["answer"], str) and isinstance(out.parsed["confident"], bool)
        else:
            with pytest.raises(CapabilityNotSupported):
                model.generate(msgs, schema)


# ---------------------------------------------------------------- 参考驱动


class LocalFolderDriver:
    """LocalFolderSource 的测试驱动：每份文档放在自己的目录里，目录下的 _acl.yaml 就是它的权限。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, name: str, data: bytes, principals: list[str]) -> str:
        d = self.root / name
        d.mkdir(exist_ok=True)
        self._write_acl(d, principals)
        f = d / f"{name}.txt"
        before = f.stat().st_mtime_ns if f.exists() else None
        f.write_bytes(data)
        if before is not None and f.stat().st_mtime_ns <= before:   # 文件系统时间精度不够时人为推后
            os.utime(f, ns=(before + 1_000_000, before + 1_000_000))
        return f"{name}/{name}.txt"

    def remove(self, source_ref: str) -> None:
        (self.root / source_ref).unlink()

    def set_acl(self, source_ref: str, principals: list[str]) -> None:
        self._write_acl((self.root / source_ref).parent, principals)

    @staticmethod
    def _write_acl(d: Path, principals: list[str]) -> None:
        (d / ACL_FILE).write_text(f"principals: [{', '.join(principals)}]\n", encoding="utf-8")


class LocalAccountsDriver:
    """LocalAuthorizer 的测试驱动：按需生成拥有指定角色的本地账号。"""

    def __init__(self, accounts: dict[str, dict], tenant_id: str = "conf-t1"):
        self.accounts, self.tenant_id = accounts, tenant_id

    def identity(self, principals: set[str]) -> Identity:
        user = "u-" + "-".join(sorted(principals))
        self.accounts[user] = {"tenant": self.tenant_id, "principals": sorted(principals)}
        return Identity(user, self.tenant_id, frozenset())

    def unknown_identity(self) -> Identity:
        return Identity("nobody-" + uuid.uuid4().hex[:6], self.tenant_id, frozenset())
