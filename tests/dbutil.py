"""数据库测试的公用工具：可控时钟和一个最小的样例数据集。"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from eaer.lifecycle.service import Lifecycle


class Clock:
    def __init__(self, start: datetime | None = None):
        self.now = start or datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.now += timedelta(microseconds=1)
        return self.now

    def set(self, d: date) -> None:
        self.now = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def seed(conn, clock: Clock | None = None):
    """租户 t1，文档 D（v1 已发布、v2 待审核，v2 带证据、待确认值、关系和任务）。"""
    clock = clock or Clock()
    lc = Lifecycle(conn, clock)
    clock.set(date(2025, 1, 1))
    lc.ensure_tenant("t1")
    acl = lc.create_acl_policy("t1", ["R_STAFF"])
    doc = lc.create_document("t1", "D", "差旅报销管理办法", "normative")
    v1 = lc.add_version(doc, content_hash="sha256:v1", raw_object_uri="file://v1", acl_policy_id=acl,
                        parser_version="test")
    lc.set_draft_validity(v1, valid_from=date(2025, 1, 1), valid_from_grade="explicit")
    lc.add_evidence(v1, "table", {"page": 1, "row": 1}, "500元/晚")
    clock.set(date(2025, 1, 3))
    lc.publish(v1, "alice")
    clock.set(date(2025, 12, 15))
    v2 = lc.add_version(doc, content_hash="sha256:v2", raw_object_uri="file://v2", acl_policy_id=acl,
                        parser_version="test")
    lc.set_draft_validity(v2, valid_from=date(2026, 1, 1), valid_from_grade="explicit")
    lc.add_draft_relation("supersedes", v2, v1)
    lc.add_evidence(v2, "table", {"page": 1, "row": 1}, "600元/晚")
    conn.execute("INSERT INTO ingest_job (id, version_id, stage, status, created_at, updated_at) "
                 "VALUES (gen_random_uuid(), %s, 'parse_text', 'done', now(), now())", (v2,))
    return lc, clock, {"acl": acl, "doc": doc, "v1": v1, "v2": v2}
