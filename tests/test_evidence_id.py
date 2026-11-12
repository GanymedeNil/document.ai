"""AC-4-3：证据 id 稳定。"""
from __future__ import annotations

import re

from eaer.domain.evidence import content_hash, evidence_id

VERSION = "8d6f2c1e-0000-4000-8000-000000000001"


def ids(blocks):
    return [evidence_id(VERSION, m, loc, content_hash(text)) for m, loc, text in blocks]


BLOCKS = [
    ("text", {"page": 1, "section": "第三条"}, "部门经理单笔采购审批上限为50,000元。"),
    ("table", {"page": 2, "table_id": "t1", "row": 1, "column": 1}, "650元/晚"),
    ("table", {"page": 2, "table_id": "t1", "row": 2, "column": 1}, "450元/晚"),
    ("figure", {"page": 3, "bbox": [10, 20, 300, 200]}, "图1 2026 年第二季度差旅总费用"),
]


def test_format():
    assert all(re.fullmatch(r"ev_[0-9a-f]{16}", i) for i in ids(BLOCKS))


def test_reparse_gives_same_ids():
    assert ids(BLOCKS) == ids([(m, dict(reversed(list(loc.items()))), t) for m, loc, t in BLOCKS])


def test_whitespace_and_fullwidth_do_not_change_ids():
    assert content_hash("650元/晚") == content_hash(" 650元/晚 \n") == content_hash("６５０元/晚")
    assert ids(BLOCKS) == ids([(m, loc, f"  {t}\n") for m, loc, t in BLOCKS])


def test_only_changed_block_gets_new_id():
    changed = list(BLOCKS)
    changed[1] = ("table", BLOCKS[1][1], "680元/晚")
    before, after = ids(BLOCKS), ids(changed)
    assert [a == b for a, b in zip(before, after)] == [True, False, True, True]


def test_ids_differ_across_versions():
    other = "8d6f2c1e-0000-4000-8000-000000000002"
    m, loc, t = BLOCKS[0]
    assert evidence_id(VERSION, m, loc, content_hash(t)) != evidence_id(other, m, loc, content_hash(t))
