"""索引对账（开发文档 5.4、AC-5-2）：数据库中可搜索的版本集合与索引中的版本集合应完全一致。"""
from __future__ import annotations

import psycopg

from eaer.retrieval.index import SearchIndex


def reconcile(conn: psycopg.Connection, index: SearchIndex, tenant_id: str) -> dict[str, set[str]]:
    """对账（AC-5-2）：数据库中可搜索的版本与索引中的版本应完全一致。"""
    db = {str(r["id"]) for r in conn.execute(
        "SELECT v.id FROM document_version v JOIN document d ON d.id = v.document_id "
        "WHERE d.tenant_id = %s AND v.status = 'searchable' "
        "AND EXISTS (SELECT 1 FROM evidence e WHERE e.version_id = v.id)", (tenant_id,))}
    ix = index.version_ids(tenant_id)
    return {"missing_in_index": db - ix, "unexpected_in_index": ix - db}
