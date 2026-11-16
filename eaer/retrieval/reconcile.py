"""索引对账（开发文档 5.4、AC-5-2）：数据库中可搜索的版本与索引中的版本应完全一致，每个版本的证据也应一条不差。"""
from __future__ import annotations

import psycopg

from eaer.retrieval.index import SearchIndex


def reconcile(conn: psycopg.Connection, index: SearchIndex, tenant_id: str) -> dict[str, set[str]]:
    """对账（AC-5-2），返回三类差异的版本 id：

    - missing_in_index：数据库里可搜索（且有证据），索引里没有这个版本；
    - unexpected_in_index：索引里有，数据库里不可搜索或没有这个版本；
    - incomplete_in_index：两边都有，索引里的证据条数与数据库不同。通常是少了一部分：深度解析（6.2 第 6 阶段）在
      发布之后追加表格与图表证据并排 index_update，这个任务用完尝试次数时版本留在索引里、只有正文证据。证据只追加
      （同一内容沿用原 id），索引里多出来的只会来自别处（索引来自别的库等），由 index_update 整版重写。

    按条数比较，不逐条比较 id：条数相同而 id 不同（证据被替换）查不出来，但证据只追加，不会出现这种情况；
    逐条比较要把整个租户的证据 id 从索引里取出来，代价与语料规模成正比。
    """
    db = {str(r["id"]): r["n"] for r in conn.execute(
        "SELECT v.id, count(*) AS n FROM document_version v JOIN document d ON d.id = v.document_id "
        "JOIN evidence e ON e.version_id = v.id "
        "WHERE d.tenant_id = %s AND v.status = 'searchable' GROUP BY v.id", (tenant_id,))}
    ix = index.evidence_counts(tenant_id)
    return {"missing_in_index": db.keys() - ix.keys(), "unexpected_in_index": ix.keys() - db.keys(),
            "incomplete_in_index": {v for v in db.keys() & ix.keys() if db[v] != ix[v]}}
