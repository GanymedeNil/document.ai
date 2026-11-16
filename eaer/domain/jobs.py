"""后台任务（ingest_job）的优先级：数字小的先领取（Worker.claim 按 priority, created_at 排序）。

- index_update（归档、取消归档、权限变更、有效期更正、深度解析追加证据）：不调用 Embedding（取消归档的向量取自缓存），
  很快；它缩短索引与数据库不一致的窗口——这段时间内结果的正确由检索后的数据库核对保证（5.2），但撤权、归档之后
  索引里多出的候选会白白占掉检索名额，所以最先做；
- publish_index（审核通过后写入）：让新发布的文档可以检索，是索引更新延迟（5.3、AC-5-6）的主体；
- parse_text、classify_extract：审核之前的解析，影响的是审核人等待的时间，不影响提问；
- parse_deep：表格与图表的深度解析，开发文档 6.4 规定它不阻塞发布，排在最后。
严格按优先级领取：持续过载时低优先级的任务会一直等，这是有意的（先保证已发布内容与权限的时效）。
"""
from __future__ import annotations

PRIORITY = {"index_update": 10, "publish_index": 20, "receive": 50, "parse_text": 50, "classify_extract": 50,
            "parse_deep": 100, "reindex": 100}
DEFAULT_PRIORITY = 100
