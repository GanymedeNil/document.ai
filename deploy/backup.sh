#!/bin/sh
# 精简版备份（开发文档 13.4）：PostgreSQL 全量备份（pg_dump 自定义格式）、文件存储打包、清单与校验和。
# 检索索引不备份：恢复数据库与原件后由 ensure-index 重建（deploy/restore.sh）。建议每天定时运行一次。
#   deploy/backup.sh [备份目录]            默认 backups/<时间>；EAER_PROJECT 指定编排项目名（默认 eaer-lite）
set -eu
cd "$(dirname "$0")/.."
P=${EAER_PROJECT:-eaer-lite}
OUT=${1:-backups/$(date +%Y%m%d-%H%M%S)}
C="docker compose -p $P --env-file deploy/eaer.env"
mkdir -p "$OUT"
$C exec -T postgres pg_dump -U eaer -d eaer -Fc > "$OUT/eaer.dump"
$C exec -T worker tar czf - -C /data/files . > "$OUT/files.tgz"
{
  echo "project=$P"
  echo "created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "schema=$($C exec -T postgres psql -U eaer -d eaer -Atc 'SELECT version_num FROM alembic_version')"
  echo "searchable_versions=$($C exec -T postgres psql -U eaer -d eaer -Atc "SELECT count(*) FROM document_version WHERE status = 'searchable'")"
} > "$OUT/manifest.txt"
(cd "$OUT" && shasum -a 256 eaer.dump files.tgz > SHA256SUMS)
echo "备份完成：$OUT"
cat "$OUT/manifest.txt"
