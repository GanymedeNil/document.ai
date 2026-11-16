#!/bin/sh
# 从备份恢复到一个新环境（开发文档 13.4、AC-13-4）：新的编排项目、新的数据卷，原环境不受影响。
#   EAER_PROJECT=eaer-restore EAER_API_PORT=8001 deploy/restore.sh backups/20261114-120000
# 步骤：起 PostgreSQL 与 OpenSearch → 校验备份 → 恢复数据库（pg_restore）→ 恢复文件存储 → 启动其余服务
# （迁移已是最新则不动；新环境没有索引，init-index 从数据库与证据向量缓存重建）。之后跑冒烟测试与回溯快照比对。
set -eu
cd "$(dirname "$0")/.."
SRC=${1:?用法：deploy/restore.sh 备份目录}
P=${EAER_PROJECT:-eaer-restore}
C="docker compose -p $P --env-file deploy/eaer.env"
(cd "$SRC" && shasum -a 256 -c SHA256SUMS)
$C up -d postgres opensearch
until [ "$(docker inspect -f '{{.State.Health.Status}}' "$P-postgres-1")" = healthy ]; do sleep 2; done
$C exec -T postgres pg_restore -U eaer -d eaer --no-owner --exit-on-error < "$SRC/eaer.dump"
$C run --rm --no-deps -T worker sh -c 'tar xzf - -C /data/files' < "$SRC/files.tgz"
$C up -d
until [ "$(docker inspect -f '{{.State.Health.Status}}' "$P-api-1" 2>/dev/null)" = healthy ]; do sleep 2; done
echo "恢复完成：项目 $P，接口端口 ${EAER_API_PORT:-8000}"
$C exec -T postgres psql -U eaer -d eaer -Atc "SELECT 'searchable_versions=' || count(*) FROM document_version WHERE status = 'searchable'"
