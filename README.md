# EAER

企业级融合 RAG 系统。开发与验收依据是《EAER 企业级融合 RAG 系统开发文档 v1.0》，放在 `docs/` 目录，导出方式见 `docs/README.md`。

用 Claude Code 继续开发时，它会先读取仓库根目录的 `CLAUDE.md`：项目背景、开发约定、当前进度和下一步都在里面。

## 当前进度

| 部分 | 内容 | 验收项 |
| --- | --- | --- |
| 数据模型 | `eaer/db/migrations`：12 张表（含向量缓存）；不变量由触发器、检查约束和部分唯一索引保证 | AC-4-1、AC-4-2 |
| 证据 id | `eaer/domain/evidence.py`：由版本、模态、定位和内容哈希计算 | AC-4-3 |
| 生命周期 | `eaer/lifecycle/service.py`：新增版本、待确认值、发布、作废日推导、更正、归档、取消归档、权限变更、删除待审核版本 | AC-4-4（数据库部分）、AC-6-3 的一部分、AC-7-4 |
| 回溯查询 | `eaer/lifecycle/asof.py`：按系统时间取当时记录的有效期、关系和可见版本 | AC-4-5 |
| 文件存储 | `eaer/storage.py`：存储接口与本地实现，集群版换 S3 兼容存储 | — |
| 入库第 1–3、6 阶段 | `eaer/ingestion/`：接收与去重、文本解析（原生 Word/HTML，Docling 负责 PDF、扫描件、PPTX、XLSX、图片）、分类与时间抽取、表格深度解析、索引任务；后台任务可重试、可续跑 | AC-6-2、AC-6-8、AC-6-9（合成语料）、AC-7-5、AC-7-7（合成语料） |
| 审核 | `eaer/review/service.py`：任务列表、逐份或批量决定、更正、删除 | AC-6-5 |
| 查询执行 | `eaer/orchestrator/`：时间点识别、Router、Fast、Deep（比较、多跳、因果三个计划模板）、Verified（语义检查与定向补检索）、LangGraph 编排与检查点、异步执行（超时转后台、进度、取消、中断恢复）、回答缓存（按租户与授权范围隔离）、Trace 清理、`query_log` | AC-7-6、AC-8-1~8-6、AC-9-2、AC-10-4、AC-10-10、AC-12-1 |
| 对外接口 | `eaer/api/`：提问（含流式）、会话、取消、上传、审核任务、审核决定、归档、取消归档、证据、强制验证清单；接口定义 `eaer/api/openapi.yaml`（OpenAPI 3.1，接口进程的 `/openapi.json`、`/docs`）；参考令牌身份校验，文档权限经授权接口 | AC-11-1（契约测试 `tests/test_contract.py`）、AC-11-2、AC-10-6 |
| 检索 | `eaer/retrieval/`：引擎接口（OpenSearch 实现与内存参考实现）、对账、混合检索与数据库核对、回溯检索；每次查询固定一个索引视图。全量重建与旧索引清理在 `eaer/ingestion/indexer.py`（`rebuild`、`drop_retired`） | AC-5-1、AC-5-2、AC-5-3、AC-5-5、AC-6-7 |
| 对接接口 | `eaer/adapters/`：DataSource、Authorizer、ModelClient 三个接口，本地文件夹与本地账号两个参考实现，一致性套件 `conformance.py` | AC-11-4 |
| 准入检查 | `eaer/evalrun/admission.py`：验证器误通过率与误拒率、Deep 正确率、能力矩阵；`eaer/orchestrator/admission.py`：按准入结论开关 Verified 与 Deep | AC-12-3、AC-12-6 |
| 提示注入防护 | `eaer/domain/injection.py`：证据区随机边界与转义、指令型文字识别；`eaer/evalrun/injection.py` 用真实模型测回答被带偏的比例 | AC-10-7 |
| 模块边界 | `tests/test_module_boundaries.py`：静态检查开发文档 3.2 的索引读写、版本状态、模型 SDK 三条规则 | AC-3-1 |
| 检索评测 | `eaer/evalrun/retrieval.py`：按日期回放合成语料，每个查询在提问日提出，计算召回、拒答与越权 | AC-5-4 的评测流程 |
| 来源等级核对 | `eaer/evalrun/grades.py`：按日期回放合成集，核对回答标注的等级与“未验证”标注与事实表一致（不调用模型服务） | AC-7-3 |
| 评测数据许可 | `eaer/evalrun/license.py`：每份评测报告带数据集标记，`--external` 生成对外报告时按许可拦截 | AC-14-4 |
| 进程隔离 | `tests/livestack.py` 以子进程启动真实的接口与后台任务进程，`tests/stubmodel.py` 是 OpenAI 兼容的模型服务替身 | AC-3-2 |
| 索引更新延迟 | `eaer/ingestion/stats.py`（`python -m eaer.ingestion stats`）统计索引任务从排队到可检索的 P95；`eaer/evalrun/indexlag.py` 压测；后台任务按阶段的优先级 `eaer/domain/jobs.py` | AC-5-6 |
| 可观测 | `eaer/observability.py`：每次查询一条 OpenTelemetry trace，原文不进 trace；`eaer/metrics.py`（`python -m eaer.metrics`）输出监控指标 | AC-13-5 |
| 评测可复现 | `eaer/evalrun/repro.py`：同一配置连续评测两次，McNemar 精确检验判断差异是否在随机波动范围内 | AC-14-5 |
| 精简版部署 | `Dockerfile`、`docker-compose.yml`、`deploy/`（配置模板、备份恢复脚本）、`docs/部署说明.md`；冒烟测试集 `eaer/smoke.py` | AC-13-1、AC-13-3、AC-13-4 |
| 横向扩展 | `tests/test_scale_out.py`：2 个接口实例、3 个后台进程，会话与异步任务不丢失 | AC-13-2 |
| 三方对比 | `eaer/evalrun/compare3.py`：传统 Hybrid RAG、Always-Agent、Adaptive 的正确率、耗时、生成 token；报告 `docs/三方对比报告.md` | AC-8-11 |
| 评测装载入口 | `eaer/evalload/loader.py`：按历史日期装载事实表，只在 `EAER_ENV=eval` 时可用 | 开发文档 14.2 |
| 事实表工具 | `factset/`：校验、生成文档、检查文档、计算答案，用法见 `factset/README.md` | AC-14-1、AC-14-3、AC-14-2 程序部分 |

## 运行测试

需要 PostgreSQL（生产锁定 18.6）、OpenSearch 3.9.0（集成测试用，可选）和中文字体（画图表用）。

```
docker compose -f docker-compose.dev.yml up -d
pip install -r requirements.txt            # 或 requirements-docling.txt：另含 Docling，解析 PDF、扫描件、PPTX、XLSX、图片
export EAER_TEST_ADMIN_URL=postgresql://postgres:postgres@localhost:5432/postgres
export EAER_TEST_OPENSEARCH_URL=http://localhost:9200     # 不设置则跳过 OpenSearch 集成测试
python -m pytest -q
```

每个数据库测试使用从已迁移模板复制出的独立数据库。连不上数据库时，数据库测试跳过，其余照常运行。`.github/workflows/ci.yml` 在 PostgreSQL 18.6 与 Python 3.14 上运行全部测试。

## 启动

```
export EAER_DATABASE_URL=postgresql://用户@主机/库名
export EAER_STORAGE_DIR=/var/lib/eaer/files
export EAER_TOKENS_FILE=/etc/eaer/tokens.json     # 参考实现的令牌表与文档权限角色，格式见 eaer/api/main.py
export EAER_OPENSEARCH_URL=http://localhost:9200
# 模型服务：生成、Embedding、重排分别配置，变量见 eaer/adapters/online.py；本地开发写在 .env（不纳入版本控制）
alembic upgrade head
python -m modelserver --port 8100 --env-file .env  # 本地 Embedding 与重排服务（Qwen3，可选；也可用线上服务）
python -m eaer.ingestion rebuild --build-id b1     # 首次部署：建索引并挂上别名
EAER_ENV=eval python -m eaer.evalrun.admission docs/事实表样例.xlsx --admin-url … --database-url $EAER_DATABASE_URL --tenant t1
                                                   # 准入检查：通过后才开启 Verified 与 Deep（本地开发可设 EAER_ADMISSION=off）
uvicorn eaer.api.main:app --port 8000              # 接口进程
python -m eaer.ingestion                           # 后台任务进程（含索引任务与转入后台的提问；缺配置即报错）
python -m eaer.ingestion drop-retired              # 定时运行：删除超过 24 小时的旧索引
python -m eaer.ingestion purge                     # 手动做一次 Trace 清理（后台任务进程每小时自动做）
```

当前模型组合（2026-11-13）：生成 DeepSeek `deepseek-flash`（关闭推理、温度 0）；Embedding 与重排为本地部署的
Qwen3-Embedding-0.6B、Qwen3-Reranker-0.6B（`modelserver/`）。文档内容会发送到线上生成服务，部署说明中须向客户明示（12.2）。

## 迁移

```
export EAER_DATABASE_URL=postgresql://用户@主机/库名
alembic upgrade head      # 升级
alembic downgrade base    # 回滚
```

## 检索评测（事实表样例，内存索引，向量化测试替身）

```
召回率（不含需视觉模型的查询）：91.7%（12 个中命中 11 个）
拒答类查询：3 个正确，0 个泄露取值
越权返回：0 次；索引对账差异：0 个版本
未命中：Q05 版本差异——只看检索时取不到已被取代的旧版本；经编排器由 Deep 的比较模板分别检索新旧版本后答对
单列：Q15 图表——需要视觉模型
```

这组数字验证的是评测流程、权限与时间过滤；样例只有 16 个查询，且没有使用真实的向量模型，不能当作产品基线。

## 几个设计取舍

- **不变量放在数据库里。** 生命周期服务先做业务检查并给出可读的错误；触发器是第二道防线，应用代码出错时直接拒绝写入。触发器的报错以 `invariant:` 开头。
- **评测与生产共用一套生命周期代码。** 评测装载入口只是换了一个按历史日期推进的时钟。
- **删除待审核版本时保留审核记录。** 记录谁在何时删除，但不指向版本、不保留抽取内容，满足开发文档 10.4 的审计要求，也满足 AC-4-4 的“无残留”。
- **权限变更新建策略行。** 旧策略保留，变更前后的角色写进生命周期事件。
- **向量缓存。** 向量按“证据 + 模型”存进数据库。取消归档、权限变更、换分词器重建索引都不重新调用模型，只有换 Embedding 模型才重新计算，避免按量计费的线上模型被重复收费。
- **索引任务按版本对齐。** 每次任务都按数据库当前状态把整个版本对齐到索引，可以重复执行、不怕乱序；检索结果返回前再到数据库核对一次，覆盖索引更新的延迟。

## 尚未实现

- **Docling 的真实样本核对（AC-6-1）。** Docling 2.143.1 已接入（PDF、扫描件、PPTX、XLSX、图片，Word 与 HTML 仍用原生解析器），只在测试生成的样例上验证过。版面与表格模型首次使用时从 Hugging Face 下载；私有化环境先运行 `docling-tools models download`，再用 `EAER_DOCLING_ARTIFACTS` 指向模型目录。
- **CI 与旧索引的定期清理。** OpenSearch 集成测试（过滤等价、写入与删除、检索评测回放、重建期间持续查询的别名切换）已在本地 OpenSearch 3.9.0 上通过；CI 已配置但还没有运行过。旧索引按 5.3 保留 24 小时，`drop_retired` 需要由部署方定时调用（第 13 节部署尚未实现）。
- **真实的基线。** 编排器已在线上生成模型与本地 Qwen3 上试跑（16 个查询答对 15 个），但样本太小，不是产品基线；需要扩充评测集（AC-5-4、AC-8-9）。
- **图表深度解析。** 需要视觉模型；目前表格已解析，图表类事实在评测中单列为“需视觉模型”。
- **OpenTelemetry trace（13.3）。** 每个节点的耗时已记在检查点与 query_log 中，尚未导出到 OpenTelemetry；接入后其后端的保留期按 trace_retention_days 配置。
- **分类的模型兜底。** 规则判断不了时目前留给审核人确认。
