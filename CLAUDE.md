# EAER 项目说明（给 Claude Code）

EAER 是面向多家客户交付的标准企业级 RAG 产品。用户不仅得到答案，还能看到答案来自哪份文档的哪个版本、哪一处，以及该证据在目标时间是否有效。核心能力是 Fast、Deep、Verified 三种执行模式，加上验证层、时间处理和文档生命周期；数据源、权限、SSO 只定义接口并附参考实现，按客户项目对接。

开发与验收的唯一依据是 `docs/开发文档-v1.0.md`。如果 `docs/` 里还没有这份文件，请提醒用户按 `docs/README.md` 导出；在此之前以本文件为准。

## 先做这几件事

1. 启动依赖：`docker compose -f docker-compose.dev.yml up -d`（PostgreSQL 18.6、OpenSearch 3.9.0）。
2. 安装依赖：`pip install -r requirements.txt`（Python 3.14；画图表需要中文字体，如 Noto Sans CJK）。要解析 PDF、扫描件、PPTX、XLSX、图片时改装 `pip install -r requirements-docling.txt`（已包含核心依赖，带 PyTorch；首次解析会下载版面模型）。
3. 设置环境变量并跑全部测试：
   ```
   export EAER_TEST_ADMIN_URL=postgresql://postgres:postgres@localhost:5432/postgres
   export EAER_TEST_OPENSEARCH_URL=http://localhost:9200
   python -m pytest -q
   ```
   2026-11-12 首次在真实环境运行（macOS + OrbStack，用 `docker-compose.dev.yml` 启动 PostgreSQL 18.6 与 OpenSearch 3.9.0，Python 3.14.5）：148 个全部通过，没有跳过和警告，没有需要修复的问题。此后 OpenSearch 改用带 smartcn、icu 插件的自定义镜像，AC-5-3 的“重建期间持续查询”已有真实引擎上的用例（`tests/test_rebuild_live.py`）。CI 仍未运行过（仓库还没有远端）。之后跑出的问题请如实告诉用户，修复后记入 `docs/设计变更记录.md`。
4. 如果还没有 git 仓库，先 `git init` 并提交一次。

## 目录

| 路径 | 内容 |
| --- | --- |
| `eaer/db/migrations/` | Alembic 迁移，原生 SQL。12 张业务表，另有查询任务表 `query_task`、LangGraph 的 4 张检查点表（0005，按库的表结构建、不在运行时建表）、回答缓存 `query_cache`（0006，失效由触发器完成）与准入记录 `model_admission`（0007，只追加）；不变量由触发器、检查约束、部分唯一索引保证 |
| `eaer/lifecycle/` | 生命周期服务（唯一能改版本状态的模块）与按系统时间的回溯查询 |
| `eaer/ingestion/` | 入库第 1–3、6 阶段，解析器接口（原生 Word/HTML 与 Docling），抽取规则，索引任务（`indexer.py`，唯一写索引的地方），后台任务进程（也接着执行转入后台的提问） |
| `eaer/review/` | 审核服务 |
| `eaer/retrieval/` | 检索引擎接口、OpenSearch 实现与客户端工厂、内存参考实现、应用侧分词（jieba 候选）、对账、检索服务（含 Deep 用的“取版本” `versions`） |
| `eaer/verifier/` | 验证层（第 9 节）：确定性检查 `deterministic.py`（引用核对、authorized、locator_exists、value_match、time_valid，可按版本指定目标时间）与语义检查 `semantic.py` |
| `eaer/orchestrator/` | 查询执行（第 8 节）：时间点识别 `timeparse.py`、Router `router.py`、Fast 与共用步骤 `fast.py`、Deep 计划模板 `deep.py`、编排与 Verified `engine.py`（LangGraph 图，状态只含 JSON 值，见 `codec.py`）、异步执行 `tasks.py`（同步时限、转后台、进度、取消、租约与检查点恢复、Trace 清理 `purge`）、回答缓存 `cache.py`、准入结论与模式开关 `admission.py`、强制验证清单 `forced_verify.py`（内置规则 + 租户追加的规则，只增不减） |
| `eaer/api/` | FastAPI 对外接口（提问、会话、取消、上传、审核、归档、取消归档、证据、强制验证清单）；文档权限经 `Authorizer`，未配置或不可用时 503。接口进程与后台任务进程的公共装配在 `eaer/runtime.py` |
| `eaer/adapters/` | 对接接口（11.3）：数据源（参考实现本地文件夹 + `_acl.yaml`）、授权（参考实现本地账号）、模型（接口、测试替身，以及线上通用客户端 `online.py`：生成、Embedding、重排分别配置，配置见该文件的文档字符串；本地开发的密钥放在未纳入版本控制的 `.env`）；`conformance.py` 是客户实现交付前必跑的一致性套件 |
| `eaer/domain/` | 证据 id 规则、身份与授权范围（`Identity`、`Scope`）、提示注入防护（`injection.py`：证据区隔离与指令型文字识别） |
| `eaer/evalload/`、`eaer/evalrun/` | 评测专用的历史日期装载入口与评测运行器（检索、Fast 与编排器试跑、分词对比、提示注入、准入检查与能力矩阵 `admission.py`），只在 `EAER_ENV=eval` 时可用 |
| `modelserver/` | 本地模型服务（参考实现，独立部署）：Qwen3-Embedding-0.6B 与 Qwen3-Reranker-0.6B，提供 `/v1/embeddings` 与 `/v1/rerank`；`python -m modelserver --port 8100 --env-file .env`，Mac 用 MPS、Linux 用 CUDA |
| `factset/` | 合成评测集工具：校验事实表、生成文档、检查文档、计算标准答案 |
| `tests/` | 每个测试文件的文档字符串注明对应的验收项（AC-x-y） |

## 必须遵守的约定

- **语言。** 文档、注释、文档字符串、错误信息、提交说明用中文；标识符用英文。
- **依赖版本。** 精确锁定，选发布满 14 天的最新稳定版（刚发布的版本可能有未发现的缺陷或恶意代码）。改依赖时同时更新 `requirements.txt` 和开发文档 3.3 节的版本表。
- **模块边界（开发文档 3.2）。** 只有 `retrieval` 读检索索引；只有 `ingestion` 和 `lifecycle` 写索引；只有 `lifecycle` 改版本状态；所有模型调用经模型接口，业务模块不直接依赖模型厂商的 SDK；授权范围只以句柄传递，不进入模型上下文。
- **不变量放在数据库。** 服务层先做业务检查并给出可读的错误，触发器是第二道防线，报错以 `invariant:` 开头。新增不变量要写迁移，并在 `tests/test_db_invariants.py` 加违规用例。
- **迁移只追加。** 已有的迁移文件不改，新改动写新的版本文件，并保证可以回滚。
- **权限由代码执行。** 每次检索都必须带授权范围；返回前在数据库中核对状态、权限和有效期；无权访问与不存在一律返回 404。
- **硬性项不能迁就实现。** 开发文档第 16.1 节列出的 31 个硬性项，测试失败就修实现，不放宽测试。
- **测试替身的数字不是产品结论。** `HashingEmbedder` 和 `MemoryIndex` 只用来验证流程和过滤逻辑，它们得出的召回率不能写进对外材料。
- **设计改动要留痕。** 相对开发文档的任何设计改动，记入 `docs/设计变更记录.md`。

## 关键规则速查

- **生命周期。** 待审核（不可搜，可删除）→ 审核通过 → 可搜索（不能删除，只能归档）→ 已归档（可取消归档，不重新审核，`approved_at` 不变）。归档以文档为单位；归档后的提问一律查不到，回溯查询也不例外；历史回答保留引用记录。删除待审核版本时，审核记录保留删除人和时间，不保留内容。审核支持逐份和批量，每份单独成事务，不做按规则自动通过。
- **时间来源等级。** 正文写明或审核人补录为 `explicit`；只有上传时间为 `inferred`；没有时间或批量导入为 `unknown`。没有依据的生效日不推断；无法换算的相对表述保留原文，不给日期。
- **作废日推导。** 新版本生效日为 `explicit` 时取该日，否则取新版本审核通过日并记为 `inferred`。新版本的生效日被更正时，旧版本的作废日随之更正。
- **三类时间查询。** 当前生效、某天实际生效（按有效期）、截至某天系统所知（回溯：按 `approved_at` 过滤，有效期取当时记录的值）。有效期与关系只追加，所以回溯可以复现。
- **强制验证清单。** 默认为“答案引用规范类文档，或查询带时间点、版本比较”；客户只能增加，不能缩小。
- **验证器。** 代码做确定性检查（定位存在、数值与日期一致、目标时间有效、有权限），模型只做语义判断；拿不准的标为“未验证”。
- **模型。** 精简版调用线上模型；客户自部署模型经同一接口接入，用自助评测工具补测，准入检查不达标就关闭 Deep 或 Verified。
- **评测数据。** FRAMES、MultiHop-RAG、ViDoRe V3 可对外；LLM-AggreFact 只能原样评测；HoH、VersionQA、CHaystack 只在内部开发使用；RGB、DragonBall、CRUD-RAG 只借鉴任务设计与结构，内容自行合成（法务已确认）。

## 当前进度（对照开发文档的验收项）

| 状态 | 验收项 |
| --- | --- |
| 已通过 | AC-3-1，AC-4-1~4-5，AC-5-1、5-2、5-3、5-5，AC-6-2、6-3、6-4（检索层面）、6-5、6-6、6-7、6-8，AC-7-2（检索层面）、7-4、7-5、7-6，AC-8-1~8-6，AC-9-1、9-2，AC-10-1（检索层面）、10-3（检索层面）、10-4、10-5、10-6、10-7（越权 0；带偏比例为指标，待基线）、10-9、10-10，AC-11-2、11-3、11-4（三个接口的参考实现均通过；ModelClient 为 DeepSeek 生成 + 本地 Qwen3 Embedding 与重排）、11-5、11-6，AC-12-1、12-3、12-5、12-6（线上模型一档，样例上），AC-14-1、14-3 |
| 合成语料上通过，待真实数据 | AC-6-9（分类）、AC-7-7（时间抽取）、AC-14-2 的程序检查部分 |
| 只有逻辑或流程，未在真实环境验证 | AC-5-4（检索评测流程已有，用的是测试替身）、AC-3-3（CI 已配置，未运行） |
| 部分完成 | AC-11-1（契约测试只覆盖已有接口，尚无 OpenAPI 文件） |
| 未开始 | AC-3-2，AC-6-1、6-10，第 12–13 节其余（计量对账 AC-12-4、部署运维） |

事实表样例上的检索评测：可计分的 12 个查询命中 11 个；3 个拒答类全部正确；越权 0；对账差异 0。未命中的 Q05 是版本差异类，只看检索时取不到旧版本，经编排器由 Deep 的比较模板答对；Q15 图表类需要视觉模型。

编排器试跑（2026-11-13，`python -m eaer.evalrun.fast --engine`，不传时间参数、由编排器从问题中识别）：生成 DeepSeek `deepseek-flash`（关闭推理、温度 0），Embedding 与重排为本地 Qwen3-0.6B，内存索引。加入 Deep 后 16 个查询正确 15 个，连续两次相同；预期模式全部出现在 Router 决定或实际路径中（12/12）。Q05 经 deep → verified：定位文档、取版本、在第 2 版与第 3 版各自有效期内分别检索，三条结论全部通过；Q16（680 的扰动结论）判为不支持。唯一未答对的 Q15 需要图表的视觉解析。Q09、Q12 两个拒答类因重排分低从 Fast 转入 Deep，仍正确拒答但多花 4–7 秒。耗时 P50 约 4.2–4.8 秒、P95 约 7.9–9.1 秒（加入 Deep 前 P50 约 3.9 秒）。编排器改为 LangGraph 图后重跑，结果相同（15/16，P50 4.5 秒、P95 8.1 秒）。样本只有 16 个查询，这不是产品基线。

分词器对比（用 Qwen3-Embedding-0.6B 重跑）：五种候选在 docx、html 两套语料上召回都是 11/12，MRR 0.79–0.88，同一分词器在两套语料上名次也会互换，样本太小，仍然区分不出来。

## 下一步（按优先级）

1. **在真实环境跑通全部测试。** 已于 2026-11-12 完成（148 个通过）。以后如果真实引擎的行为与内存实现不一致，以等价性测试 `test_dsl_and_python_filters_agree` 的思路排查。
2. **中文分词。** 自定义镜像（smartcn、icu）、应用侧 jieba 候选和对比工具 `python -m eaer.evalrun.analyzers` 已完成（2026-11-12）。事实表样例只有 12 个可计分查询，五种候选召回都是 11/12，MRR 在 0.79–0.88 之间，接入真实 Embedding（Qwen3-0.6B）后依然区分不出来；用户倾向 jieba。扩充评测集后重跑，按 AC-5-4 选定。jieba 0.42.1 自 2020 年未更新，在 Python 3.14 上导入有 SyntaxWarning，正式采用前评估替代实现。
3. **补齐第 4 周关口中不依赖外部资源的项。** 已于 2026-11-12 完成：AC-3-1（`tests/test_module_boundaries.py`）、AC-6-3（`tests/test_state_machine.py`）、AC-6-4（`tests/test_pending_not_searchable.py`，检索层面）、AC-11-4（`eaer/adapters/conformance.py`）。
4. **接入 Docling。** 已于 2026-11-12 完成：`eaer/ingestion/docling_parser.py`，覆盖 PDF、扫描件（RapidOCR）、PPTX、XLSX、图片，证据定位带页码；依赖单独锁定在 `requirements-docling.txt`。只在合成样例上验证过，AC-6-1 要求的“每种格式 20 份真实样本”人工核对尚未做。
5. **模型接口、Fast、Router 与 Verified。** 已完成（2026-11-13）：线上通用客户端、本地 Qwen3 模型服务、Fast、Router（8.1）、Verified（语义检查、定向补检索）、编排器（`eaer/orchestrator/engine.py`）、提问与会话接口、`query_log`。后台任务进程已接上索引任务：需要 `EAER_OPENSEARCH_URL` 与 Embedding 服务，缺少即启动报错（`--no-index` 只做解析）；先用 `python -m eaer.ingestion rebuild --build-id …` 建索引，`drop-retired` 要定时运行。
6. **Deep 模式。** 已完成（2026-11-13）：`eaer/orchestrator/deep.py` 的比较（版本、时间点、一般对象）、多跳、因果三个计划模板，Fast 的三个信号转入 Deep，检索服务的“取版本”，确定性检查按版本的目标时间，`tests/test_deep.py`。默认开启，`EAER_DEEP_ENABLED=0` 关闭。
7. **异步执行。** 已完成（2026-11-13）：编排器改为 LangGraph 图（prepare 之后 first → escalate → verify → assemble），检查点写入 PostgreSQL（迁移 0005）；`eaer/orchestrator/tasks.py` 的同步时限（`budgets.sync_seconds`，初值 30 秒）、转后台、进度、取消、租约与中断恢复；后台任务进程接着执行，需要与接口进程相同的查询配置与 `EAER_TOKENS_FILE`；`tests/test_async.py`。
8. **提示注入防护。** 已完成（2026-11-13）：证据区随机边界与转义、系统提示声明区内指令不执行、指令型文字识别（引用即转 Verified、入库时提示审核人）、`tests/test_injection.py`（照办一切指令的最坏情况模型，植入 60 条，越权 0）。真实模型（DeepSeek）在事实表样例植入 56 段后：越权 0；回答被带偏首次 3/15，修正拒答说明后 0/15；正确 13/15（未植入时 15/16）。`python -m eaer.evalrun.injection` 复测。
9. **缓存隔离与 Trace 清理。** 已完成（2026-11-13）：回答缓存（键含租户、授权范围哈希、索引构建编号；数据库触发器在权限、状态、有效期、关系、租户配置变化时清除该租户缓存；命中后仍核对权限）、Trace 清理（已结束任务的检查点与任务行按 trace_retention_days 删除，后台任务进程每小时一次，`python -m eaer.ingestion purge` 手动），`tests/test_cache.py`。OpenTelemetry trace（13.3）未接入。
10. **准入检查。** 已完成（2026-11-13）：`python -m eaer.evalrun.admission`（验证器评测集由事实表生成、标定与评测分开、误通过率与误拒率门槛；Deep 比较题正确率门槛），结论写入 `model_admission`，部署时据此开关 Verified 与 Deep，没有记录的模型两者关闭（`EAER_ADMISSION=off` 只用于本地开发）。DeepSeek + 本地 Qwen3 在样例上：Verified 误通过 0%、误拒 22%，Deep 10/10，能力矩阵见 `docs/能力矩阵-线上模型.md`。本地跑接口前先用 `--database-url … --tenant …` 写入准入结论。
11. **其余未开始的项。** OpenAPI 文件与契约测试（AC-11-1）、扩充评测集后选定分词与模型（AC-5-4）。

## 需要用户决定或提供的事项

- 模型服务（2026-11-13 确定）：生成用 DeepSeek（`deepseek-flash`）；Embedding 与重排由用户选定本地部署 Qwen3（`modelserver/`，先用 0.6B）。是否换 4B/8B 在扩充评测集后按 AC-5-4 比较决定。
- 确认“删除待审核版本时保留审核记录”。
- 集群版私有化环境的 S3 兼容存储产品（第 13 周前）。MinIO 社区版已停止维护，候选为 Garage、SeaweedFS、Ceph RGW。
- 各指标项的门槛，在基线评测后与用户一起回填到开发文档 16.2 节；准入门槛的初值（误通过率 ≤ 5%、误拒率 ≤ 30%、Deep 正确率 ≥ 70%）待确认。
