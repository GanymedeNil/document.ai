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
   2026-11-12 首次在真实环境运行（macOS + OrbStack，用 `docker-compose.dev.yml` 启动 PostgreSQL 18.6 与 OpenSearch 3.9.0，Python 3.14.5）：148 个全部通过，没有跳过和警告，没有需要修复的问题。此后 OpenSearch 改用带 smartcn、icu 插件的自定义镜像（默认分词已选定内置的 standard，插件留给对比工具与选用它们的客户），AC-5-3 的“重建期间持续查询”已有真实引擎上的用例（`tests/test_rebuild_live.py`）。CI 仍未运行过（仓库还没有远端）。之后跑出的问题请如实告诉用户，修复后记入 `docs/设计变更记录.md`。
4. 如果还没有 git 仓库，先 `git init` 并提交一次。

## 目录

| 路径 | 内容 |
| --- | --- |
| `eaer/db/migrations/` | Alembic 迁移，原生 SQL。12 张业务表，另有查询任务表 `query_task`、LangGraph 的 4 张检查点表（0005，按库的表结构建、不在运行时建表）、回答缓存 `query_cache`（0006，失效由触发器完成）与准入记录 `model_admission`（0007，只追加）；0008 给审核任务加所属租户 `tenant_id`；0009 让索引任务完成时也清缓存，并加每租户的失效序号 `query_cache_epoch`（写缓存前比较，堵住先算后写的竞争）；不变量由触发器、检查约束、部分唯一索引保证 |
| `eaer/lifecycle/` | 生命周期服务（唯一能改版本状态的模块）与按系统时间的回溯查询 |
| `eaer/ingestion/` | 入库第 1–3、6 阶段，解析器接口（原生 Word/HTML 与 Docling），抽取规则，索引任务（`indexer.py`，唯一写索引的地方），后台任务进程（也接着执行转入后台的提问；任务按阶段的优先级领取，见 `eaer/domain/jobs.py`），队列与索引更新延迟统计（`stats.py`，`python -m eaer.ingestion stats` 输出 JSON） |
| `eaer/review/` | 审核服务 |
| `eaer/retrieval/` | 检索引擎接口、OpenSearch 实现与客户端工厂、内存参考实现、应用侧分词（jieba 候选）；默认分词方式 `DEFAULT_ANALYZER = "standard"`（`opensearch.py`，`EAER_TEXT_ANALYZER` 可改，改了要重建索引）、对账、检索服务（含 Deep 用的“取版本” `versions`） |
| `eaer/verifier/` | 验证层（第 9 节）：确定性检查 `deterministic.py`（引用核对、authorized、locator_exists、value_match、time_valid，可按版本指定目标时间）与语义检查 `semantic.py` |
| `eaer/orchestrator/` | 查询执行（第 8 节）：时间点识别 `timeparse.py`、Router `router.py`、Fast 与共用步骤 `fast.py`、Deep 计划模板 `deep.py`、编排与 Verified `engine.py`（LangGraph 图，状态只含 JSON 值，见 `codec.py`）、异步执行 `tasks.py`（同步时限、转后台、进度、取消、租约与检查点恢复、Trace 清理 `purge`）、回答缓存 `cache.py`、准入结论与模式开关 `admission.py`、强制验证清单 `forced_verify.py`（内置规则 + 租户追加的规则，只增不减） |
| `eaer/api/` | FastAPI 对外接口（提问、会话、取消、上传、审核、归档、取消归档、证据、强制验证清单）；接口的唯一定义是 `openapi.yaml`（OpenAPI 3.1，`/openapi.json` 展示的就是它，改接口先改它，契约测试 `tests/test_contract.py`）；文档权限经 `Authorizer`，未配置或不可用时 503。接口进程与后台任务进程的公共装配在 `eaer/runtime.py` |
| `eaer/adapters/` | 对接接口（11.3）：数据源（参考实现本地文件夹 + `_acl.yaml`）、授权（参考实现本地账号）、模型（接口、测试替身，以及线上通用客户端 `online.py`：生成、Embedding、重排分别配置，配置见该文件的文档字符串；本地开发的密钥放在未纳入版本控制的 `.env`）；`conformance.py` 是客户实现交付前必跑的一致性套件 |
| `eaer/domain/` | 证据 id 规则、身份与授权范围（`Identity`、`Scope`）、提示注入防护（`injection.py`：证据区隔离与指令型文字识别）、后台任务优先级（`jobs.py`） |
| `eaer/observability.py`、`eaer/metrics.py` | 可观测（13.3）：每次查询一条 OpenTelemetry trace（节点、检索候选数、模型 token 数；原文不进 trace），`EAER_TRACE_EXPORTER` 选 none / console / otlp（otlp 需装 `requirements-otlp.txt`）；`python -m eaer.metrics` 输出监控指标 JSON |
| `eaer/evalload/`、`eaer/evalrun/` | 评测专用的历史日期装载入口与评测运行器（检索、Fast 与编排器试跑、分词对比、提示注入、索引更新延迟压测 `indexlag.py`、复现检查 `repro.py`、准入检查与能力矩阵 `admission.py`、来源等级核对 `grades.py`），只在 `EAER_ENV=eval` 时可用；报告都经 `license.py` 加数据集标记，`--external` 时按许可拦截 |
| `modelserver/` | 本地模型服务（参考实现，独立部署）：Qwen3-Embedding-0.6B 与 Qwen3-Reranker-0.6B，提供 `/v1/embeddings` 与 `/v1/rerank`；`python -m modelserver --port 8100 --env-file .env`，Mac 用 MPS、Linux 用 CUDA |
| `factset/` | 合成评测集工具：校验事实表、生成文档、检查文档、计算标准答案、生成合成集（`synth`）、抽检单（`sample`）；评测集在 `docs/评测集/`（当前 v2，2000 条） |
| `tests/` | 每个测试文件的文档字符串注明对应的验收项（AC-x-y）；`livestack.py` 以子进程启动真实的接口与后台任务进程，`stubmodel.py` 是 OpenAI 兼容的模型服务替身（不调用线上模型） |

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
| 已通过 | AC-3-1、3-2（真实的接口与后台任务进程，停掉后台进程时提问成功率 100%、任务排队，恢复后全部完成），AC-4-1~4-5，AC-5-1、5-2、5-3、5-5，AC-6-2、6-3、6-4（检索层面）、6-5、6-6、6-7、6-8，AC-7-2（检索层面）、7-4、7-5、7-6，AC-8-1~8-6，AC-9-1、9-2，AC-10-1（检索层面）、10-3（检索层面）、10-4、10-5、10-6、10-7（越权 0；带偏比例为指标，待基线）、10-9、10-10，AC-11-1（契约测试：11 个操作的每个声明状态码）、11-2、11-3、11-4（三个接口的参考实现均通过；ModelClient 为 DeepSeek 生成 + 本地 Qwen3 Embedding 与重排）、11-5、11-6，AC-12-1、12-3、12-5、12-6（线上模型一档，样例上），AC-14-1、14-3、14-4（许可登记与对外报告拦截；人工复核未做）、14-5（`python -m eaer.evalrun.repro`：DeepSeek + 本地 Qwen3、温度 0，事实表样例两次逐条结果 16/16 相同；合成集 v2 测试集每类 5 条共 50 条，两次各对 42 条、不一致对 1 + 1、McNemar p = 1.0；回答原文完全相同的只有 19/50，措辞有波动、结论不变）。另有 AC-7-1（事实表样例 Q01–Q08 端到端冒烟，需模型服务：DeepSeek + 本地 Qwen3 上 8/8，连续两次）、AC-7-3（合成集 v2 全量 1408 个查询、1932 条依据，等级与“未验证”标注不一致 0）、AC-10-2（两个世界对照，用户看到的与模型看到的完全相同） |
| 合成语料上通过，待真实数据 | AC-6-9（分类）、AC-7-7（时间抽取）、AC-14-2 的程序检查部分（合成集 v2 全部通过，人工抽检待做） |
| 有试跑数字，待基线（门槛与用户一起回填 16.2） | AC-5-6（索引更新延迟：持续负载通过、突发受 Embedding 吞吐所限，见下文“索引更新延迟压测”；以哪种负载验收待用户决定）、AC-7-8（历史时点与回溯）、8-7（Router）、8-8（拒答；还需 MultiHop-RAG）、8-9（正确率）、8-10（延迟与成本；Fast 的 P95 未单独统计）、9-3（准入检查上误通过 0%、误拒 22%；LLM-AggreFact 未跑）、14-6（图表 0/10，需视觉模型） |
| 已实现，待人工核对 | AC-9-5（标定与评测数据的划分记录）、AC-13-5（可观测：自动化用例已覆盖每个节点都有 span、一次查询一条 trace（跨进程续接）、原文不进 trace；人工抽查待做） |
| 只有逻辑或流程，未在真实环境验证 | AC-3-3（CI 已配置，未运行） |
| 部分完成 | AC-5-4（分词器已选定 standard；英文召回未测；Embedding 与重排模型的规格待选定） |
| 需要人工或外部资源 | AC-6-1（每种格式 20 份真实样本）、9-4（人工核对的结论集）、10-8（人工红队）、12-2（第二个线上模型）、12-4（模型账单） |
| 未开始 | AC-6-10（入库吞吐）、8-11（三方对比报告），AC-13-1~13-4、13-6（部署、集群扩展、重建与备份演练、机器规格表） |

事实表样例上的检索评测：可计分的 12 个查询命中 11 个；3 个拒答类全部正确；越权 0；对账差异 0。未命中的 Q05 是版本差异类，只看检索时取不到旧版本，经编排器由 Deep 的比较模板答对；Q15 图表类需要视觉模型。

编排器试跑（2026-11-13，`python -m eaer.evalrun.fast --engine`，不传时间参数、由编排器从问题中识别）：生成 DeepSeek `deepseek-flash`（关闭推理、温度 0），Embedding 与重排为本地 Qwen3-0.6B，内存索引。加入 Deep 后 16 个查询正确 15 个，连续两次相同；预期模式全部出现在 Router 决定或实际路径中（12/12）。Q05 经 deep → verified：定位文档、取版本、在第 2 版与第 3 版各自有效期内分别检索，三条结论全部通过；Q16（680 的扰动结论）判为不支持。唯一未答对的 Q15 需要图表的视觉解析。Q09、Q12 两个拒答类因重排分低从 Fast 转入 Deep，仍正确拒答但多花 4–7 秒。耗时 P50 约 4.2–4.8 秒、P95 约 7.9–9.1 秒（加入 Deep 前 P50 约 3.9 秒）。编排器改为 LangGraph 图后重跑，结果相同（15/16，P50 4.5 秒、P95 8.1 秒）。样本只有 16 个查询，这不是产品基线。

索引更新延迟压测（AC-5-6，2026-11-14，`python -m eaer.evalrun.indexlag`，真实 OpenSearch、本地 Qwen3-Embedding-0.6B（MPS），后台 2 个线程，同时有每秒 2 个变更事件（归档、取消归档、权限、有效期）、陆续上传的新文档、4 个查询线程；每份文档 9 条证据）：持续负载每秒审核通过 1 份（共 300 份）——索引任务 P95 0.7 秒、最大 2.2 秒，通过；突发 200 份——写入任务 P95 59.7 秒，刚好在 60 秒以内；突发 500 份——写入任务 P95 140 秒，不通过，索引更新任务 P95 0.7 秒。瓶颈是 Embedding 吞吐（单份 8 条约 0.16 秒，合批不变快，OpenSearch 写入每份约 0.02 秒），突发的延迟约等于“突发证据条数 ÷ Embedding 吞吐”，加后台线程无用，要靠更快的 Embedding 服务。首次压测时索引更新任务排在写入任务后面（P95 147 秒），已改为按阶段的优先级（设计变更记录）。所有压测任务失败 0、对账差异 0、查询错误 0。

分词器对比（合成集 v2 测试集 1400 条，可计分 1052 条，Qwen3-Embedding-0.6B，真实 OpenSearch，HTML 语料）：原词问法各分词器都是 92% 左右；改写问法（不含事实键原词）拉开差距——仅关键词：standard 86.7%、cjk 73.8%、icu 68.8%、jieba 66.2%、smartcn 65.6%；关键词加向量：standard 91.2%、cjk 88.4%、icu 87.3%、jieba 86.9%、smartcn 85.4%。拒答 278/278，越权 0。版本差异类 0/70 是设计如此（检索只取当前版本，由 Deep 分别检索新旧版本）。内存索引上的旁证：改写问法单走向量的召回高于关键词加向量（RRF 融合把只在向量一路排前的结果压下去了），融合方式值得按标定集调。编排器试跑（合成集 v2 测试集每类 10 条，共 100 条，DeepSeek + 本地 Qwen3）：正确 82/100，除图表类外 82/90；
图表类 0/10（需视觉模型，全部拒答而不是编造）；版本差异、当前生效、权限边界、无答案、验证器各 10/10；冲突类 6/10——同等可信的补充通知与
主制度取值不同时，模型常按“新的覆盖旧的”只答一个，而规则要求全部列出，这是需要改进的地方；另有 5 条错在三个不等价的改写说法上
（评测集的问题，见 `docs/评测集/README.md`）。P50 5.9 秒、P95 15.8 秒。这次试跑还发现并修正了 Router 对“从多少改成了多少”的漏判
与两处计分错误。注意：2026-11-12 至 11-13 早先的对比（含“五种候选都是 11/12”与合成集 v1）里“仅关键词”一栏因工具缺陷实际是关键词加向量，已修正。合成数据、改写说法由我手写，结论待真实语料复核。

## 下一步（按优先级）

1. **在真实环境跑通全部测试。** 已于 2026-11-12 完成（148 个通过）。以后如果真实引擎的行为与内存实现不一致，以等价性测试 `test_dsl_and_python_filters_agree` 的思路排查。
2. **中文分词。** 自定义镜像（smartcn、icu）、应用侧 jieba 候选和对比工具 `python -m eaer.evalrun.analyzers` 已完成（2026-11-12）。合成集 v2 上已能区分（见上文）：改写问法下按字切分的 standard 最好、jieba 与 smartcn 最差。**2026-11-13 用户选定 standard**，已改为部署默认值（映射、后台任务进程、`rebuild` 命令都默认用它；已有的 smartcn 索引须用 `python -m eaer.ingestion rebuild` 重建）。按字切分在大语料上可能带来更多噪声与更大的索引，接入真实语料后复核召回、精度与索引大小。smartcn、icu、jieba 保留为可选项；jieba 0.42.1 自 2020 年未更新，在 Python 3.14 上导入有 SyntaxWarning，有客户要用时再评估替代实现。
3. **补齐第 4 周关口中不依赖外部资源的项。** 已于 2026-11-12 完成：AC-3-1（`tests/test_module_boundaries.py`）、AC-6-3（`tests/test_state_machine.py`）、AC-6-4（`tests/test_pending_not_searchable.py`，检索层面）、AC-11-4（`eaer/adapters/conformance.py`）。
4. **接入 Docling。** 已于 2026-11-12 完成：`eaer/ingestion/docling_parser.py`，覆盖 PDF、扫描件（RapidOCR）、PPTX、XLSX、图片，证据定位带页码；依赖单独锁定在 `requirements-docling.txt`。只在合成样例上验证过，AC-6-1 要求的“每种格式 20 份真实样本”人工核对尚未做。
5. **模型接口、Fast、Router 与 Verified。** 已完成（2026-11-13）：线上通用客户端、本地 Qwen3 模型服务、Fast、Router（8.1）、Verified（语义检查、定向补检索）、编排器（`eaer/orchestrator/engine.py`）、提问与会话接口、`query_log`。后台任务进程已接上索引任务：需要 `EAER_OPENSEARCH_URL` 与 Embedding 服务，缺少即启动报错（`--no-index` 只做解析）；先用 `python -m eaer.ingestion rebuild --build-id …` 建索引，`drop-retired` 要定时运行。
6. **Deep 模式。** 已完成（2026-11-13）：`eaer/orchestrator/deep.py` 的比较（版本、时间点、一般对象）、多跳、因果三个计划模板，Fast 的三个信号转入 Deep，检索服务的“取版本”，确定性检查按版本的目标时间，`tests/test_deep.py`。默认开启，`EAER_DEEP_ENABLED=0` 关闭。
7. **异步执行。** 已完成（2026-11-13）：编排器改为 LangGraph 图（prepare 之后 first → escalate → verify → assemble），检查点写入 PostgreSQL（迁移 0005）；`eaer/orchestrator/tasks.py` 的同步时限（`budgets.sync_seconds`，初值 30 秒）、转后台、进度、取消、租约与中断恢复；后台任务进程接着执行，需要与接口进程相同的查询配置与 `EAER_TOKENS_FILE`；`tests/test_async.py`。
8. **提示注入防护。** 已完成（2026-11-13）：证据区随机边界与转义、系统提示声明区内指令不执行、指令型文字识别（引用即转 Verified、入库时提示审核人）、`tests/test_injection.py`（照办一切指令的最坏情况模型，植入 60 条，越权 0）。真实模型（DeepSeek）在事实表样例植入 56 段后：越权 0；回答被带偏首次 3/15，修正拒答说明后 0/15；正确 13/15（未植入时 15/16）。`python -m eaer.evalrun.injection` 复测。
9. **缓存隔离、Trace 清理与可观测。** 已完成（2026-11-13，可观测 2026-11-14）：回答缓存（键含租户、授权范围哈希、索引构建编号；数据库触发器在权限、状态、有效期、关系、租户配置变化时清除该租户缓存；命中后仍核对权限）、Trace 清理（已结束任务的检查点与任务行按 trace_retention_days 删除，后台任务进程每小时一次，`python -m eaer.ingestion purge` 手动），`tests/test_cache.py`。OpenTelemetry trace 与监控指标（13.3）于 2026-11-14 接入：`eaer/observability.py`、`python -m eaer.metrics`、`tests/test_observability.py`。
10. **准入检查。** 已完成（2026-11-13）：`python -m eaer.evalrun.admission`（验证器评测集由事实表生成、标定与评测分开、误通过率与误拒率门槛；Deep 比较题正确率门槛），结论写入 `model_admission`，部署时据此开关 Verified 与 Deep，没有记录的模型两者关闭（`EAER_ADMISSION=off` 只用于本地开发）。DeepSeek + 本地 Qwen3 在样例上：Verified 误通过 0%、误拒 22%，Deep 10/10，能力矩阵见 `docs/能力矩阵-线上模型.md`。本地跑接口前先用 `--database-url … --tenant …` 写入准入结论。
11. **扩充评测集。** 已完成（2026-11-13）：`factset/synth.py` 按 14.3 的样本量生成合成集（高风险类别各 300、其他各 100，共 2000 条，30% 标定、70% 测试），v2 约四成查询用改写问法；`docs/评测集/合成集-v2.xlsx` 冻结（测试核对与生成器一致），抽检单 `合成集-v2-抽检单.xlsx`（100 条）待人工核对。评测运行器支持 `--split 测试`、`--per-type N`。
12. **OpenAPI 文件与契约测试（AC-11-1）。** 已完成（2026-11-13）：`eaer/api/openapi.yaml`、`tests/contract.py`（按文件校验请求与响应的检查器）、`tests/test_contract.py`。审核决定的 `corrections` 改为按格式校验（格式错误整个请求 400）。当时发现的 `GET /v1/review/tasks?status=deleted` 永远为空（删除后任务不再指向版本，按租户过滤要经版本关联文档）已于 2026-11-14 修正：迁移 0008 给 review_task 加 tenant_id（从版本回填；迁移前已删除的任务无从得知租户，保持为空、不推断），触发器保证与版本所属文档的租户一致且不可修改；列表与处理都按任务自己的租户判断，每项多了 reviewer、decided_at。429 频率限制未实现。
13. **补齐未过的硬性项。** 已完成（2026-11-14）：AC-10-2 关系泄露（`tests/test_relation_leak.py`，同一语料放进有无 R_FIN 文件的两个租户对照，实现无需改动；OpenSearch BM25 全局统计量的侧信道记入设计变更记录）、AC-7-3 来源等级如实（`python -m eaer.evalrun.grades`，固定依据的检索与模型替身，不调模型服务）、AC-14-4 许可合规（`eaer/evalrun/license.py`，数据来源按“数据来源”表或冻结评测集的 SHA-256 判断，运行器 `--external` 时按许可拦截；冻结新评测集要在 `FROZEN` 登记）、AC-7-1 样例冒烟（`tests/test_smoke_sample.py`，没配模型服务时跳过；本地运行前先启动 `python -m modelserver --port 8100 --env-file .env`，再 `set -a; source .env; set +a`）。31 个硬性项全部有自动化验收。顺带发现评测集问题：Q1621 等版本差异题两个版本取值相同。
14. **其余未开始的项。** 选定 Embedding 与重排模型的规格（AC-5-4）、计量对账（AC-12-4）、部署运维。

## 需要用户决定或提供的事项

- 模型服务（2026-11-13 确定）：生成用 DeepSeek（`deepseek-flash`）；Embedding 与重排由用户选定本地部署 Qwen3（`modelserver/`，先用 0.6B）。是否换 4B/8B 可在合成集 v2 上按 AC-5-4 比较后决定。
- 中文分词（AC-5-4，2026-11-13 确定）：用户依据合成集 v2 的对比选定 standard（按字切分）。
- 合成集 v2 的人工抽检（14.2 第 6 步）：`docs/评测集/合成集-v2-抽检单.xlsx` 100 条。
- 确认“删除待审核版本时保留审核记录”。
- 集群版私有化环境的 S3 兼容存储产品（第 13 周前）。MinIO 社区版已停止维护，候选为 Garage、SeaweedFS、Ceph RGW。
- AC-5-6 以哪种负载验收：持续负载（每秒 1 份）远低于 60 秒；突发审核的延迟取决于 Embedding 吞吐（本机 0.6B 约 200 份到 60 秒）。需要决定突发规模，或把突发容量写进机器规格对照表（AC-13-6）。
- 各指标项的门槛，在基线评测后与用户一起回填到开发文档 16.2 节；准入门槛的初值（误通过率 ≤ 5%、误拒率 ≤ 30%、Deep 正确率 ≥ 70%）待确认。
