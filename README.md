# document.ai · EAER

可溯源、带时效判断的企业知识库问答。

用户不仅得到答案，还能看到答案来自哪份文档的哪个版本、哪一处，以及这条依据在问题关心的那一天是否有效。制度会修订、会作废、会被补充通知覆盖，“查到一段相似的话”不等于“答对了”；EAER 把版本、有效期、权限和验证当作问答本身的一部分来做。

> document.ai 2023 年的版本（向量数据库 + GPT-3.5 的本地知识库方案与 MSD 医疗问答示例）保留在 [`legacy/`](legacy/)，对应的提交打了标签 `msd-demo-2023`。

## 能做什么

- **三种执行模式。** Fast 一次检索直接作答；Deep 按计划模板（比较、多跳、因果）分步检索；Verified 在作答后做语义检查与定向补检索。Router 按问题选入口，证据不足时逐级升级，预算用尽仍无法确认就拒答。
- **验证层。** 代码做确定性检查：引用的证据是不是本次检索到的、用户有没有权限、定位与原文是否对得上、数值与日期是否出现在证据里、证据在目标时间是否有效；模型只判断证据是否真正支持结论。拿不准的标“未验证”，不支持的删除，核心结论不过则整题拒答并说明缺少什么。
- **三类时间查询。** 当前生效；某天实际生效（按有效期）；截至某天系统所知（回溯，按审核通过时间，有效期取当时记录的值）。生效日的来源分正文写明、推断、未知三级，如实标注。
- **文档生命周期。** 上传 → 审核（逐份或批量，可补录更正）→ 可搜索 → 归档 / 取消归档。新版本的生效日推出旧版本的作废日；有效期与版本关系只追加，回溯可以复现。归档后的提问一律查不到，历史回答保留引用记录。
- **权限由代码执行。** 每次检索都带授权范围，返回前在数据库里再核对状态、权限与有效期；无权访问与不存在一律返回 404，拒答文本不透露无权文档的存在。授权范围只以句柄传递，不进入模型上下文。
- **严格模式与宽松模式。** 默认宽松：可以按文件里的单价与问题给的数量写出算式算合计（代码核对算术），可以按常识判断问题中的对象属于文件里的哪一类（单列一条、标“常识推断”），只找到一部分时先答能答的部分。严格模式只认文件原文。两种模式下制度的规定与取值都只能出自文件。按租户设置，提问可按次覆盖。
- **多轮追问。** 追问结合会话上文改写成完整的问题再检索与回答；上文只取之前提问的原话与仍有权查看的引用文档标题。
- **解析。** Word、HTML 用原生解析器；PDF、扫描件、PPTX、XLSX、图片经 Docling（含 OCR）；表格按多层表头拆成带表题的单元格证据；图表可交给支持看图的模型读出数据点。
- **运维。** 超过同步时限的提问转入后台、可查进度与取消、进程中断后从检查点恢复；索引任务失败退避重试，每小时对账补写；每次提问一条 OpenTelemetry trace（原文不进 trace）；回答缓存按租户与授权范围隔离，权限或状态变化时由数据库触发器清除。
- **提示注入防护。** 证据放进带随机边界的证据区并转义，系统提示声明区内指令不执行；指令型文字入库时提醒审核人，被引用时转入 Verified。

## 架构

```
            ┌──────────── 对外接口 eaer/api（FastAPI，OpenAPI 3.1）────────────┐
 上传/审核 → │  入库 eaer/ingestion ──→ 索引任务（唯一写索引）──→ OpenSearch      │
            │        │ 生命周期 eaer/lifecycle（唯一改版本状态）                 │
 提问      → │  编排 eaer/orchestrator（LangGraph）：时间点 → Router → Fast/Deep  │
            │        → 升级判断 → Verified → 组装          检索 eaer/retrieval    │
            │  验证 eaer/verifier：确定性检查 + 语义检查                         │
            └──────── PostgreSQL：版本、有效期、关系、权限、审计、任务、检查点 ─────┘
     对接接口 eaer/adapters：数据源、授权、模型（参考实现 + 一致性套件）
```

| 路径 | 内容 |
| --- | --- |
| `eaer/` | 核心引擎：数据库迁移、生命周期、入库、审核、检索、验证、编排、接口、对接接口、可观测 |
| `modelserver/` | 本地模型服务（参考实现）：Qwen3-Embedding 与 Qwen3-Reranker，提供 `/v1/embeddings` 与 `/v1/rerank`，带内存预算与看门狗 |
| `web/` | 简易 Web UI（Next.js + shadcn/ui）：对话、证据与全文、上传、审核、归档、配置、身份切换，见 [`web/README.md`](web/README.md) |
| `factset/` | 合成评测集工具：事实表 → 文档 → 标准答案 |
| `tests/` | 每个测试文件的文档字符串注明对应的验收项（AC-x-y） |
| `deploy/`、`Dockerfile`、`docker-compose.yml` | 精简版部署：一个编排文件起全部组件，配置模板，备份与恢复脚本 |
| `docs/` | 开发文档、设计变更记录、部署说明、评测集与评测报告，见下文 |

设计上的几条硬规则：只有检索模块读索引，只有入库与生命周期模块写索引，只有生命周期模块改版本状态；所有模型调用经模型接口，业务代码不依赖模型厂商的 SDK；不变量放在数据库（触发器、检查约束、部分唯一索引），服务层先给出可读的错误，触发器是第二道防线。

## 快速开始

需要 Python 3.14、Docker、Node.js 20.9 以上（Web UI），以及一个 OpenAI 兼容的生成服务（如 DeepSeek）。

1. 启动依赖（PostgreSQL 18.6、OpenSearch 3.9.0）：

   ```bash
   docker compose -f docker-compose.dev.yml up -d
   ```

2. 安装依赖。要解析 PDF、扫描件、PPTX、XLSX、图片时改装 `requirements-docling.txt`（含 PyTorch）：

   ```bash
   python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```

3. 配置模型服务，写在仓库根目录的 `.env`（已在 `.gitignore` 中，不要提交）。生成、Embedding、重排分别配置，模板是 [`deploy/eaer.env.example`](deploy/eaer.env.example)，变量的完整说明见 [`eaer/adapters/online.py`](eaer/adapters/online.py)。用 DeepSeek 生成、本地 Qwen3 做 Embedding 与重排时大致是：

   ```bash
   EAER_GEN_BASE_URL=https://api.deepseek.com
   EAER_GEN_API_KEY=你的密钥
   EAER_GEN_MODEL=deepseek-flash
   EAER_GEN_TEMPERATURE=0
   EAER_GEN_VISION=1                      # 生成模型能看图时设 1，图表交给它读出数据点
   EAER_EMBED_BASE_URL=http://127.0.0.1:8100/v1
   EAER_EMBED_MODEL=Qwen3-Embedding-0.6B
   EAER_EMBED_DIMS=1024
   EAER_EMBED_QUERY_INSTRUCTION="Given a question about company policies, retrieve passages that answer the question"
   EAER_RERANK_BASE_URL=http://127.0.0.1:8100/v1
   EAER_RERANK_MODEL=Qwen3-Reranker-0.6B
   ```

   Embedding 与重排可以用本地模型服务（Mac 用 MPS、Linux 用 CUDA，首次运行下载模型，参数见 [`modelserver/__main__.py`](modelserver/__main__.py)）：

   ```bash
   .venv/bin/python -m modelserver --port 8100 --env-file .env
   ```

4. 建库、建索引并启动接口进程与后台任务进程：

   ```bash
   export EAER_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/eaer
   export EAER_STORAGE_DIR=$PWD/files
   export EAER_TOKENS_FILE=$PWD/deploy/tokens.json     # 参考实现的令牌表，格式见 deploy/tokens.example.json
   export EAER_OPENSEARCH_URL=http://localhost:9200
   export EAER_ADMISSION=off                           # 本地开发：不做准入检查，Deep 与 Verified 直接开放
   set -a; . ./.env; set +a
   .venv/bin/alembic upgrade head
   .venv/bin/python -m eaer.ingestion rebuild --build-id b1
   .venv/bin/uvicorn eaer.api.main:app --port 8000 &
   .venv/bin/python -m eaer.ingestion &
   ```

5. 打开 Web UI：

   ```bash
   cd web && npm ci && npm run dev     # http://localhost:3200
   ```

正式部署（含准入检查、旧索引定时清理、备份恢复与冒烟测试）见 [`docs/部署说明.md`](docs/部署说明.md)。生产环境的准入检查不达标时会自动关闭 Deep 或 Verified；文档内容会发送到所配置的线上生成服务，部署时须向使用方明示。

## 运行测试

```bash
docker compose -f docker-compose.dev.yml up -d
export EAER_TEST_ADMIN_URL=postgresql://postgres:postgres@localhost:5432/postgres
export EAER_TEST_OPENSEARCH_URL=http://localhost:9200     # 不设置则跳过 OpenSearch 集成测试
.venv/bin/python -m pytest -q
```

每个数据库测试使用从已迁移模板复制出的独立数据库；连不上数据库时数据库测试跳过，其余照常运行。需要模型服务的少数用例在没配置时跳过。CI 配置在 `.github/workflows/ci.yml`。

## 评测

公开评测集只用来发现缺陷，不针对某个集合的问法或分数调规则与阈值；合成数据与小样本上的数字只说明方向，不是产品基线。下列结果用 DeepSeek 生成、本地 Qwen3-0.6B 做 Embedding 与重排：

| 评测 | 结果 |
| --- | --- |
| 三方对比（合成集 v2，9 类各 5 问，[报告](docs/三方对比报告.md)） | 传统 Hybrid RAG 31/45；Always-Agent 43/45，P50 8.9 秒、每问 3925 token；Adaptive（EAER）43/45，P50 6.8 秒、3264 token；与传统做法的差距来自历史时点、回溯、版本差异三类 |
| 合成集 v2 测试集，每类 10 问 | 96–97/100；应拒答的全部拒答，越权 0 |
| MultiHop-RAG（609 篇英文新闻） | 检索 Recall@10 按依据：混合检索 70.1%，加重排 76.7%；无答案的 50 问全部拒答 |
| LLM-AggreFact（语义检查原样评测） | 均衡准确率 77.7%，误通过 31.8%（语义检查偏宽，调阈值无效） |
| ViDoRe V3 HR 子集（图表与表格 60 问） | 答对 30%、部分答对 27%、拒答 38% |

评测工具在 `eaer/evalrun/`（只在 `EAER_ENV=eval` 时可用），每份报告带数据集许可标记，生成对外报告时按许可拦截。完整记录与已知局限见 [`CLAUDE.md`](CLAUDE.md) 的“当前进度”。

## 文档

| 文件 | 内容 |
| --- | --- |
| [`docs/开发文档-v1.0.md`](docs/开发文档-v1.0.md) | 开发与验收的依据：需求、架构、验收项（AC）与硬性项 |
| [`docs/设计变更记录.md`](docs/设计变更记录.md) | 相对开发文档的每一处设计改动及其原因 |
| [`docs/部署说明.md`](docs/部署说明.md) | 精简版部署、备份恢复与演练记录 |
| [`docs/机器规格对照表.md`](docs/机器规格对照表.md) | 入库吞吐与机器规格 |
| [`docs/能力矩阵-线上模型.md`](docs/能力矩阵-线上模型.md) | 准入检查的结果 |
| [`docs/三方对比报告.md`](docs/三方对比报告.md) | 传统 Hybrid RAG、Always-Agent、Adaptive 的对比 |
| [`docs/评测集/`](docs/评测集/) | 合成评测集与抽检单 |
| [`eaer/api/openapi.yaml`](eaer/api/openapi.yaml) | 对外接口的唯一定义（OpenAPI 3.1） |

## 现状与局限

精简版（单机，PostgreSQL + OpenSearch + 线上生成模型）已实现并演练过部署、备份恢复与横向扩展；集群版（Temporal 任务编排、S3 兼容存储、PostgreSQL 主从与 WAL 持续归档）尚未实现。解析器、分类、时间抽取只在合成语料与公开评测集上验证过，真实客户文档的人工核对还没有做。语义检查在 LLM-AggreFact 上偏宽。各项进度与待决定事项见 [`CLAUDE.md`](CLAUDE.md)。

## 参与开发

仓库根目录的 [`CLAUDE.md`](CLAUDE.md) 写着开发约定、目录说明、关键规则和当前进度，用 Claude Code 开发时会先读取它。几条主要约定：文档、注释、错误信息与提交说明用中文，标识符用英文；依赖精确锁定；迁移只追加、可回滚；开发文档第 16.1 节的硬性项测试失败就修实现、不放宽测试；相对开发文档的设计改动记入设计变更记录。

## 许可证

[AGPL-3.0](LICENSE)。

## Buy me a coffee

<div align="center">
<a href="https://www.buymeacoffee.com/ganymedenil" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
</div>
<div align="center">
<img height="360" src="https://user-images.githubusercontent.com/9687786/224522468-eafb7042-d000-4799-9d16-450489e8efa4.png"/>
<img height="360" src="https://user-images.githubusercontent.com/9687786/224522477-46f3e80b-0733-4be9-a829-37928260038c.png"/>
</div>
