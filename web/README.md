# EAER Web UI（简易演示版）

用浏览器走通 EAER 的对外接口：对话式提问（Fast / Deep / Verified / 自动；回答方式可选租户默认、宽松或严格，结论标出“计算”“常识推断”，回答前展示思考过程，转入后台时显示进度、可以取消）、在抽屉里查看引用的证据与高亮的全文、下载原件、上传、审核（逐份与批量、补录更正、阅读待审核文档全文、已删除记录）、归档与取消归档、强制验证清单与租户默认的回答方式，以及切换身份对比权限差异。

- Next.js 16（App Router）+ TypeScript，界面统一用 [shadcn/ui](https://ui.shadcn.com)（Radix 版、Tailwind CSS v4），组件源码在 `components/ui/`。
- 接口以 `eaer/api/openapi.yaml` 为唯一依据，数据结构见 `lib/types.ts`。
- 浏览器只访问本服务的 `/eaer-api/*`，由 `next.config.ts` 的 rewrites 代理到接口进程，后端不需要开 CORS。
- 只是演示界面：令牌保存在浏览器的 localStorage，不做登录；客户现场由 SSO 提供身份。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `EAER_API_URL` | `http://127.0.0.1:8000` | 接口进程地址。rewrites 在 `next build` 时写进构建产物，生产构建要在 build 之前设置 |

前端没有别的配置；令牌在页面上导入（见下文“身份”）。

## 启动

需要 Node.js 20.9 以上（本机验证用 Node 24）。

```bash
cd web
npm ci
npm run dev          # http://localhost:3200
```

生产方式：

```bash
cd web
EAER_API_URL=http://127.0.0.1:8000 npm run build
npm run start        # http://localhost:3200
```

在 Claude Code 桌面版里，`.claude/launch.json` 已有名为 `web` 的启动配置（`npm --prefix web run dev`，端口 3200）。

## 准备后端

前端只是接口的客户端，先按仓库根目录的 `CLAUDE.md` 起好依赖与接口进程。下面是一套与开发库隔开的演示环境（独立的数据库 `eaer_webui` 与索引别名 `eaer_webui`），在仓库根目录执行：

```bash
docker compose -f docker-compose.dev.yml up -d
docker exec eaer-postgres-1 psql -U postgres -c "CREATE DATABASE eaer_webui"
```

```bash
EAER_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/eaer_webui .venv/bin/python -m eaer.db upgrade
```

本地模型服务（Embedding 与重排；生成用 `.env` 里配置的线上模型）：

```bash
.venv/bin/python -m modelserver --port 8100 --env-file .env
```

令牌表（参考身份实现，格式同 `deploy/tokens.example.json`）。演示权限差异建议至少有这几种身份，把键换成随机令牌，文件不要提交：

```json
{
  "换成随机令牌-1": {"user": "admin", "tenant": "t1", "roles": ["editor", "reviewer", "admin"], "principals": ["R_STAFF", "R_FIN"]},
  "换成随机令牌-2": {"user": "editor1", "tenant": "t1", "roles": ["editor"]},
  "换成随机令牌-3": {"user": "reviewer1", "tenant": "t1", "roles": ["reviewer"]},
  "换成随机令牌-4": {"user": "staff1", "tenant": "t1", "roles": [], "principals": ["R_STAFF"]},
  "换成随机令牌-5": {"user": "fin1", "tenant": "t1", "roles": [], "principals": ["R_FIN"]},
  "换成随机令牌-6": {"user": "eve", "tenant": "t2", "roles": ["editor", "reviewer"], "principals": ["R_STAFF", "R_FIN"]}
}
```

生成随机令牌：`python3 -c "import secrets; print(secrets.token_urlsafe(18))"`。`roles` 是接口角色（editor 上传与归档、reviewer 审核、admin 追加强制验证规则），`principals` 是文档权限角色；没有 `principals` 的账号无法确定授权范围，提问与证据接口返回 403。

接口进程与后台任务进程用同一组环境变量（模型配置从 `.env` 读入，不要打印）。把下面几行存成一个脚本（如 `webui-env.sh`，放在仓库外或不提交），`TOKENS` 与 `FILES` 换成你的路径：

```bash
#!/bin/sh
cd /path/to/eaer
set -a; . ./.env; set +a
export EAER_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/eaer_webui
export EAER_STORAGE_DIR=/path/to/FILES
export EAER_TOKENS_FILE=/path/to/TOKENS.json
export EAER_OPENSEARCH_URL=http://localhost:9200
export EAER_INDEX_ALIAS=eaer_webui
export EAER_ADMISSION=off          # 只用于本地开发：没有准入记录时也开放 Deep 与 Verified
export EAER_GEN_THINKING=enabled   # 可选：开启生成模型的思考，对话里能看到模型自身的思考（更慢、更费 token）
export EAER_POLL_SECONDS=1
exec "$@"
```

```bash
./webui-env.sh .venv/bin/python -m eaer.ingestion ensure-index
```

```bash
./webui-env.sh .venv/bin/uvicorn eaer.api.main:app --host 127.0.0.1 --port 8000
```

```bash
./webui-env.sh .venv/bin/python -m eaer.ingestion
```

## 页面与接口

| 页面 | 接口 | 需要的角色 |
| --- | --- | --- |
| 对话 `/` | `POST /v1/query`（可选 SSE 流式）、`GET /v1/sessions/{id}`、`POST /v1/sessions/{id}/cancel`。左侧是会话，右侧一问一答；回答前的“思考过程”（shadcn Reasoning / Trace）：模型自身的思考按步骤各一行（接口的 `reasoning`，生成服务开启思考时才有），最后一行是经过的模式、后台执行的步骤、时间点怎么定的、核对结果；点结论后的编号、引用的“证据”“全文”，在右侧抽屉里看证据与高亮的全文 | 有文档权限角色的任何身份；会话只属于发起人 |
| 版本全文 `/versions/{id}?ev=证据 id` | `GET /v1/versions/{id}`（全部证据按页、段落、表格、图表排列，高亮被引的一处）、`GET /v1/versions/{id}/original`（下载原件，一律作为附件） | 有权的任何身份；待审核的版本只对本租户审核人开放 |
| 上传 `/upload` | `POST /v1/documents` | editor |
| 审核 `/review` | `GET /v1/review/tasks`（待审核、已通过、已删除，按批次筛选，游标分页）、`POST /v1/review/decisions`（逐份或批量通过、删除，补录文档类型、有效期、关系）；每份任务有“全文”链接 | reviewer |
| 文档 `/documents` | `POST /v1/documents/{id}/archive`、`/unarchive` | editor |
| 证据 `/evidence`、`/evidence/{id}` | `GET /v1/evidence/{id}`（原文、定位、有效期、文档日期与出处；已归档只给提示）；可用全部身份各取一次对比状态码 | 有文档权限角色的任何身份 |
| 强制验证 `/config` | `GET`、`POST /v1/config/forced-verify`（只能追加） | 读：任何身份；追加：admin |
| 身份 `/identities` | 导入令牌表或逐个添加令牌，检查令牌是否有效 | — |

`/sessions`、`/sessions/{id}` 已并入对话页（跳到 `/?session={id}`）。

接口没有“列出会话”“列出文档”的操作，所以：

- 对话页左侧的会话是当前身份在本浏览器里提问过的会话；也可以输入会话 id 打开。
- 文档列表合并已通过的审核任务（需要 reviewer 角色才能读）与本浏览器上传过的文档；也可以输入文档 id 操作。

## 演示流程

1. 身份页粘贴令牌表 → 导入 → “检查全部令牌”。
2. 切到 editor1，上传一份 HTML 制度（`principals` 填 `R_STAFF`）。
3. 切到 reviewer1，审核页点“全文”读待审核的内容（表格与图表在审核通过后的深度解析里才抽取，可下载原件对照），等“已完成”抽取后通过（可展开“更正”补录生效日等）；删除会先弹确认，删除后在“已删除”里只剩删除人与时间。
4. 切到 staff1，在对话里提问：思考过程先展开（思考中），答完收起为“已思考 · 用时 N 秒”；每条结论标“已验证 / 未验证”，后面的编号与引用里的“证据”“全文”在右侧抽屉里打开，全文高亮被引的那一处。证据页的“按身份对比”可见 fin1（只有 R_FIN）与其他租户都是 404。
5. 切到 editor1 在文档页归档 → staff1 再问，拒答；旧回答仍保留引用，点开只给归档提示，全文与原件也不再给。
6. 取消归档 → 再问，又能引用。

转入后台与取消：同步时限默认 30 秒，演示时可临时调小演示租户的 `budgets.sync_seconds`（仅演示库）：

```bash
docker exec eaer-postgres-1 psql -U postgres -d eaer_webui -c "UPDATE tenant_config SET budgets = budgets || '{\"sync_seconds\": 1}' WHERE tenant_id = 't1'"
```

演示完恢复默认值：

```bash
docker exec eaer-postgres-1 psql -U postgres -d eaer_webui -c "UPDATE tenant_config SET budgets = budgets - 'sync_seconds' WHERE tenant_id = 't1'"
```

## 依赖与 shadcn/ui

- 版本精确锁定（`.npmrc` 里 `save-exact=true`），选发布满 14 天的最新稳定版；传递依赖也按同一规则，用 `npm install --before=<14 天前的日期>` 生成 `package-lock.json`。
- shadcn 的 CLI 不作为依赖安装，加组件时临时调用固定版本：`npx shadcn@4.23.0 add <组件>`。CLI 会顺带装最新版的依赖、并把组件里的 `cn` 写成从 `cn` 包导入：加完后把 `from "cn"` 改成 `from "@/lib/utils"`，卸掉 `cn` 包，再按上一条重新锁定版本。
- `app/shadcn.css` 是 shadcn@4.23.0 的 `dist/tailwind.css`（组件用到的动画、`shimmer` 与自定义变体），复制进来以免把 CLI 装成运行依赖。
- 用到的 shadcn 组件在 `components/ui/`，其中 `trace`（Reasoning 样式）与 `drawer`（依赖 vaul 1.1.2）用于对话页。
- 模型的思考按 Markdown 显示：react-markdown 10.1.0 加 remark-gfm 4.0.1（`components/Markdown.tsx`），不渲染原始 HTML（思考里可能引用证据原文）。
- 字体用系统字体，构建时不从 Google 下载，私有化环境也能构建。
