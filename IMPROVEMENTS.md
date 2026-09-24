# AI 简历助手 · 可改进与可迭代项清单

> 基于 2026-09-07 功能测试与代码走读整理。按「影响 / 紧急度」分层，方便你按阶段推进。

---

## 一、高优先级（建议近期处理）

### 1.1 文档诚信风险（已发现，待你决策）

| 项 | 问题 | 建议 |
|---|---|---|
| **README 量化数据** | 宣称 30 条 query、R@1 64.3%、MRR +0.083；实测 14 条、R@1 75.0%、MRR −0.011 | 补齐到 30 条后重跑更新；或改成 14 条的真实数字并注明样本量 |
| **HANDOFF.md 事实错误** | 写「根目录 knowledge/ 是真实未脱敏数据」，实测为虚构「张明」演示数据 | 更新 HANDOFF，避免误导不敢推 GitHub |

**风险**：求职项目里，面试官最可能复现核对的就是 README 数据表；HANDOFF 错误则会让协作人误判。

---

### 1.2 Docker 构建链（已验证可用，但根目录缺 `pyproject.toml`）

**状态更新**：经复核，`backend/requirements.txt` 已存在且已跟踪，Dockerfile 引用路径正确，`docker compose -f deploy/docker-compose.yml up --build` 理论上可构建。此前报告为误判。

**仍建议**：
- 根目录补一个 `pyproject.toml`，把项目做成可 `pip install -e .` 的包，免去手动设置 `PYTHONPATH`
- 把 `deploy/online/backend/requirements.txt` 纳入 `sync_deploy.py` 的同步产物，确保部署副本与根目录一致

---

### 1.3 `deploy/online` 双份代码副本已脱同步

**问题**：
- 根目录 `backend/` 与 `deploy/online/backend/` 同时存在
- 当前 `deploy/online/backend/rag/retriever.py`、`vectorstore.py`、`scripts/bootstrap.py` 与根目录**不一致**（未包含本次修复）
- `deploy/online/frontend/`、`deploy/online/knowledge/` 也可能与根目录不同步

**风险**：真正上线时如果打的是 `deploy/online` 的包，会把旧 bug（P0 锁冲突等）重新带上去。

**修复**：
1. 立即执行一次 `python -m backend.scripts.sync_deploy`
2. 在 CI / 发布脚本里强制要求：发布前必须先跑 `sync_deploy`，并校验 diff 为空
3. 长期更优方案：取消 `deploy/online` 里的源码副本，Docker 构建时直接用根目录作为 `context`，只把 `.model_cache`、`.env`、`start.py` 放在部署目录

---

### 1.4 根目录缺少依赖文件（本地 setup 不完整）

**问题**：README 写「准备虚拟环境（依赖：fastapi uvicorn langgraph…）」，但根目录没有 `requirements.txt` 或 `pyproject.toml`，新人只能自己拼依赖。

**修复**：
- 新增根目录 `requirements.txt`（与 `deploy/online/backend/requirements.txt` 内容一致）
- 更好：新增 `pyproject.toml`，把项目做成可 `pip install -e .` 的包，解决 `PYTHONPATH` 手动设置问题

---

### 1.5 Query Rewriter 模块级常量无法热更新

**问题**：
- `backend/agent/query_rewrite.py:25-30` 在模块导入时读取 `settings.subject_name` 和 `get_persona()`
- 执行 `/api/bootstrap` 换人后，虽然 `persona.json` 已更新、`reload_persona()` 已调用，但 **QueryRewriter 里的 `SUBJECT`、`KNOWN_PROJECTS`、`PROJECT_KEYWORDS` 仍是旧值**
- 结果：换人后多轮对话的指代消解还是用旧名字 / 旧项目白名单

**修复**：
- 把常量改为函数内读取，或在 `reload_persona()` 里重新加载这些模块级变量
- 更简单：`QueryRewriter.__init__` 里动态读 `get_persona()`，不要依赖模块导入时的快照

---

## 二、中优先级（质量与体验）

### 2.1 记忆持久化与架构描述不符

**问题**：
- README 写「记忆：滑动窗口 + 摘要（多轮上下文管理）」「PostgreSQL/Redis」
- 实际实现：`backend/agent/memory.py` 用内存字典 + 文件（`knowledge/memory/long_term_{session_id}.json`）
- 会话一多会写大量小文件；服务重启后短期记忆丢失；无 Redis/PostgreSQL 接入

**修复**：
- 保留内存缓存，但把长期记忆/会话历史写入 Redis（轻量）或 PostgreSQL（需要复杂查询时）
- 如果暂时不改实现，README 应降低描述，避免面试时被追问「你的 Redis 在哪」

---

### 2.2 LLM 客户端缺乏容错

**问题**：`backend/core/llm.py` 的 `OpenAICompatibleClient`：
- 无重试机制
- 无超时配置（依赖 openai 默认）
- 流式出错只打一个 error event，不会自动降级到 DummyLLMClient
- 未处理 API key 无效 / 余额不足 / 模型不存在等具体错误

**修复**：
- 加 `tenacity` 重试（指数退避，只重试 429/5xx，不重试 4xx）
- 配置 `timeout` 与 `max_retries`
- 真实 LLM 连续失败时， gracefully 降级到规则生成，并告知用户「当前走演示模式」

---

### 2.3 管理接口缺少限流与防并发滥用

**问题**：
- `/api/reindex` 和 `/api/bootstrap` 虽有 token 鉴权和互斥锁，但无请求频率限制
- 反复调用会反复重建索引、消耗 embedding 资源
- 上传接口虽有 10MB 单文件限制，但无总大小限制、无请求总数限制

**修复**：
- 管理接口加基于 IP / token 的速率限制（如每 5 分钟 3 次）
- 上传总大小限制
- 可在 nginx 层做限流，更轻量

---

### 2.4 健康检查不够深

**问题**：`/api/health` 只检查 `VectorStore.count()`，不验证：
- Qdrant 是否真的可读写（嵌入式模式下可能锁冲突）
- embedding 服务是否可推理
- LLM 是否可达（如果配置了 key）

**修复**：
- health 增加轻量探测：尝试对固定 query 做 embed_dense_query
- 配置 LLM key 时，可选做一次 cheap 调用探测（注意成本和延迟）
- 返回更细的 status 字段：`qdrant_ok`、`embedding_ok`、`llm_ok`

---

### 2.5 文件上传校验可加强

**问题**：
- 只校验扩展名，不校验文件内容 / MIME / 魔数
- `.md` 可以随便改扩展名上传
- 无内容安全扫描（简历项目里文件来自用户自己，风险低，但上线后要考虑）

**修复**：
- 对 PDF/docx 用库读取时失败即拒绝
- 校验 MIME type（UploadFile 提供 `content_type`）
- 文本文件大小写敏感扩展名统一转小写（已实现，可保持）

---

### 2.6 前端健壮性

**问题**（基于 721 行 `frontend/app.js` 走读）：
- SSE 断线无自动重连
- 错误提示依赖后端返回的字符串，无错误码映射
- 没有 loading 超时提示
- 「更换资料」弹层提交成功后没有明确反馈
- 移动端适配未验证

**修复**：
- SSE 加 `onerror` 重连（最多 3 次）
- 统一错误提示组件
- 提交 bootstrap 后显示「入库中…」并轮询 health 直到 doc_count 更新

---

### 2.7 切片与检索可迭代

**问题**：
- 当前切片策略较简单：按标题 + 硬切 + 重叠
- 未针对简历/项目文档做语义切分
- 未评估不同 chunk_size 对指标的影响
- `eval/run_eval.py` 只测 R@K/MRR，不测生成质量

**修复**：
- 加 ablation：对比 `chunk_size=400/600/800` 的指标
- 引入语义切分（如按段落主题模型）或 overlap 策略优化
- 评测集增加生成质量评分（需要 LLM-as-judge 或人工）

---

## 三、低优先级 / 长期建设

### 3.1 可观测性

- 无 Prometheus / metrics
- 无分布式 tracing
- 日志是 plaintext，未结构化 JSON
- 无法观测每次请求的 retrieve latency、LLM latency、token 数

**建议**：
- 加 `prometheus-fastapi-instrumentator`
- 关键路径打结构化 log：query、rewritten、recall 数、reliable、latency

---

### 3.2 自动化测试与 CI

- 目前只有 `.test/run_tests.py` 这个集成测试脚本
- 无单元测试（chunker、rewriter、retriever 都可单测）
- 无 CI 跑测试 / 构建 Docker / 校验 `sync_deploy`

**建议**：
- 用 `pytest` 写单元测试，覆盖 chunker、rewriter、retriever
- GitHub Actions：lint → test → build Docker → sync_deploy diff check

---

### 3.3 代码组织

- `backend/` 与 `deploy/online/backend/` 重复（已在上文）
- `backend/scripts/` 里混了 CLI 工具、Web 共用逻辑、部署同步脚本，边界可更清晰
- 可考虑把 `bootstrap.py` 拆成：IO/落盘层、persona 生成层、入库编排层

---

### 3.4 安全加固

- CORS `allow_origins` 包含 `http://localhost:5173`，生产环境应收紧
- 未配置 HTTPS 强制 / HSTS（nginx 层可处理）
- 无 API 请求签名 / nonce（管理 token 已够用，但上线前建议轮换机制）

---

## 四、推荐推进顺序

如果你是 26 届应届生、项目还要往简历上写，建议按这个顺序：

1. **立刻修**：README 数据对齐 + HANDOFF 修正 + Docker requirements.txt + 跑 `sync_deploy`
2. **本周修**：QueryRewriter 热更新 + 健康检查深化 + LLM 重试与降级
3. **面试前修**：管理接口限流 + 前端 SSE 重连 + 单元测试 + CI
4. **有余力**：PostgreSQL/Redis 记忆持久化 + 可观测性 + 语义切片 ablation

---

## 五、新增发现 vs 已修复项

| 状态 | 项 |
|---|---|
| ✅ 已修复 | P0 锁冲突、P1 拒答失效、P1 bootstrap 原子性、P1 评测降级告警、Windows 覆盖重建失效、HANDOFF 事实错误、deploy/online 脱同步、QueryRewriter 热更新 |
| ⚠️ 待决策 | README 量化数据 |
| 🔧 新增高优（待处理） | pyproject.toml 已补；Docker 构建链经验证可用 |
| 💡 建议迭代 | 记忆持久化、LLM 容错、限流、健康检查、前端健壮性、可观测性、CI |
