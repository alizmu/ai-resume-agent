# 项目交接简报 · AI简历助手-public

> 给接手本项目的 WorkBuddy Agent 看的第一份文件。动手前先读完本文件 + `README.md`。

## 一句话定位

面向求职场景的 RAG Agent 框架：把简历变成「可对话、可溯源、可量化」的 AI 助手。

技术栈：FastAPI + LangGraph(StateGraph) + Qdrant(embedded) + bge-small-zh-v1.5 稠密 + BM25 稀疏 + RRF 融合 + 可选 bge-reranker(默认关) + DeepSeek LLM + SSE 流式；前端零构建原生 JS 单页。

架构/换人方式/本地运行见 `README.md`（那份很全，先看它）。

## 当前线上状态（截至 2026-09-08）

- 线上链接：在「设置—数据管理—我发布的应用」中查看，或直接取 `workbuddy_sites_deploy` 返回的 `app.workbuddy.link` 地址。
  - ⚠️ 给用户只用 `app.workbuddy.link`。不要给 `*.sandbox.cloudstudio.club` 那种沙箱直连地址——平台会自动休眠，返回 403「工作空间已停止」。
- 已部署目录：`deploy/online/`（= backend + frontend + knowledge + 离线模型 `.model_cache`），**该目录已在 `.gitignore` 中，不进仓库**。
- 知识库 `doc_count=57`，线上是**虚构演示数据**（候选人「张明」），与仓库内 `knowledge/` 一致。
- LLM 已接入 DeepSeek（`llm=configured`）。`backend/core/llm.py` 带超时降级：调用失败时自动退化为规则生成，不会 502。

### 换成真实资料

网页端「更换资料」入口可自助上传，也可以调 `POST /api/bootstrap`。上传后会自动刷新发布清单并重建索引。

⚠️ 线上链接是公开可访问的，上传真实个人资料前请确认是否可接受；默认的 `REINDEX_TOKEN` 较弱，正式使用前建议更换。

## 关键决策（务必知道，别踩反）

1. **两层 PII 架构**：代码零硬编码个人信息，所有「服务谁」外置为数据——`knowledge/persona.json` + `knowledge/raw/`。换人 = 替换 `knowledge/` 并重新入库，框架零改动。
2. **仓库只放虚构演示数据**：根目录 `knowledge/` 是虚构的「张明」，可安全推公开仓库。真实资料只放在部署包 `deploy/online/`（不进 git）或用户自行上传。
3. **发布机制**：用 `workbuddy_sites_deploy` 部署 `deploy/online/`，必须显式指定 `startCmd="python start.py"`（自动探测会误找 `main.py`）。`.env` 需配 `HF_OFFLINE=true`（模型随包，禁联网）+ `REINDEX_TOKEN`（公开站点必须设，否则任何人可重建知识库）。

## ⚠️ 必踩的坑（红线）

- **改了知识库后必须 `POST /api/reindex`**（请求头 `x-reindex-token: <REINDEX_TOKEN>`）才会重建索引。平台复用持久卷，只改文件不会自动重建——这是最大坑。
- **部署沙箱复用持久卷，重新发布不会删除旧文件**。删掉的文件靠 `knowledge/manifest.txt` 发布清单过滤（`load_documents()` 只加载清单内的文件）；新增/修改文件后必须手动 reindex。
- **reindex 是同步阻塞调用**，会把平台存活探针拖超时 → 触发容器重启 / 临时 502。结束后若链接 502，等一两分钟或重新发布即可，索引已落盘在卷里。
- 给用户链接用 `app.workbuddy.link`，**绝不用** `cloudstudio.club` 沙箱直连（会被回收休眠）。
- 不要把 `deploy/online/.env`、个人联系方式等 PII 提交到仓库。

## 待办（用户可能让你继续）

1. 可选优化：开启 reranker（`enable_rerank=True`，需打包 2.2G 模型，慎重）。
2. 评测集目前 14 条，可扩充到 30 条并同步更新 `README.md` 的离线评测表。

## 给新 Agent 的启动关键词（可整段粘贴到新会话）

> 你是接手「AI简历助手-public」项目的工程师。先读 `HANDOFF.md` 和 `README.md` 了解全貌再动手。这是一个面向求职的 RAG 简历助手（FastAPI + LangGraph + Qdrant + bge-small + BM25 + RRF + DeepSeek），仓库里的 `knowledge/` 是虚构「张明」演示数据。红线：① 改知识库后必须 `POST /api/reindex`（header `x-reindex-token`）重建索引，平台复用持久卷不会自动重建；② 给用户链接用 `app.workbuddy.link`，绝不用 `cloudstudio.club` 沙箱直连；③ `deploy/online/` 不进仓库，不要把 `.env` 和 PII 提交上去。
