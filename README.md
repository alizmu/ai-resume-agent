# RAG 简历助手 · 可复用的 AI 简历 Agent 框架（演示版）

一个面向「AI 应用开发 / Agent 工程师」求职场景的 RAG Agent 框架：让面试官通过对话了解候选人，底层是一套**完整、可评测的 RAG 检索增强系统**，对话界面只是展示层。

> ⚠️ 隐私说明：本仓库**不含任何真实个人隐私**。内置的 `knowledge/` 知识库是**虚构演示数据**（候选人「张明」），仅用于展示框架效果。代码层零硬编码个人信息——所有「服务谁」的配置都外置为数据（`knowledge/persona.json` + `knowledge/raw/`），换一个人只需替换数据、重新入库，框架零改动。

## 它解决了什么问题

传统简历是静态 PDF，面试官只能被动阅读。这个项目把简历变成**可对话、可溯源、可量化**的 Agent：

- **可对话**：自然语言问答，覆盖项目经历、技术栈、求职动机
- **可溯源**：每条回答标注知识库来源，面试官可点开看原文
- **可量化**：内置离线评测脚本，用 Recall@K / MRR 证明检索优化效果
- **可复用**：代码与数据彻底解耦，同一套框架服务不同候选人

## 技术选型（已落地）

| 层 | 选型 |
|---|---|
| 前端 | 原生 SPA（零构建），SSE 流式输出、打字机效果、引用卡片 |
| 服务 | FastAPI（`/api/chat` SSE 流式、`/api/health` 探活），会话隔离 |
| 编排 | LangGraph StateGraph：改写 → 检索 → 相关性判定 → 生成 → 引用 |
| 向量库 | Qdrant（嵌入式开发 / 服务端部署，payload 元数据过滤） |
| 混检 | BM25 稀疏 + bge-small-zh-v1.5 稠密双路召回，RRF 融合 |
| 重排 | bge-reranker-base（默认关闭，部署时启用，链路已支持） |
| 记忆 | 滑动窗口 + 摘要（多轮上下文管理） |
| Query 改写 | 指代消解 + 追问补全（规则版，无 LLM 可跑，白名单防污染） |
| 拒答 | 词面重叠度 + 姓名扣除判定，知识库无依据时坦诚拒答 |
| 模型适配 | 统一 LLM 接口，DeepSeek / Qwen / GLM 可切换，无 key 时规则降级 |
| Embedding | 开发档 BAAI/bge-small-zh-v1.5（512 维）；可切 jina-embeddings-v3（1024 维） |
| 换人入口 | Web 表单 `/api/bootstrap`（multipart：填写 + 上传）与 CLI `bootstrap` 共用同一套落盘入库逻辑 |
| 交付 | Docker Compose 一键部署（app + qdrant + nginx） |

## 离线评测（量化数据）

当前评测集共 **14 条 query**（12 条可召回 + 2 条拒答类），在本地 Qdrant 嵌入式模式 + bge-small-zh-v1.5 + BM25 上跑出的结果：

| 配置 | R@1 | Recall@5 | MRR |
|---|---|---|---|
| 仅稠密 | 75.0% | 91.7% | 0.806 |
| 稠密 + 稀疏（RRF） | 75.0% | 91.7% | 0.794 |
| 完整（+重排） | 75.0% | 91.7% | 0.794* |

> *注：当前部署包未内置 `bge-reranker-base` 模型，重排链路会降级为 RRF 融合排序，因此「完整（+重排）」这一行的指标与「稠密 + 稀疏」一致。接入模型或设置 `HF_OFFLINE=false` 后可恢复真实重排效果。

相对基线（仅稠密）：Recall@5 持平，MRR −0.011。目前 14 条样本下稀疏召回对 dense 基线提升有限，主要由于项目描述本身较短，dense embedding 已能较好覆盖。

## 目录结构

```
├── knowledge/
│   ├── persona.json            人格配置（姓名 / 项目白名单 / 关键词）—— 数据，非代码
│   ├── raw/                     RAG 原始素材（按类归档，换人只改这里）
│   │   ├── 01-profile/          基本信息、个人档案
│   │   ├── 02-projects/         项目文档
│   │   └── 03-qa/              模拟面试问答对
│   └── processed/              切片产物（ingest 自动生成，可删）
├── backend/
│   ├── api/                    FastAPI 服务入口（SSE）
│   ├── agent/                  LangGraph 编排、记忆、Query 改写
│   ├── rag/                    加载、切片、向量库、检索、评测
│   ├── core/                   Embedding / Reranker / LLM 抽象层 / persona
│   ├── scripts/                入库、一键换人（bootstrap，Web 与 CLI 共用）、检索测试
│   └── config/settings.py      全局配置（可用 .env 覆盖）
├── frontend/                   零构建 SPA（index.html / app.js / style.css）
├── eval/                       评测集 + 评测脚本
└── deploy/                     Dockerfile / nginx.conf / docker-compose.yml
```

## 把它改成「你自己的」简历助手

框架与数据解耦，**换一个人只需提供资料，无需改一行代码**。三种方式任选：

### 方式一：网页端一键换人（最省事）

启动服务后点击右上角 **「更换资料」**，在弹层里**直接填写**或**上传文件**（pdf / docx / md / txt，支持多选），点「保存并入库」即可，后端会自动切片、写入向量库并让新资料立即生效（无需重启）。

- 每个项目**单独填写会落盘成独立文档**，切片边界更干净，不会把两个项目揉进同一段；
- 文本首行以 `# ` 开头会被识别为文档标题，并自动收录进项目白名单（供指代消解使用）；
- 服务端若配置了 `REINDEX_TOKEN`，需在「高级选项 → 管理 Token」里填写——否则任何人都能重建你的知识库；
- 入库是**覆盖式**的：成功后自动开新对话，因为旧会话的多轮上下文属于上一个人。

### 方式二：命令行 bootstrap（适合本地批量换资料）

准备你的资料（简历 pdf/docx/md、项目文档、面试问答 md），一条命令完成「写入知识库 → 生成 persona → 自动切片入库」：

```bash
$PY -m backend.scripts.bootstrap \
    --name 张三 \
    --profile resume.pdf \
    --projects project1.md project2.md \
    --qa interview_qa.md
```

- `--anchors` / `--keywords` 可省略：不传时 `anchors` 自动取你项目文件的标题；
- 只想先看效果、暂不下载向量模型，加 `--dry-run`（仅生成 `knowledge/raw` 与 `persona.json`）；
- bootstrap 是**覆盖式**：每次运行会清空旧资料、重建向量库，不会出现上一个人的碎片残留。

### 方式三：手动

1. 编辑 `knowledge/persona.json`：`name` / `project_anchors` / `project_keywords`
2. 替换 `knowledge/raw/` 下的 markdown（个人档案、项目、问答对）
3. 运行 `python -m backend.scripts.ingest` 重建向量库

代码无需任何改动。

## 本地运行

```bash
# 0. 准备虚拟环境（依赖：fastapi uvicorn langgraph qdrant-client fastembed loguru pydantic-settings）
PY=python            # 或你的 venv 解释器
export PYTHONPATH=$(pwd) && unset http_proxy https_proxy

# 1. 知识库入库（首次下载 bge-small + bm25，约 100MB，走镜像源）
$PY -m backend.scripts.ingest

# 2. 启动服务（默认 8000 端口，无 LLM key 时走演示降级）
$PY -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000

# 3. 浏览器打开 http://127.0.0.1:8000
```

配置真实 LLM（生成质量升级，可选）：复制 `.env.example` 为 `.env`，填入 `LLM_API_KEY`（DeepSeek / Qwen / GLM 任一，兼容 OpenAI 接口）。

## Docker 部署

```bash
docker compose -f deploy/docker-compose.yml up --build
```

## 隐私与合规

- 代码层**零硬编码个人信息**；所有「服务谁」的信息都在 `knowledge/` 数据目录。
- 换人 = 替换 `knowledge/` 并重新入库，无需改一行代码。
- `.env`（含 API key）、`.qdrant_data/`（向量）、`.model_cache/`（模型）均在 `.gitignore` 中，不会提交。
