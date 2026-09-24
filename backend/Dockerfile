# 后端服务镜像：FastAPI + RAG 检索链路
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

# 先装依赖（利用层缓存）
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt

# 复制代码
COPY backend/ ./backend/
COPY knowledge/ ./knowledge/
COPY eval/ ./eval/

# 国内环境下载模型必须的配置：走镜像源、禁用代理与 symlink
ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_PROXY=1 \
    HF_HUB_DISABLE_SYMLINKS=True \
    HF_HUB_DOWNLOAD_TIMEOUT=300 \
    no_proxy='*' NO_PROXY='*' \
    QDRANT_MODE=server \
    QDRANT_URL=http://qdrant:6333

EXPOSE 8000

# 启动：先建库（首次会从镜像站下载模型），再起服务
CMD ["sh", "-c", "python -m backend.scripts.ingest && python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000"]
