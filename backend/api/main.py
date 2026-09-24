"""FastAPI 服务：SSE 流式对话 + 静态前端托管。

- POST /api/chat   流式对话（Server-Sent Events），支持多轮会话隔离
- GET  /api/health 健康检查（含模型 / 知识库状态）
- GET  /           托管 frontend/ 下的单页前端
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from backend.config.settings import settings
from backend.agent import run_stream


def _bootstrap() -> None:
    """启动自检：知识库为空则自动入库，并预热向量模型。

    部署到干净环境（如容器/云端沙箱）时无需手工执行入库脚本；
    预热把模型提前加载进内存，避免第一个请求卡在模型下载或加载上。
    任一环节失败都不阻断服务启动，仅记录告警。
    """
    from backend.core.embedding import get_embedding_service
    from backend.rag.chunker import chunk_documents
    from backend.rag.loader import load_documents
    from backend.rag.vectorstore import VectorStore

    vs = VectorStore()
    if vs.count() == 0:
        logger.info("知识库为空，启动时自动入库…")
        docs = load_documents()
        if docs:
            chunks = chunk_documents(docs)
            vs.ensure_collection(recreate=True)
            vs.upsert(chunks)
            logger.success(f"自动入库完成，共 {vs.count()} 个片段")
        else:
            logger.warning(f"未找到知识库文档: {settings.knowledge_raw_dir}")

    # 预热：触发稠密/稀疏模型加载
    get_embedding_service().embed_dense_query("预热")
    logger.info("向量模型预热完成")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _bootstrap()
    except Exception as e:  # 启动失败不应阻断服务，便于排障
        logger.warning(f"启动自检未完成（服务继续运行）: {e}")
    yield


app = FastAPI(title="RAG 简历助手 · AI Resume Assistant", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = "default"
    query: str


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


_reindex_lock = threading.Lock()


def _check_admin_token(request: Request):
    """校验管理类接口（reindex / bootstrap）的 token。

    未配置 REINDEX_TOKEN 时放行，方便本地开发；
    一旦配置就必须携带 x-reindex-token，否则任何人都能重建知识库、打满资源。
    返回 None 表示通过；否则返回可直接 return 的错误响应。
    """
    expected = settings.reindex_token
    if not expected:
        return None
    provided = request.headers.get("x-reindex-token", "")
    if not secrets.compare_digest(provided, expected):
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    return None


@app.post("/api/reindex")
def reindex(request: Request):
    """重新入库：清空当前集合并从 knowledge/raw 重建。

    用于知识库文件更新后刷新线上向量库（部署平台会复用持久卷，
    仅改文件不会自动重建索引，需显式触发一次）。

    鉴权：配置了 REINDEX_TOKEN 时必须携带 x-reindex-token 请求头。
    线上若不加校验，任何人都能触发一次全量重建（反复调用即可打满资源）。
    并发保护：重建期间会清空集合，两个请求同时进来会互相踩踏，
    因此用一把互斥锁串行化。
    """
    denied = _check_admin_token(request)
    if denied:
        return denied

    from backend.core.embedding import get_embedding_service
    from backend.rag.chunker import chunk_documents
    from backend.rag.loader import load_documents
    from backend.rag.vectorstore import VectorStore

    with _reindex_lock:
        try:
            vs = VectorStore()
            docs = load_documents()
            if not docs:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"未找到知识库文档: {settings.knowledge_raw_dir}"},
                )
            chunks = chunk_documents(docs)
            vs.ensure_collection(recreate=True)
            vs.upsert(chunks)
            get_embedding_service().embed_dense_query("预热")
            return {
                "status": "ok",
                "doc_count": vs.count(),
                "chunk_count": len(chunks),
            }
        except Exception as e:  # pragma: no cover
            logger.exception("重新入库失败")
            return JSONResponse(status_code=500, content={"error": str(e)})


# 单个上传文件大小上限（10MB）。简历/项目文档远小于此，
# 设上限是防止接口被人当网盘用、把内存和磁盘打满。
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _split_terms(s: str) -> list[str]:
    """把「逗号 / 顿号 / 分号 / 换行」分隔的输入拆成列表。"""
    if not s:
        return []
    return [p.strip() for p in re.split(r"[,，、;；\r\n]+", s) if p.strip()]


def _save_uploads(files: list[UploadFile] | None, tmpdir: Path, sub: str) -> list[Path]:
    """把上传文件落到临时目录并校验格式/大小，返回本地路径列表。

    先落盘再交给 bootstrap，是为了让「上传的文件」和「命令行传入的文件」
    走完全相同的处理路径，最终都进入 knowledge/raw 成为可重建的数据源。
    """
    from backend.scripts.bootstrap import SUPPORTED, safe_filename

    saved: list[Path] = []
    for f in files or []:
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix.lower()
        if suffix not in SUPPORTED:
            raise ValueError(
                f"不支持的文件格式: {f.filename}（支持 {', '.join(sorted(SUPPORTED))}）"
            )
        data = f.file.read()
        if not data:
            continue
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"文件过大: {f.filename}（{len(data) / 1024 / 1024:.1f}MB，上限 10MB）"
            )

        target_dir = tmpdir / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        # 文件名可能来自不可信客户端，必须净化后再拼路径（防目录穿越）
        fname = safe_filename(f.filename, default="资料")
        if not fname.lower().endswith(tuple(SUPPORTED)):
            fname += suffix
        dest = target_dir / fname
        stem, ext, i = dest.stem, dest.suffix, 1
        while dest.exists():  # 同名文件加序号，避免互相覆盖
            dest = target_dir / f"{stem}_{i}{ext}"
            i += 1
        dest.write_bytes(data)
        saved.append(dest)
        logger.info(f"收到上传文件 {sub}/{dest.name}（{len(data)} 字节）")
    return saved


@app.post("/api/bootstrap")
def bootstrap_api(
    request: Request,
    name: str = Form(..., description="候选人姓名"),
    profile_text: str = Form("", description="个人档案（网页直接填写）"),
    project_text: list[str] = Form(None, description="项目经历，可重复提交多个"),
    qa_text: str = Form("", description="面试问答对"),
    anchors: str = Form("", description="项目白名单，逗号分隔；留空则自动推断"),
    keywords: str = Form("", description="项目关键词，逗号分隔"),
    profile_files: list[UploadFile] = File(None),
    project_files: list[UploadFile] = File(None),
    qa_files: list[UploadFile] = File(None),
):
    """网页端一键换人：填写文本 / 上传文件 -> 写入 knowledge/raw -> 自动切片入库。

    与 CLI `python -m backend.scripts.bootstrap` 共用同一套落盘与入库逻辑，
    区别只在输入来源：CLI 读本地路径，这里读 multipart 表单。

    用同步 def 定义（FastAPI 会放进线程池执行）是因为入库涉及模型推理与磁盘写入，
    属于阻塞操作，不能占着事件循环——与 /api/reindex 的处理方式一致。
    """
    denied = _check_admin_token(request)
    if denied:
        return denied

    from backend.scripts.bootstrap import bootstrap

    texts = [t for t in (project_text or []) if t and t.strip()]
    if not (
        profile_files
        or project_files
        or qa_files
        or profile_text.strip()
        or qa_text.strip()
        or texts
    ):
        return JSONResponse(
            status_code=400, content={"error": "请至少填写一段资料或上传一个文件"}
        )

    tmpdir: Path | None = None
    try:
        tmpdir = Path(tempfile.mkdtemp(prefix="bootstrap_"))
        pf = _save_uploads(profile_files, tmpdir, "profile")
        jf = _save_uploads(project_files, tmpdir, "projects")
        qf = _save_uploads(qa_files, tmpdir, "qa")

        # 与 reindex 共用同一把锁：换人同样会清空并重建集合，不能并发
        with _reindex_lock:
            result = bootstrap(
                name=name,
                profile_files=pf,
                project_files=jf,
                qa_files=qf,
                profile_text=profile_text,
                project_texts=texts,
                qa_text=qa_text,
                anchors=_split_terms(anchors) or None,
                keywords=_split_terms(keywords),
                do_ingest=True,
            )
        return result
    except ValueError as e:
        # 格式/大小等属于用户输入问题，回 400 让前端直接展示原因
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:  # pragma: no cover
        logger.exception("一键换人失败")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # 临时目录仅作中转，真正的数据已落进 knowledge/raw
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/api/health")
def health():
    from backend.rag.vectorstore import VectorStore

    try:
        vs = VectorStore()
        count = vs.count()
    except Exception as e:  # pragma: no cover
        count = -1
        logger.warning(f"知识库状态获取失败: {e}")
    model_dir = settings.model_cache_dir
    cached = sorted(p.name for p in model_dir.iterdir()) if model_dir.exists() else []
    return {
        "status": "ok",
        "llm": "configured" if settings.llm_api_key else "demo(fallback)",
        "embedding": settings.embedding_backend,
        "rerank": "on" if settings.enable_rerank else "off(fallback)",
        "doc_count": count,
        # 运行期诊断：模型是否随包提供、离线模式是否生效
        "model_cache_dir": str(model_dir),
        "model_cache_exists": model_dir.exists(),
        "model_cache_entries": cached[:10],
        "hf_offline": os.environ.get("HF_HUB_OFFLINE", "0"),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    query = (req.query or "").strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": "query 不能为空"})

    async def event_gen():
        # run_stream 是同步生成器（内部含检索/LLM 等阻塞调用），
        # 放到线程池执行，避免阻塞 FastAPI 事件循环，保证并发请求互不卡死（P0-6）
        loop = asyncio.get_event_loop()
        gen = run_stream(req.session_id, query)
        while True:
            event = await loop.run_in_executor(None, next, gen, None)
            if event is None:
                break
            yield _sse(event)
        yield ": keep-alive end\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/")
def index():
    index_path: Path = settings.frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"msg": "前端未构建，请访问 /api/chat 进行接口联调"})


@app.get("/{path:path}")
def spa(path: str):
    """SPA 兜底：命中静态文件则返回，否则回退到 index.html。"""
    target = settings.frontend_dir / path
    if target.exists() and target.is_file():
        return FileResponse(target)
    index_path = settings.frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"msg": "not found"}, status_code=404)
