"""全局配置。所有可通过 .env 覆盖，见 .env.example。"""

import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.core.persona import get_persona

# backend/ 的上一级即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_subject_name() -> str:
    """候选人姓名默认从 persona.json 读取，避免写死在代码里（隐私 + 可复用）。"""
    try:
        return get_persona().name
    except Exception:
        return "候选人"


def _apply_hf_endpoint(endpoint: str) -> None:
    """国内网络直连 HuggingFace 经常超时，默认改走镜像站。

    仅在本进程内生效，且不会覆盖用户显式设置的环境变量。
    大文件握手耗时长，huggingface_hub 默认 10 秒超时会误判为失败，这里放宽。
    """
    if endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = endpoint
    if not os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT"):
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
    # Xet 是 HuggingFace 新的大文件存储后端，在国内网络下常返回 401，
    # 关掉后回退到传统 HTTP 下载，实测可达满速。
    if not os.environ.get("HF_HUB_DISABLE_XET"):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    # 关键：本机环境代理对 HuggingFace/镜像站的文件下载会返回 502，
    # 但直连（不走代理）反而满速。因此下载相关请求全部禁用代理。
    if not os.environ.get("HF_HUB_DISABLE_PROXY"):
        os.environ["HF_HUB_DISABLE_PROXY"] = "1"
    if not os.environ.get("no_proxy"):
        os.environ["no_proxy"] = "*"
        os.environ["NO_PROXY"] = "*"
    # Windows 不支持 symlink，huggingface_hub 退化出的快照文件会是 0 字节空壳，
    # 导致 onnx 加载报 "ModelProto does not have a graph"。禁用 symlink 改用复制，
    # 文件即为完整副本，彻底规避该问题。
    if not os.environ.get("HF_HUB_DISABLE_SYMLINKS"):
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "True"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- 知识库 ----------
    knowledge_raw_dir: Path = PROJECT_ROOT / "knowledge" / "raw"
    knowledge_processed_dir: Path = PROJECT_ROOT / "knowledge" / "processed"
    # 候选人姓名（改写与拒答判定共用，避免把姓名误当「实质命中」）。
    # 默认从 knowledge/persona.json 读取；也可用 .env 的 SUBJECT_NAME 覆盖。
    subject_name: str = _default_subject_name()

    # ---------- Qdrant ----------
    # local = 嵌入式模式（无需 Docker，直接读写本地目录）
    # server = 服务端模式（Docker Compose 部署的 Qdrant）
    qdrant_mode: str = "local"
    qdrant_path: Path = PROJECT_ROOT / ".qdrant_data"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "resume_agent"

    # ---------- 向量模型 ----------
    # 模型分两档，通过 .env 一行切换，代码层完全无感：
    #   dev 档（默认，约 100MB，分钟级可下载完，开发调试用）
    #     dense = BAAI/bge-small-zh-v1.5        512 维
    #   prod 档（约 2.3GB，效果更好，部署前再下载）
    #     dense = jinaai/jina-embeddings-v3     1024 维
    # 注意：jina-v3 内部用 task_id 区分「查询」与「文档」编码，
    # 入库必须走 passage_embed、检索必须走 query_embed，不能混用同一个 embed。
    dense_model: str = "BAAI/bge-small-zh-v1.5"
    dense_dim: int = 512
    # 编码后端：fastembed=真实模型（需先下载到本地缓存），
    #          dummy=离线哈希向量（零下载，仅用于验证整条工程链路，检索质量远低于真实模型）
    embedding_backend: str = "fastembed"
    # 离线后端（dummy）的向量维度
    dummy_dim: int = 256
    # 稀疏向量负责关键词侧召回，补上稠密模型对低频专有名词的短板
    sparse_model: str = "Qdrant/bm25"
    enable_sparse: bool = True
    # 重排模型：中文场景需加载 1GB 级的 cross-encoder，默认关闭以加快冷启动。
    # 关掉时检索链路自动降级为「双路召回 + RRF 融合」，功能完整。
    # 部署前设 ENABLE_RERANK=true 并下载 BAAI/bge-reranker-base 即可启用。
    rerank_model: str = "BAAI/bge-reranker-base"
    enable_rerank: bool = False
    # 模型缓存目录，默认放项目下便于整体迁移
    model_cache_dir: Path = PROJECT_ROOT / ".model_cache"
    # 模型下载镜像，留空则用 HuggingFace 官方源
    hf_endpoint: str = "https://hf-mirror.com"
    # 离线模式：模型已随部署包提供时开启，禁止运行时联网。
    # 在无法访问 HuggingFace 的环境（如部分云端沙箱）里，
    # 不开启会导致首次编码长时间挂起（等待网络超时），表现为请求无响应。
    hf_offline: bool = False

    # ---------- 检索参数 ----------
    # 每路召回的候选数量，融合后再交给 rerank 精排
    prefetch_limit: int = 20
    # rerank 之后最终返回的片段数
    final_top_k: int = 5
    # rerank 打分低于该阈值时判定为「知识库中没有可靠依据」
    relevance_threshold: float = 0.3
    # 关闭重排时，相关性兜底判定扫描的召回片段数量（默认看前 5 个，而非只看 Top1，
    # 避免 RRF 因候选人姓名高频导致 Top1 偏离时误拒库内问题）
    reliability_scan_k: int = 5

    # ---------- LLM（生成环节，多模型抽象层）----------
    # 兼容 OpenAI 接口，支持 DeepSeek / Qwen / GLM 切换。
    # 未配置 key 时自动降级为规则生成（基于检索上下文拼接），保证链路可跑通。
    llm_provider: str = "deepseek"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    # LLM 流式首 token / 单次读取超时（秒）。部署环境网关通常有更短的上层超时，
    # 设置合理上限可避免请求一直挂起；超时时自动降级为规则生成。
    llm_timeout: float = 10.0

    # ---------- 切片策略 ----------
    # 不同文档类型使用不同的块大小：
    # - qa 问答对语义完整，必须整块保留，一旦切断上下文就丢了
    # - profile / projects 按标题层级切，允许更小的块以换取检索精度
    chunk_size: int = 600
    chunk_overlap: int = 80
    min_chunk_size: int = 40

    # ---------- 服务与前端 ----------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # 允许跨域的前端来源（部署时由 nginx 反代可收紧为同源）
    cors_origins: list[str] = [
        "http://localhost:8000",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
    ]
    # 前端静态文件目录（Docker 内由 nginx 托管，本地开发由 FastAPI 托管）
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    # 重新入库接口的访问令牌。留空表示不校验（仅建议本地开发时如此）；
    # 线上部署必须设置，否则任何人都能调用 /api/reindex 重建向量库。
    reindex_token: str = ""

    @field_validator("model_cache_dir", "qdrant_path", mode="after")
    @classmethod
    def _resolve_relative(cls, v: Path) -> Path:
        """相对路径统一按项目根目录解析。

        部署时 .env 里写相对目录更方便（不依赖宿主机绝对路径），
        但进程工作目录未必是项目根，不解析会写错位置。
        """
        p = Path(v)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def chunk_size_by_type(self) -> dict[str, int]:
        return {
            "profile": 500,
            "projects": 700,
            "qa": 2000,
            "notes": 600,
            "jd": 600,
        }


def _apply_hf_offline(offline: bool) -> None:
    """离线模式：模型文件已随部署包提供时开启。

    不开启的话，fastembed 在加载模型时可能先去 HuggingFace 校验，
    在无法出网的环境里会一直等到网络超时，表现为接口卡死无响应。
    """
    if offline and not os.environ.get("HF_HUB_OFFLINE"):
        os.environ["HF_HUB_OFFLINE"] = "1"


settings = Settings()
_apply_hf_endpoint(settings.hf_endpoint)
_apply_hf_offline(settings.hf_offline)
