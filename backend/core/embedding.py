"""双向量编码服务（稠密 + 稀疏）。

架构上把「编码后端」抽象出来，目前有两种实现：

- ``FastEmbedBackend``：基于 fastembed 的真实模型（bge-small-zh-v1.5 / jina-embeddings-v3）。
  这是生产档，需要先把模型下载到本地缓存，质量最好。
- ``DummyBackend``：纯算法生成的确定性向量（字符 / 词粒度的哈希特征），
  **完全离线、零下载**，相似度来自字面重叠。它用来在「模型还没下载」或
  「跑单元测试」时把整条 RAG 链路（切片 -> 入库 -> 混合检索 -> 重排）跑通，
  验证工程正确性。它不是最终方案，切换回真实模型只需改一个配置项。

切法：``settings.embedding_backend`` —— ``"dummy"`` 走离线后端，``"fastembed"`` 走真实模型。
向量库维度以当前后端为准，切换后端后入库脚本会自动重建集合，无需手工清理。
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import Iterable, Sequence

from loguru import logger
from qdrant_client.models import SparseVector

from backend.config.settings import settings


# --------------------------------------------------------------------------- #
# 后端接口
# --------------------------------------------------------------------------- #
class BaseEmbeddingBackend:
    """所有编码后端必须实现的接口。"""

    dim: int = 256

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_dense_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_sparse(
        self, texts: Sequence[str]
    ) -> tuple[list[SparseVector] | None, bool]:
        raise NotImplementedError

    def embed_sparse_query(self, text: str):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 真实后端：fastembed
# --------------------------------------------------------------------------- #
class FastEmbedBackend(BaseEmbeddingBackend):
    def __init__(self) -> None:
        self._dense = None
        self._sparse = None
        self._lock = threading.Lock()
        self._sparse_broken = False
        self.dim = settings.dense_dim

    @property
    def dense(self):
        if self._dense is None:
            with self._lock:
                if self._dense is None:
                    from fastembed import TextEmbedding

                    logger.info(f"加载稠密向量模型: {settings.dense_model}")
                    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
                    self._dense = TextEmbedding(
                        model_name=settings.dense_model,
                        cache_dir=str(settings.model_cache_dir),
                    )
        return self._dense

    @property
    def sparse(self):
        if not settings.enable_sparse or self._sparse_broken:
            return None
        if self._sparse is None:
            with self._lock:
                if self._sparse is None and not self._sparse_broken:
                    try:
                        from fastembed import SparseTextEmbedding

                        logger.info(f"加载稀疏向量模型: {settings.sparse_model}")
                        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
                        self._sparse = SparseTextEmbedding(
                            model_name=settings.sparse_model,
                            cache_dir=str(settings.model_cache_dir),
                        )
                    except Exception as e:  # 稀疏侧失败不应阻断主流程
                        logger.warning(f"稀疏向量模型不可用，降级为纯稠密检索: {e}")
                        self._sparse_broken = True
                        return None
        return self._sparse

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self.dense.passage_embed(list(texts))]

    def embed_dense_query(self, text: str) -> list[float]:
        return next(iter(self.dense.query_embed(text))).tolist()

    def embed_sparse(
        self, texts: Sequence[str]
    ) -> tuple[list[SparseVector] | None, bool]:
        model = self.sparse
        if model is None:
            return None, False
        vectors = [
            SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
            for emb in model.embed(list(texts))
        ]
        return vectors, True

    def embed_sparse_query(self, text: str):
        model = self.sparse
        if model is None:
            return None, False
        emb = next(iter(model.query_embed(text)))
        return (
            SparseVector(
                indices=emb.indices.tolist(), values=emb.values.tolist()
            ),
            True,
        )


# --------------------------------------------------------------------------- #
# 离线后端：确定性哈希向量（零下载）
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[一-鿿]|[a-zA-Z0-9]+")


def _tokens(text: str) -> list[str]:
    """CJK 单字 + 连续英文/数字词，统一小写。"""
    return [m.group().lower() for m in _TOKEN_RE.finditer(text or "")]


def _hashed_vector(text: str, dim: int) -> list[float]:
    """把文本映射成固定维度、单位长度的向量。

    特征 = 所有 token 的 unigram + 相邻 token 的 bigram，用哈希技巧落到维度上。
    相似度由字面重叠决定——足够用来验证检索链路，但不是语义向量。
    """
    vec = [0.0] * dim
    toks = _tokens(text)
    feats = set(toks)
    for i in range(len(toks) - 1):
        feats.add(f"{toks[i]}|{toks[i + 1]}")
    for feat in feats:
        h = int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class DummyBackend(BaseEmbeddingBackend):
    """离线后端：不依赖任何模型文件，纯算法生成向量。

    注意：它提供的是「字面相似」，不是「语义相似」。用它跑通的是工程链路，
    检索质量远低于真实模型——切勿把它当成最终效果写进简历。
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hashed_vector(t, self.dim) for t in texts]

    def embed_dense_query(self, text: str) -> list[float]:
        return _hashed_vector(text, self.dim)

    def embed_sparse(
        self, texts: Sequence[str]
    ) -> tuple[list[SparseVector] | None, bool]:
        # 离线后端只提供稠密向量，稀疏路标记为不可用，检索自动退化为单路。
        return None, False

    def embed_sparse_query(self, text: str):
        return None, False


# --------------------------------------------------------------------------- #
# 服务层
# --------------------------------------------------------------------------- #
class EmbeddingService:
    """双向量编码服务，后端可插拔、线程安全。"""

    def __init__(self) -> None:
        backend = settings.embedding_backend.lower()
        if backend == "fastembed":
            logger.info("编码后端: fastembed（真实模型）")
            self.backend: BaseEmbeddingBackend = FastEmbedBackend()
        elif backend == "dummy":
            logger.info("编码后端: dummy（离线哈希向量，零下载）")
            self.backend = DummyBackend(dim=settings.dummy_dim)
        else:
            raise ValueError(f"未知 embedding_backend: {settings.embedding_backend}")

    @property
    def dense_dim(self) -> int:
        return self.backend.dim

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        return self.backend.embed_dense(texts)

    def embed_dense_query(self, text: str) -> list[float]:
        return self.backend.embed_dense_query(text)

    def embed_sparse(
        self, texts: Sequence[str]
    ) -> tuple[list[SparseVector] | None, bool]:
        return self.backend.embed_sparse(texts)

    def embed_sparse_query(self, text: str):
        return self.backend.embed_sparse_query(text)


_service: EmbeddingService | None = None
_service_lock = threading.Lock()


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = EmbeddingService()
    return _service
