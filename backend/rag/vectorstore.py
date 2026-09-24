"""Qdrant 向量库：建库、写入、混合检索。

本地开发用嵌入式模式（path=），无需 Docker，直接读写本地目录，
API 与服务端模式完全一致；部署时把 qdrant_mode 切成 server 即可，业务代码零改动。

⚠️ 客户端必须是进程级单例（见 _get_client）：嵌入式模式下每次 QdrantClient(path=)
都会对 <path>/.lock 加 portalocker 独占非阻塞锁，并一直持有到实例关闭。
服务进程内若存在第二个实例（如 /api/reindex 里新建的 VectorStore），
就会抛 "Storage folder ... is already accessed by another instance"，
表现为「服务被访问一次之后，重建索引与一键换人永久 500」。
"""

from __future__ import annotations

import gc
import shutil
import threading
import uuid
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient, models

from backend.config.settings import settings
from backend.core.embedding import get_embedding_service
from backend.rag.chunker import Chunk


def _stable_id(source: str, chunk_index: int) -> str:
    """基于来源与序号生成稳定 ID，重复入库不会产生脏数据。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{chunk_index}"))


# 进程级共享客户端 + 互斥锁
# 读写互斥：reindex / bootstrap 会 delete+create 集合，
# 此时若有检索并发进来，会读到「集合不存在」而抛错，因此用同一把锁串行化。
_client: QdrantClient | None = None
_client_lock = threading.Lock()
_io_lock = threading.RLock()


def _new_client() -> QdrantClient:
    """真正创建客户端（仅应被调用一次）。"""
    if settings.qdrant_mode == "server":
        logger.info(f"连接 Qdrant 服务端: {settings.qdrant_url}")
        return QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None
        )
    path = str(settings.qdrant_path)
    logger.info(f"使用 Qdrant 嵌入式模式，数据目录: {path}")
    return QdrantClient(path=path)


def _get_client() -> QdrantClient:
    """获取进程级唯一的 Qdrant 客户端。

    嵌入式模式的 .lock 是独占的，重复实例化必然失败；
    这里做单例缓存，让 VectorStore / Retriever / 各接口共用同一个句柄。
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = _new_client()
    return _client


class VectorStore:
    def __init__(self) -> None:
        self.embeddings = get_embedding_service()
        self.collection = settings.qdrant_collection
        self.client = _get_client()

    @property
    def io_lock(self) -> threading.RLock:
        """集合级读写锁：重建集合与检索需互斥，避免重建期间检索踩空。"""
        return _io_lock

    # ---------- 建库 ----------
    def ensure_collection(self, recreate: bool = False) -> None:
        with _io_lock:
            self._ensure_collection_locked(recreate)

    def _ensure_collection_locked(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            logger.info(f"复用已有集合: {self.collection}")
            return
        if exists:
            logger.info(f"重建集合: {self.collection}")
            if self._drop_collection():
                self._create_collection()
                return
            # 目录删不掉（Windows 常见）：退化为「重建 + 清空全部点」，
            # 宁可多一步，也不能让旧人的资料残留在新集合里。
            if not self.client.collection_exists(self.collection):
                self._create_collection()
            self._clear_points()
            return

        self._create_collection()

    def _create_collection(self) -> None:
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=settings.dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )
        logger.info(f"集合已创建: {self.collection}")

    def _collection_dir(self) -> Path | None:
        """集合在本地磁盘上的目录（仅嵌入式模式有，服务端模式为 None）。"""
        local = getattr(self.client, "_client", None)
        fn = getattr(local, "_collection_path", None)
        if not callable(fn):
            return None
        try:
            return Path(fn(self.collection))
        except Exception:  # pragma: no cover
            return None

    def _drop_collection(self) -> bool:
        """彻底删除集合，返回是否成功。

        嵌入式模式在 Windows 上有个隐蔽的坑：集合持有 sqlite 连接时，
        delete_collection 内部的 shutil.rmtree(..., ignore_errors=True) 会静默失败——
        目录和数据都还在，而紧接着的 create_collection 会复用同一目录，
        把旧点原封不动加载回来。表现就是「覆盖式换人」看似成功，
        库内却仍残留上一个人的全部片段，检索时被一起召回。

        因此这里先显式关闭集合句柄释放文件锁，再删除，并校验目录确实消失。
        """
        name = self.collection
        # 1) 关闭句柄，释放 Windows 上的 sqlite 文件锁
        local = getattr(self.client, "_client", None)
        col = getattr(local, "collections", {}).get(name) if local is not None else None
        if col is not None:
            try:
                col.close()
            except Exception:  # pragma: no cover
                pass

        # 2) 删除（内部会 rmtree，但可能被 ignore_errors 吞掉）
        try:
            self.client.delete_collection(name)
        except Exception as e:  # pragma: no cover
            logger.warning(f"删除集合失败: {e}")
            return False

        # 3) 校验并兜底再删一次
        path = self._collection_dir()
        if path is not None and path.exists():
            gc.collect()
            shutil.rmtree(path, ignore_errors=True)
        if path is not None and path.exists():
            logger.warning(f"集合目录仍存在，判定删除未生效: {path}")
            return False
        return True

    def _clear_points(self) -> None:
        """清空集合内全部点（清空而非删除集合，用于目录删不掉的兜底）。"""
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[])),
        )
        logger.info(f"已清空集合内全部点: {self.collection}")

    def count(self) -> int:
        with _io_lock:
            return self._count_locked()

    def _count_locked(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count

    # ---------- 写入 ----------
    def upsert(self, chunks: list[Chunk], batch_size: int = 32) -> int:
        if not chunks:
            return 0
        with _io_lock:
            return self._upsert_locked(chunks, batch_size)

    def _upsert_locked(self, chunks: list[Chunk], batch_size: int = 32) -> int:
        self._ensure_collection_locked()
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]
            payloads = [c.to_payload() for c in batch]

            dense_vecs = self.embeddings.embed_dense(texts)
            sparse_vecs, ok = self.embeddings.embed_sparse(texts)

            points = []
            for chunk, payload, dense in zip(batch, payloads, dense_vecs):
                vector: dict = {"dense": dense}
                if ok and sparse_vecs:
                    vector["sparse"] = sparse_vecs[len(points)]
                points.append(
                    models.PointStruct(
                        id=_stable_id(chunk.source, chunk.chunk_index),
                        vector=vector,
                        payload=payload,
                    )
                )

            self.client.upsert(collection_name=self.collection, points=points)
            total += len(points)
            logger.debug(f"已写入 {total}/{len(chunks)}")
        logger.info(f"入库完成，共 {total} 个向量")
        return total

    # ---------- 检索 ----------
    def _build_filter(self, doc_type: str | None):
        if not doc_type:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_type", match=models.MatchValue(value=doc_type)
                )
            ]
        )

    def hybrid_search(
        self,
        query: str,
        limit: int | None = None,
        doc_type: str | None = None,
    ) -> list[dict]:
        """稠密 + 稀疏双路召回，Qdrant 侧 RRF 融合。

        稀疏路失败时自动退化为纯稠密检索。
        """
        limit = limit or settings.prefetch_limit
        with _io_lock:
            return self._hybrid_search_locked(query, limit, doc_type)

    def _hybrid_search_locked(
        self, query: str, limit: int, doc_type: str | None
    ) -> list[dict]:
        if self._count_locked() == 0:
            logger.warning("知识库为空，请先执行入库")
            return []

        flt = self._build_filter(doc_type)
        dense_query = self.embeddings.embed_dense_query(query)
        sparse_query, sparse_ok = self.embeddings.embed_sparse_query(query)

        prefetch = [
            models.Prefetch(
                query=dense_query, using="dense", limit=limit, filter=flt
            )
        ]
        if sparse_ok:
            prefetch.append(
                models.Prefetch(
                    query=sparse_query, using="sparse", limit=limit, filter=flt
                )
            )

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        results = []
        for i, point in enumerate(response.points):
            payload = point.payload or {}
            results.append(
                {
                    "text": payload.get("text", ""),
                    "doc_type": payload.get("doc_type", ""),
                    "source": payload.get("source", ""),
                    "title": payload.get("title", ""),
                    "heading_path": payload.get("heading_path", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "fusion_rank": i + 1,
                    "fusion_score": float(point.score),
                }
            )
        return results
