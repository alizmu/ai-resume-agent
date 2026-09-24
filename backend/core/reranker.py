"""Cross-Encoder 重排序。

为什么需要这一步：
双路召回（稠密 + 稀疏）拿到的是「可能相关」的候选，排序依据是向量相似度，
它只看查询和文档各自的表征，两者没有真正交互。Cross-Encoder 把查询和文档拼在一起
过一遍模型，能捕捉到向量检索漏掉的细微相关性，代价是慢——所以只用在 RRF 融合后的
Top-20 上，不用于全库。
"""

from __future__ import annotations

import threading
from typing import Sequence

from loguru import logger

from backend.config.settings import settings


class Reranker:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._broken = False

    @property
    def model(self):
        if not settings.enable_rerank or self._broken:
            return None
        if self._model is None:
            with self._lock:
                if self._model is None and not self._broken:
                    try:
                        from fastembed.rerank.cross_encoder import TextCrossEncoder

                        logger.info(f"加载重排模型: {settings.rerank_model}")
                        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
                        self._model = TextCrossEncoder(
                            model_name=settings.rerank_model,
                            cache_dir=str(settings.model_cache_dir),
                        )
                    except Exception as e:
                        logger.warning(f"重排模型不可用，降级为融合排序: {e}")
                        self._broken = True
                        return None
        return self._model

    def rerank(self, query: str, documents: Sequence[str]) -> list[float] | None:
        """返回与 documents 等长的分数列表；不可用时返回 None。"""
        model = self.model
        if model is None or not documents:
            return None

        texts = list(documents)
        try:
            if hasattr(model, "rerank"):
                scores = list(model.rerank(query, texts))
            elif hasattr(model, "rank"):
                results = list(model.rank(query, texts))
                by_index = {r.index: r.score for r in results}
                scores = [by_index.get(i, 0.0) for i in range(len(texts))]
            else:  # 兜底：直接调用
                scores = list(model(query, texts))
        except Exception as e:
            logger.warning(f"重排失败，降级为融合排序: {e}")
            return None

        if len(scores) != len(texts):
            logger.warning("重排返回长度不匹配，降级为融合排序")
            return None
        return [float(s) for s in scores]


_reranker: Reranker | None = None
_lock = threading.Lock()


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = Reranker()
    return _reranker
