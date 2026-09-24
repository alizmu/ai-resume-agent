"""知识入库脚本。

用法：
    python -m backend.scripts.ingest              # 全量重建
    python -m backend.scripts.ingest --append     # 追加模式，保留已有集合

核心逻辑抽成 build_index()，可被 bootstrap 等脚本以编程方式调用，
避免「换人入库」与「手动入库」两套重复实现。
"""

from __future__ import annotations

import argparse
import json
import time

from loguru import logger

from backend.config.settings import settings
from backend.rag.chunker import chunk_documents
from backend.rag.loader import load_documents
from backend.rag.vectorstore import VectorStore


def dump_chunks(chunks) -> None:
    """把切片结果落盘，便于人工检查切片质量——这一步是调优检索效果的主要依据。"""
    out_dir = settings.knowledge_processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "chunks.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_payload(), ensure_ascii=False) + "\n")
    logger.info(f"切片结果已导出: {out_file}")


def build_index(recreate: bool = True) -> dict:
    """加载原始知识库 -> 切片 -> 落盘 -> 写入向量库。

    recreate=True 时先清空旧集合再建（覆盖式），避免换人后旧碎片残留。
    返回统计信息供调用方展示。
    """
    started = time.time()
    docs = load_documents()
    if not docs:
        logger.error(f"未找到任何文档，请检查目录: {settings.knowledge_raw_dir}")
        return {"docs": 0, "chunks": 0, "stored": 0}

    chunks = chunk_documents(docs)
    dump_chunks(chunks)

    store = VectorStore()
    store.ensure_collection(recreate=recreate)
    store.upsert(chunks)

    stats = {
        "docs": len(docs),
        "chunks": len(chunks),
        "stored": store.count(),
        "elapsed": round(time.time() - started, 1),
    }
    logger.success(
        f"入库完成：{stats['docs']} 篇文档 -> {stats['chunks']} 个片段，"
        f"库内总量 {stats['stored']}，耗时 {stats['elapsed']}s"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="构建向量知识库")
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加模式，不重建集合（适合增量更新）",
    )
    args = parser.parse_args()
    build_index(recreate=not args.append)


if __name__ == "__main__":
    main()
