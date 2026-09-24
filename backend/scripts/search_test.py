"""检索效果检查脚本（不需要 LLM 就能跑）。

用法：
    python -m backend.scripts.search_test "政务大厅项目他做了什么"
    python -m backend.scripts.search_test                 # 进入交互模式
    python -m backend.scripts.search_test "..." --compare # 对比 RRF 与 Rerank 的排序差异

这个脚本是调优检索效果的主要工具：直接看每次查询召回了哪些片段、排序如何，
比看任何汇总指标都直观。
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from backend.config.settings import settings
from backend.rag.retriever import get_retriever


def print_results(query: str, results, top_k: int) -> None:
    print(f"\n查询：{query}")
    print(f"召回 {len(results)} 条（配置 top_k={top_k}，阈值 {settings.relevance_threshold}）")
    print("=" * 78)
    if not results:
        print("没有召回任何内容。知识库可能为空，请先运行入库脚本。")
        return

    reliable = results[0].final_score >= settings.relevance_threshold
    for i, r in enumerate(results, start=1):
        rerank = f"{r.rerank_score:.4f}" if r.rerank_score is not None else "  --  "
        print(f"\n[{i}] 最终分 {r.final_score:.4f} | RRF第{r.fusion_rank}位 | Rerank {rerank}")
        print(f"    来源：{r.citation}")
        print(f"    类型：{r.doc_type}  文件：{r.source}")
        preview = r.text.replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "…"
        print(f"    内容：{preview}")

    print("\n" + "-" * 78)
    print(f"相关性判定：{'通过，可生成回答' if reliable else '不通过，应走拒答分支'}")
    print("=" * 78)


def print_compare(query: str, results) -> None:
    """对比 RRF 融合排序与 Rerank 精排排序的差异。"""
    by_fusion = sorted(results, key=lambda r: r.fusion_rank)
    print(f"\n查询：{query}")
    print("=" * 78)
    print(f"{'RRF 顺序':<34} | {'Rerank 顺序':<34}")
    print("-" * 78)
    for i, (a, b) in enumerate(zip(by_fusion, results), start=1):
        left = f"{i}. {a.citation[:30]}"
        right = f"{i}. {b.citation[:30]}"
        flag = "" if a.source == b.source and a.chunk_index == b.chunk_index else "  <== 位次变化"
        print(f"{left:<34} | {right:<34}{flag}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="检索效果检查")
    parser.add_argument("query", nargs="?", help="查询内容")
    parser.add_argument("--top-k", type=int, default=None, help="返回条数")
    parser.add_argument("--type", dest="doc_type", default=None, help="限定文档类型")
    parser.add_argument("--compare", action="store_true", help="对比融合与精排排序")
    args = parser.parse_args()

    retriever = get_retriever()
    top_k = args.top_k or settings.final_top_k

    queries = [args.query] if args.query else []
    if not queries:
        print("进入交互模式，输入空行退出。")
        while True:
            try:
                q = input("\n查询> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            queries.append(q)

    for q in queries:
        results = retriever.retrieve(q, top_k=top_k, doc_type=args.doc_type)
        if args.compare:
            print_compare(q, results)
        print_results(q, results, top_k)


if __name__ == "__main__":
    main()
