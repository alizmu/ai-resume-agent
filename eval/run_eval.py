"""离线检索评测 + 消融实验。

为什么先做这套「不依赖 LLM」的指标：
RAGAS 那类基于 LLM 的评测要调 API、有成本、结果还会波动。而检索环节的质量是
可以用确定性指标衡量的——只要标注了「期望命中的片段包含哪些关键词」，就能算出
Recall@K 和 MRR，跑一次几百毫秒，结果完全可复现。

真正有价值的是消融实验：分别关掉稀疏路和重排，看指标掉多少。
这组对比数据既是调优依据，也是简历上最硬的成果证明。

用法：
    python -m eval.run_eval                 # 跑全量对比
    python -m eval.run_eval --verbose       # 打印每条查询的命中情况
    python -m eval.run_eval --configs full  # 只跑完整配置
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import settings  # noqa: E402
from backend.rag.retriever import get_retriever  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class EvalItem:
    id: int
    category: str
    query: str
    must_include: list[str]
    should_refuse: bool = False


@dataclass
class ConfigResult:
    name: str
    sparse: bool
    rerank: bool
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    n: int = 0
    latency_ms: float = 0.0
    failed: list[dict] = field(default_factory=list)
    refuse_scores: list[tuple[str, float]] = field(default_factory=list)
    # 重排是否真的生效。模型缺失（未打包 / 离线）时 Reranker 会静默降级为 RRF，
    # 此时「完整（+重排）」这组指标其实等于「稠密+稀疏」，若不加区分地写进结论，
    # 消融对比就会失真——README 上的数字就是这么来的。
    rerank_active: bool = False


def load_eval_set(path: Path = EVAL_SET) -> list[EvalItem]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(
                EvalItem(
                    id=d["id"],
                    category=d.get("category", ""),
                    query=d["query"],
                    must_include=d.get("must_include", []),
                    should_refuse=d.get("should_refuse", False),
                )
            )
    return items


def is_hit(chunk_text: str, must_include: list[str]) -> bool:
    return any(k in chunk_text for k in must_include)


def evaluate_config(
    items: list[EvalItem],
    retriever,
    name: str,
    sparse: bool,
    rerank: bool,
    top_k: int,
) -> ConfigResult:
    settings.enable_sparse = sparse
    settings.enable_rerank = rerank

    # 校验重排模型是否真的可用：不可用时应显式标注，而不是静默按 RRF 出数
    rerank_active = False
    if rerank:
        try:
            rerank_active = retriever.reranker.model is not None
        except Exception:  # pragma: no cover
            rerank_active = False
        if not rerank_active:
            print(
                "  ⚠️  重排模型不可用，本组已降级为 RRF 融合排序，"
                "指标不等同于真正的「完整（+重排）」配置"
            )

    res = ConfigResult(name=name, sparse=sparse, rerank=rerank, rerank_active=rerank_active)
    scored = [it for it in items if it.must_include]
    res.n = len(scored)

    hits_1 = hits_3 = hits_5 = 0
    reciprocal_sum = 0.0
    total_ms = 0.0

    for item in scored:
        t0 = time.perf_counter()
        chunks = retriever.retrieve(item.query, top_k=top_k)
        total_ms += (time.perf_counter() - t0) * 1000

        rank = 0
        for i, c in enumerate(chunks, start=1):
            if is_hit(c.text, item.must_include):
                rank = i
                break

        if rank == 1:
            hits_1 += 1
        if rank and rank <= 3:
            hits_3 += 1
        if rank and rank <= 5:
            hits_5 += 1
        reciprocal_sum += 1.0 / rank if rank else 0.0

        if not rank:
            got = [c.citation for c in chunks[:3]]
            res.failed.append(
                {"id": item.id, "query": item.query, "category": item.category, "got": got}
            )

    # 拒答类：记录最高分，用于校准相关性阈值
    for item in items:
        if not item.should_refuse:
            continue
        chunks = retriever.retrieve(item.query, top_k=top_k)
        top = chunks[0].final_score if chunks else 0.0
        res.refuse_scores.append((item.query, top))

    if res.n:
        res.recall_at_1 = hits_1 / res.n
        res.recall_at_3 = hits_3 / res.n
        res.recall_at_5 = hits_5 / res.n
        res.mrr = reciprocal_sum / res.n
        res.latency_ms = total_ms / res.n
    return res


def print_table(results: list[ConfigResult]) -> None:
    print("\n" + "=" * 74)
    print(f"{'配置':<26}{'R@1':>10}{'R@3':>10}{'R@5':>10}{'MRR':>10}{'耗时':>10}")
    print("-" * 74)
    for r in results:
        name = r.name + (" ⚠️降级" if (r.rerank and not r.rerank_active) else "")
        print(
            f"{name:<26}{r.recall_at_1:>10.1%}{r.recall_at_3:>10.1%}"
            f"{r.recall_at_5:>10.1%}{r.mrr:>10.3f}{r.latency_ms:>9.0f}ms"
        )
    print("=" * 74)
    degraded = [r.name for r in results if r.rerank and not r.rerank_active]
    if degraded:
        print("⚠️  以下配置的重排未真正生效，指标等同 RRF 融合，请勿据此下「重排有效」的结论：")
        for n in degraded:
            print(f"      - {n}")
        print("=" * 74)


def main() -> None:
    parser = argparse.ArgumentParser(description="检索效果评测与消融实验")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true", help="打印未命中明细")
    parser.add_argument(
        "--configs",
        default="all",
        help="all=跑三组消融；full=只跑完整配置",
    )
    args = parser.parse_args()

    items = load_eval_set()
    print(f"评测集：{len(items)} 条（其中拒答类 {sum(i.should_refuse for i in items)} 条）")

    retriever = get_retriever()

    if args.configs == "full":
        matrix = [("完整（稠密+稀疏+重排）", True, True)]
    else:
        matrix = [
            ("A 仅稠密", False, False),
            ("B 稠密+稀疏", True, False),
            ("C 完整（+重排）", True, True),
        ]

    results = []
    for name, sparse, rerank in matrix:
        print(f"\n正在评测：{name} ...")
        r = evaluate_config(items, retriever, name, sparse, rerank, args.top_k)
        results.append(r)

    print_table(results)

    if len(results) > 1:
        base, full = results[0], results[-1]
        print(
            f"\n相对基线（{base.name}）：Recall@5 "
            f"{base.recall_at_5:.1%} -> {full.recall_at_5:.1%}"
            f"（{full.recall_at_5 - base.recall_at_5:+.1%}），"
            f"MRR {base.mrr:.3f} -> {full.mrr:.3f}（{full.mrr - base.mrr:+.3f}）"
        )

    if args.verbose:
        print("\n未命中明细：")
        for r in results:
            print(f"\n--- {r.name}（{len(r.failed)} 条未命中）---")
            for f in r.failed:
                print(f"  #{f['id']} [{f['category']}] {f['query']}")
                for g in f["got"]:
                    print(f"        实际命中：{g}")

    print("\n拒答类查询的最高分（用于校准 RELEVANCE_THRESHOLD）：")
    for r in results:
        if not r.refuse_scores:
            continue
        scores = [s for _, s in r.refuse_scores]
        print(
            f"  {r.name}：最高 {max(scores):.4f} / 最低 {min(scores):.4f} / 均值 {sum(scores)/len(scores):.4f}"
        )
        for q, s in r.refuse_scores:
            print(f"      {s:>8.4f}  {q}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = [
        {
            "name": r.name,
            "sparse": r.sparse,
            "rerank": r.rerank,
            "rerank_active": r.rerank_active,
            "recall_at_1": r.recall_at_1,
            "recall_at_3": r.recall_at_3,
            "recall_at_5": r.recall_at_5,
            "mrr": r.mrr,
            "latency_ms": r.latency_ms,
            "failed": r.failed,
        }
        for r in results
    ]
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已保存：{out}")


if __name__ == "__main__":
    main()
