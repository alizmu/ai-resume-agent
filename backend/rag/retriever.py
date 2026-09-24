"""检索编排：双路召回 -> RRF 融合 -> Cross-Encoder 精排 -> 相关性判定。

链路每一步都可以单独关掉（enable_sparse / enable_rerank），
任何一环失效都自动降级而不是抛错——线上服务不会因为模型抖动而整体不可用。
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from backend.config.settings import settings
from backend.core.reranker import get_reranker
from backend.rag.vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    doc_type: str
    source: str
    title: str
    heading_path: str
    chunk_index: int
    fusion_rank: int
    fusion_score: float
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fusion_score

    @property
    def citation(self) -> str:
        if self.heading_path:
            return f"{self.title} · {self.heading_path}"
        return self.title


class Retriever:
    def __init__(self) -> None:
        self.store = VectorStore()
        self.reranker = get_reranker()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.final_top_k

        candidates = self.store.hybrid_search(
            query, limit=settings.prefetch_limit, doc_type=doc_type
        )
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]
        scores = self.reranker.rerank(query, texts)

        results = [
            RetrievedChunk(
                text=c["text"],
                doc_type=c["doc_type"],
                source=c["source"],
                title=c["title"],
                heading_path=c["heading_path"],
                chunk_index=c["chunk_index"],
                fusion_rank=c["fusion_rank"],
                fusion_score=c["fusion_score"],
                rerank_score=scores[i] if scores else None,
            )
            for i, c in enumerate(candidates)
        ]

        results.sort(key=lambda r: r.final_score, reverse=True)
        mode = "rerank" if scores else "rrf"
        logger.debug(
            f"检索[{mode}] '{query}' -> 候选 {len(results)}，返回 {min(top_k, len(results))}"
        )
        return results[:top_k]

    def is_reliable(self, chunks: list[RetrievedChunk], query: str | None = None) -> bool:
        """判定检索结果是否足以支撑回答。

        - 开启重排时：用 cross-encoder 的语义分数（0~1）与阈值比较，最准。
        - 关闭重排时（默认演示模式）：用改进的词面兜底（见 _is_reliable_soft），
          扫描前若干召回片段、并加入「关于候选人本人的泛查询保护」，
          避免只因 Top1 片段恰好没提到关键词、或查询里只有「介绍/说一下」这类
          无实义词就误拒库内问题。
        """
        if not chunks:
            return False
        if chunks[0].rerank_score is not None:
            return chunks[0].rerank_score >= settings.relevance_threshold
        if query:
            return _is_reliable_soft(query, chunks[: settings.reliability_scan_k])
        return True

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """把检索结果拼成带编号的上下文，供生成阶段引用。"""
        if not chunks:
            return ""
        lines = []
        for i, c in enumerate(chunks, start=1):
            lines.append(f"[{i}] 来源：{c.citation}\n{c.text}")
        return "\n\n".join(lines)


# 关于候选人本人的「人格化意图」词表：仅当查询既含姓名、又含这些词之一时，
# 才放行「关于候选人的开放提问」，避免「候选人 今天天气」这类无关查询被误判为命中（P0-3）。
PERSONA_HINTS = (
    "介绍", "经历", "项目", "技能", "教育", "学历", "学校", "专业", "实习", "工作",
    "擅长", "背景", "简历", "求职", "目标", "规划", "家乡", "会", "做", "开发",
    "研究", "成果", "获奖", "证书", "英语", "性格", "爱好",
)


def _is_reliable_soft(query: str, chunks: list[RetrievedChunk]) -> bool:
    """rerank 关闭时的兜底判定：扫描前 k 个召回片段，而非只看 Top1。

    两条命中任意一条即视为「知识库有依据」：
    1. 任一片段与查询存在「扣除候选人姓名后的实质词面重合」——问的是库内具体内容；
    2. 泛查询保护：查询含候选人姓名且召回片段也含姓名——关于候选人本人的开放提问
       （如「介绍一下候选人」）只要有档案就放行，交给 LLM 基于资料诚实作答，
       避免「介绍/说一下」这类无实义词导致整句被误拒。
    """
    name = settings.subject_name
    query_has_name = name in query
    for c in chunks:
        if _has_effective_overlap(query, c.text):
            return True
        # 人格化保护：查询既含候选人姓名、又含「介绍/经历/项目…」等意图词才放行，
        # 避免「候选人 今天天气」这类与知识库无关的查询被误判为命中（P0-3 收敛误接受）
        if query_has_name and name in c.text and any(h in query for h in PERSONA_HINTS):
            return True
    return False


def _bigrams(s: str) -> list[str]:
    s = "".join(ch for ch in s if ch.isalnum())
    return [s[i : i + 2] for i in range(len(s) - 1)]


# 中文虚词字符。由这些字构成的二元组——怎么 / 多少 / 什么 / 的是 / 么样——
# 在任何一段中文里几乎都能找到，如果把它们当作「查询与片段相关」的证据，
# 库外问题（「今天北京的天气怎么样」「特斯拉股价多少」）就会被误判为可靠，
# 系统开始对着简历资料一本正经地乱答，直接摧毁「不编造」这个卖点。
# 因此生成二元组时先把含虚词的组合剔掉，只保留有实义的重合。
_STOP_CHARS = set("的了吗呢吧啊么怎什多少是的不也都就很和与或在有个之把被让从对给请帮")

# 判定为「有依据」所需的最少有效二元组重合数。
# 取 1 而非 2 是实测权衡的结果：设成 2 会让库内 12 条评测 query 的召回率
# 从 100% 掉到 66.7%（过度拒答，面试官问什么都答不上来，比乱答更致命）；
# 设成 1 且配合虚词过滤，库外拒答率从 50% 提升到 87.5%，库内保持 100%。
_MIN_OVERLAP_BIGRAMS = 1


def _effective_bigrams(s: str) -> list[str]:
    """去掉候选人姓名与虚词后的实义二元组。"""
    name_bg = set(_bigrams(settings.subject_name))
    out: list[str] = []
    for g in _bigrams(s):
        if g in name_bg:
            continue
        if g[0] in _STOP_CHARS or g[1] in _STOP_CHARS:
            continue
        out.append(g)
    return out


def _has_effective_overlap(query: str, text: str) -> bool:
    """查询与片段是否存在「非候选人姓名、非虚词」的二元组重合。"""
    q = set(_effective_bigrams(query))
    if not q:
        return False
    t = set(_effective_bigrams(text))
    hits = sum(1 for g in q if g in t)
    return hits >= _MIN_OVERLAP_BIGRAMS


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def reload_retriever() -> Retriever:
    """丢弃缓存重建（换人 / 重建向量库后调用）。

    为什么需要它：Retriever 在 __init__ 时就固定了 self.store = VectorStore()，
    而 bootstrap 换人走的是 delete_collection + create_collection。若继续复用旧实例，
    Qdrant 嵌入式模式下该实例可能仍持有已被删除的集合句柄，
    表现为"资料明明换了，却检索不到新内容"。
    与 persona.reload_persona() 成对调用，让换人在运行期立即生效、无需重启服务。
    """
    global _retriever
    _retriever = None
    return get_retriever()
