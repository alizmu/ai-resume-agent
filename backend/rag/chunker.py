"""分类切片策略。

为什么按文档类型分开处理，而不是全局统一 chunk_size：

- qa（问答对）：一问一答是一个完整的语义单元。若在答案中间切断，检索到的片段
  会变成半个回答，模型据此生成的答案必然残缺。所以给它最大的块（2000），
  保证整对问答完整落入同一个 chunk。
- projects（项目文档）：按标题层级切，每个二级标题下的内容独立成块。
  面试官问「某项目你做了什么」时，命中的应该是完整的某个环节，而不是跨环节的杂烩。
- profile（个人档案）：字段密集、彼此独立，小块（500）反而检索更准。
- notes / jd：通用递归切分。

另外，每个 chunk 都带上标题路径（heading_path）作为元数据。它的作用有两个：
一是给模型提供上下文线索（这段来自「政务大厅项目 > 技术实施」），
二是支持按来源过滤，让「项目相关问题」只在项目文档里检索。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger

from backend.config.settings import settings
from backend.rag.loader import RawDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_QA_SPLIT_RE = re.compile(r"^\s*(#{1,6}\s+.*|\*\*Q[:：]?.*|Q[:：].*)$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    doc_type: str
    source: str
    title: str
    heading_path: str
    chunk_index: int

    @property
    def citation(self) -> str:
        """给前端展示的引用来源。"""
        path = self.heading_path or self.title
        return f"{self.title} · {path}" if path != self.title else self.title

    def to_payload(self) -> dict:
        return {
            "text": self.text,
            "doc_type": self.doc_type,
            "source": self.source,
            "title": self.title,
            "heading_path": self.heading_path,
            "chunk_index": self.chunk_index,
        }


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题切分，返回 (标题路径, 正文) 列表。"""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    current: list[str] = []
    current_path: list[str] = []

    def flush():
        if current:
            body = "\n".join(current).strip()
            if body:
                sections.append((" > ".join(current_path), body))

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            current = []
            level = len(m.group(1))
            heading = m.group(2).strip()
            stack = stack[: level - 1]
            stack.append(heading)
            current_path = stack.copy()
        else:
            current.append(line)
    flush()
    return sections


def _split_qa_pairs(text: str) -> list[str]:
    """问答对按「问」的边界切，保证一问一答不被拆开。"""
    matches = list(_QA_SPLIT_RE.finditer(text))
    if len(matches) < 2:
        return [text]

    blocks: list[str] = []
    head = text[: matches[0].start()].strip()
    if head:
        blocks.append(head)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.start() : end].strip()
        if block:
            blocks.append(block)
    return blocks


def _hard_split(text: str, size: int) -> list[str]:
    """单个段落过长时按长度硬切（按句子边界优先）。"""
    pieces: list[str] = []
    while len(text) > size:
        window = text[:size]
        cut = max(
            window.rfind("。"),
            window.rfind("；"),
            window.rfind("\n"),
            window.rfind(" "),
        )
        cut = cut + 1 if cut > size * 0.5 else size
        pieces.append(text[:cut].strip())
        text = text[cut:]
    if text.strip():
        pieces.append(text.strip())
    return pieces


def _pack_paragraphs(text: str, size: int, overlap: int) -> list[str]:
    """贪心地把段落打包进不超过 size 的块，块间保留 overlap 个字符的重叠。"""
    if len(text) <= size:
        return [text]

    paragraphs: list[str] = []
    for p in text.split("\n\n"):
        p = p.strip()
        if not p:
            continue
        if len(p) > size:
            paragraphs.extend(_hard_split(p, size))
        else:
            paragraphs.append(p)

    chunks: list[str] = []
    cur = ""
    for p in paragraphs:
        candidate = f"{cur}\n\n{p}" if cur else p
        if len(candidate) <= size:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)

    # 补重叠：让相邻块之间保留上一段的尾巴，避免跨块的关键信息被切断
    if overlap > 0 and len(chunks) > 1:
        merged: list[str] = [chunks[0]]
        for prev, nxt in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            merged.append(f"{tail}\n{nxt}" if tail else nxt)
        chunks = merged
    return chunks


def _merge_tiny(chunks: list[str], min_size: int) -> list[str]:
    """过小的块信息量不足，合并到相邻块。"""
    if len(chunks) <= 1:
        return chunks
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < min_size:
            merged[-1] = f"{merged[-1]}\n\n{c}"
        else:
            merged.append(c)
    return merged


def chunk_document(doc: RawDocument) -> list[Chunk]:
    size = settings.chunk_size_by_type.get(doc.doc_type, settings.chunk_size)
    overlap = settings.chunk_overlap

    if doc.doc_type == "qa":
        blocks = _split_qa_pairs(doc.text)
        sections: list[tuple[str, str]] = [("", b) for b in blocks]
    else:
        sections = _split_into_sections(doc.text)
        if not sections:
            sections = [("", doc.text)]

    chunks: list[Chunk] = []
    for heading_path, body in sections:
        pieces = _merge_tiny(
            _pack_paragraphs(body, size, overlap), settings.min_chunk_size
        )
        for piece in pieces:
            chunks.append(
                Chunk(
                    text=piece,
                    doc_type=doc.doc_type,
                    source=doc.source,
                    title=doc.title,
                    heading_path=heading_path,
                    chunk_index=len(chunks),
                )
            )
    return chunks


def chunk_documents(docs: list[RawDocument]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        cs = chunk_document(doc)
        all_chunks.extend(cs)
        logger.debug(f"{doc.source}: {len(cs)} 块")

    counter: dict[str, int] = {}
    for c in all_chunks:
        counter[c.doc_type] = counter.get(c.doc_type, 0) + 1
    logger.info(f"切片完成，共 {len(all_chunks)} 块 -> {counter}")
    return all_chunks
