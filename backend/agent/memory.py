"""多轮对话记忆：滑动窗口 + 摘要压缩 + 长期关注点。

设计要点（对应简历「多轮记忆」）：
- 滑动窗口：保留最近 N 轮（按 token 预算），超出部分压缩进 summary。
- 摘要压缩：无 LLM 时按规则拼接（保留原文，降级可用）；有 LLM 时可升级为语义摘要。
- 长期记忆：记录提问者关注点（跨会话），持久化到文件。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from backend.config.settings import settings


@dataclass
class Message:
    role: str  # user | assistant
    content: str
    ts: float = field(default_factory=time.time)

    def tokens(self) -> int:
        # 粗略估算：中文约 4 字符/token
        return max(1, len(self.content) // 4)


class ConversationMemory:
    def __init__(
        self, session_id: str = "default", window_turns: int = 6, max_tokens: int = 2000
    ) -> None:
        self.session_id = session_id
        self.window_turns = window_turns
        self.max_tokens = max_tokens
        self.history: list[Message] = []
        self.summary: str = ""
        # 长期关注点（跨会话累积）
        self.long_term_file = settings.knowledge_raw_dir.parent / "memory" / f"long_term_{session_id}.json"
        self.long_term: dict = self._load_long_term()

    def _load_long_term(self) -> dict:
        if self.long_term_file.exists():
            try:
                return json.loads(self.long_term_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"topics": [], "last_updated": None}

    def _save_long_term(self) -> None:
        self.long_term_file.parent.mkdir(parents=True, exist_ok=True)
        self.long_term["last_updated"] = time.time()
        self.long_term_file.write_text(
            json.dumps(self.long_term, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add(self, role: str, content: str) -> None:
        self.history.append(Message(role=role, content=content))
        self._maybe_compress()
        # 记录关注点（简单：抽取用户问题到长期记忆）
        if role == "user":
            self.long_term.setdefault("topics", [])
            if content not in self.long_term["topics"]:
                self.long_term["topics"].append(content)
                self.long_term["topics"] = self.long_term["topics"][-20:]
            self._save_long_term()

    def _maybe_compress(self) -> None:
        total = sum(m.tokens() for m in self.history)
        if total <= self.max_tokens:
            return
        keep = self.history[-self.window_turns :]
        old = self.history[: -self.window_turns]
        if old:
            compressed = "\n".join(f"{m.role}: {m.content}" for m in old)
            self.summary = (self.summary + "\n" + compressed).strip() if self.summary else compressed
            self.history = keep
            logger.debug(f"记忆压缩：{len(old)} 轮并入摘要，当前窗口 {len(keep)} 轮")

    def get_context_messages(self) -> list[dict]:
        """返回给 LLM 的历史（摘要 + 最近窗口）。"""
        msgs: list[dict] = []
        if self.summary:
            msgs.append({"role": "system", "content": f"[对话历史摘要] {self.summary}"})
        for m in self.history:
            msgs.append({"role": m.role, "content": m.content})
        return msgs

    def window_text(self) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in self.history)
