"""Query 改写：多轮对话中的指代消解与追问补全（规则版，不依赖 LLM）。

对应简历「Query 改写：指代消解与追问补全」。
为什么必须做：滑动窗口解决了「上一轮说了什么」，但「他第二段实习呢」这类追问里只有代词、
没有实体，直接拿去检索必然召回错误。本模块把代词补全为带实体的完整查询。

规则实现（无 LLM 可跑）：
- 从近期对话累积实体表（人名 / 项目名 / 技术名）
- 代词替换：他/她 -> 最近人名；那个项目/这个项目 -> 最近项目名
- 追问补全：含「第二段/第一次/上次」等但缺实体时，附加最近项目名
- 元指令追问补全：「更简短/换个说法/举例/继续说」这类追问本身不含任何实体，
  直接检索必然落空被拒答，因此补上上一轮问题作为检索主题。
有 LLM 时可升级为语义改写（更准），接口保持一致。
"""

from __future__ import annotations

import re

from loguru import logger

from backend.agent.memory import ConversationMemory
from backend.config.settings import settings
from backend.core.persona import get_persona

# 已知项目白名单与关键词均来自 persona.json（数据），不写死在代码里。
# 只从这份清单里解析「项目」实体，避免把文档正文误判为项目名。
# 注意：不在模块导入时读取 persona，而是在 QueryRewriter 实例化时动态读取，
# 这样 /api/bootstrap 换人并调用 reload_persona() 后，下一次改写会自动用新 persona。

def _get_subject() -> str:
    """当前候选人姓名（支持热更新）。

    persona.json 是候选人信息的唯一数据源；.env 中的 SUBJECT_NAME
    仅作为 persona.json 缺失时的兜底。
    """
    return get_persona().name or settings.subject_name or "候选人"


def _get_persona_for_rewrite():
    """当前 persona 数据（支持热更新）。"""
    return get_persona()


# 代词「他/她」的边界匹配。
# 不能裸用 str.replace("他", person)：那会把「其他项目」打成「其<person>项目」、
# 「他们的分工」打成「<person>们的分工」，而这两类表达在面试追问里非常高频。
# 这里只替换独立出现的代词，排除 其他/他们/他人/利他/排他/吉他 等常见复合词。
_DEICTIC_HE = re.compile(r"(?<![其利排吉无自])他(?!们|人|乡|日|杀|山|国)")
_DEICTIC_SHE = re.compile(r"(?<![其])她(?!们|人)")

# 元指令追问：只在「怎么说」上提要求，不指向任何具体实体。
# 例如「能用更简短的话重说一遍吗」「换个说法」「举个例子」「继续」。
# 这类 query 单独拿去检索没有可匹配的内容，必须回补上一轮主题。
_META_FOLLOWUP_RE = re.compile(
    r"更(简短|简洁|详细|具体|通俗|清楚)"
    r"|换个(说法|方式|角度)|换种(说法|方式)"
    r"|(再|重新)(说|讲|写)(一遍|一次|清楚|明白)"
    r"|简单(点|一点|一些|些)?(说|讲|描述)"
    r"|详细(点|一点|一些|些)?(说|讲|描述|展开)"
    r"|展开(讲|说|讲讲|说说)?(一下)?"
    r"|举(一)?个?例(子)?|总结(一)?下"
    r"|用(一句|英文|中文|大白话)"
    r"|通俗(点|一点|些)"
    r"|^(为什么|怎么做的|怎么实现|具体呢|然后呢|还有呢|继续|接着说|说人话)[？?。!！]*$"
)


class QueryRewriter:
    def __init__(self) -> None:
        # 每次实例化都重新读取 persona，保证 bootstrap 换人后立刻生效
        self.subject = _get_subject()
        persona = _get_persona_for_rewrite()
        self.known_projects = list(persona.project_anchors)
        self.project_keywords = list(persona.project_keywords)
        self.entities: dict[str, str] = {"person": self.subject, "project": ""}

    def _resolve_project_from_user(self, memory: ConversationMemory) -> str:
        """只在用户提问里解析项目实体（不扫助手回答，避免文档正文污染）。

        命中 persona 里的项目关键词时，返回该项目在白名单里的规范名称，
        供后续「那个项目 / 这个项目」指代消解使用。没有命中则返回空串。
        """
        for m in memory.history:
            if getattr(m, "role", "user") != "user":
                continue
            for kw in self.project_keywords:
                if kw in m.content and self.known_projects:
                    # 返回白名单里的主项目（第一个），避免把文档正文误当项目名
                    return self.known_projects[0]
        # 兜底：会话中从未提及任何项目时返回空串，让查询保持原样。
        return ""

    def _last_user_query(self, memory: ConversationMemory) -> str:
        """取上一轮用户问题（改写发生在当前轮入库之前，history 里只有历史轮次）。"""
        for m in reversed(memory.history):
            if getattr(m, "role", "user") == "user":
                return m.content.strip()
        return ""

    def rewrite(self, query: str, memory: ConversationMemory) -> str:
        project = self._resolve_project_from_user(memory)
        q = query.strip()
        person = self.entities["person"] or self.subject

        # 1. 代词消解：他/她 -> 主语（候选人），仅替换独立出现的代词
        q = _DEICTIC_HE.sub(person, q)
        q = _DEICTIC_SHE.sub(person, q)

        # 2. 项目指代消解
        if project:
            q = (
                q.replace("那个项目", project)
                .replace("这个项目", project)
                .replace("该项目的", f"{project}的")
            )

        # 3. 追问补全：含「第二段/第一次/上次」且缺实体时，附上最近项目名
        if project and re.search(r"第二段|第一次|上次|当时", q) and project not in q:
            q = f"{project} {q}"

        # 4. 元指令追问补全：query 只在表达形式/详略上提要求、不带任何实体时，
        #    补上上一轮问题作为检索主题，否则检索落空 → 判不可靠 → 被拒答。
        #    query 已含具体项目名时说明主题明确，不再回补，避免引入噪声。
        last_q = self._last_user_query(memory)
        if (
            last_q
            and last_q not in q
            and (not project or project not in q)
            and _META_FOLLOWUP_RE.search(q)
        ):
            q = f"{last_q} {q}"

        result = q.strip() or query
        if result != query:
            logger.debug(f"Query 改写: {query!r} -> {result!r}")
        return result
