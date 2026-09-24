"""Agent 编排（LangGraph StateGraph）。

链路：rewrite(Query改写) -> retrieve(混合检索+重排) -> judge(相关性判定/拒答) -> generate(生成) -> cite(引用注入)

生成节点在「无 LLM key」时降级为规则拼接，保证整条链路可端到端运行与演示；
配置 key 后自动切换真实模型，代码零改动。
"""

from __future__ import annotations

import re
import time
from typing import TypedDict

from loguru import logger

from backend.config.settings import settings
from backend.core.llm import get_llm_client
from backend.core.persona import get_persona
from backend.rag.retriever import get_retriever
from backend.agent.memory import ConversationMemory
from backend.agent.query_rewrite import QueryRewriter

# langgraph 延迟导入：未安装时不阻塞模块加载，安装后 build_graph 才用到
try:
    from langgraph.graph import StateGraph, END, START  # noqa: F401
except Exception:  # pragma: no cover
    StateGraph = END = START = None  # type: ignore


class AgentState(TypedDict, total=False):
    session_id: str
    query: str
    rewritten: str
    retrieved: list
    context: str
    answer: str
    citations: list
    reliable: bool
    memory: ConversationMemory


# 会话记忆缓存：带 TTL 回收，避免长期运行后内存泄漏（P0-7）
_SESSIONS: dict[str, tuple[ConversationMemory, float]] = {}
_SESSION_TTL = 1800.0       # 30 分钟无活动则回收
_MAX_SESSIONS = 500          # 上限保护，超限时清掉最久未活动的一半


def _get_memory(session_id: str) -> ConversationMemory:
    now = time.monotonic()
    entry = _SESSIONS.get(session_id)
    if entry is not None:
        mem, _ = entry
        _SESSIONS[session_id] = (mem, now)
        _evict_expired(now)
        return mem
    if len(_SESSIONS) >= _MAX_SESSIONS:
        _evict_expired(now, force=True)
    mem = ConversationMemory(session_id=session_id)
    _SESSIONS[session_id] = (mem, now)
    return mem


def _evict_expired(now: float, force: bool = False) -> None:
    if force or len(_SESSIONS) >= _MAX_SESSIONS:
        # 超限：按最近活动时间升序，淘汰最旧的一半
        ordered = sorted(_SESSIONS.items(), key=lambda kv: kv[1][1])
        for sid, _ in ordered[: max(1, len(ordered) // 2)]:
            _SESSIONS.pop(sid, None)
    else:
        stale = [sid for sid, (_, ts) in _SESSIONS.items() if now - ts > _SESSION_TTL]
        for sid in stale:
            _SESSIONS.pop(sid, None)


# 系统提示词：姓名等人格信息全部来自 persona.json（数据），模板本身不含任何个人信息。
_PERSONA = get_persona()
SYSTEM_PROMPT = f"""你是「{_PERSONA.name} 的 AI 助手」，负责向面试官介绍{_PERSONA.name}的教育背景、项目经历与技术能力。
规则：
1. 只基于提供的【参考资料】回答，不编造知识库之外的信息。参考资料已按 [1]、[2]… 编号。
2. 回答中凡是依据某条参考资料的内容，请在对应句末用 [n] 标注来源编号（n 与资料编号一致），例如「他主导了某项目的接口联调[1]」；多条来源并列写 [1][3]。若一句综合了多条资料，可标注多处。
3. 如果参考资料不足以回答，坦诚说明「这一点我暂时没有更详细的信息」，不要硬编。
4. 回答简洁、专业，突出量化成果与技术深度。"""

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _sanitize_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    """剔除越界的 [n] 引用角标，防止「幻觉引用」。

    引用编号由 LLM 自行标注，它可能写出超出实际检索片段数的编号（如只有 5 条
    资料却标了 [7]）。前端按编号去 citations 里取原文，越界会取空或取错，
    用户看到的就是一个指向不存在的角标——这是 RAG 系统最容易被质疑的地方。

    这里做一道确定性兜底：只保留 1..n_sources 范围内的角标，其余直接删除，
    并把被剔除的编号回传，便于在链路上观察到「模型确实编过」。
    """
    if n_sources <= 0:
        return _CITATION_RE.sub("", answer), []

    dropped: list[int] = []

    def _repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            return m.group(0)
        dropped.append(n)
        return ""

    cleaned = _CITATION_RE.sub(_repl, answer)
    # 删除角标后可能留下连续空格，收一下
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned, dropped


def rewrite_node(state: AgentState) -> dict:
    mem = state["memory"]
    rewritten = QueryRewriter().rewrite(state["query"], mem)
    return {"rewritten": rewritten}


def retrieve_node(state: AgentState) -> dict:
    retriever = get_retriever()
    chunks = retriever.retrieve(state["rewritten"], top_k=settings.final_top_k)
    context = retriever.format_context(chunks)
    citations = [
        {
            "id": i + 1,
            "source": c.source,
            "title": c.title,
            "heading": c.heading_path,
            # 去重标签：有层级标题（如「教育背景 > 本科」）用标题，否则取片段首句摘要，
            # 避免 QA 类片段因共享同一 source 标题而在前端展示成 5 个相同的「面试问答集」（P0-4）
            "label": c.heading_path
            or (c.text[:22].replace("\n", " ").strip() + ("…" if len(c.text) > 22 else "")),
            "text": c.text[:300],
        }
        for i, c in enumerate(chunks)
    ]
    return {"retrieved": chunks, "context": context, "citations": citations}


def judge_node(state: AgentState) -> dict:
    chunks = state.get("retrieved") or []
    reliable = get_retriever().is_reliable(chunks, query=state.get("rewritten", ""))
    return {"reliable": reliable}


def generate_node(state: AgentState) -> dict:
    llm = get_llm_client()
    mem = state["memory"]
    history = mem.get_context_messages()
    answer = llm.chat(
        SYSTEM_PROMPT,
        history + [{"role": "user", "content": state["rewritten"]}],
        context=state.get("context", ""),
    )
    mem.add("user", state["query"])
    mem.add("assistant", answer)
    return {"answer": answer}


def refuse_node(state: AgentState) -> dict:
    answer = "抱歉，关于这个问题我的知识库里暂时没有可靠的信息，无法准确回答。你可以补充相关素材，或换个角度提问。"
    mem = state["memory"]
    mem.add("user", state["query"])
    mem.add("assistant", answer)
    return {"answer": answer, "citations": []}


def build_graph():
    if StateGraph is None:
        raise RuntimeError("langgraph 未安装，请先 pip install langgraph")
    g = StateGraph(AgentState)
    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("judge", judge_node)
    g.add_node("generate", generate_node)
    g.add_node("refuse", refuse_node)
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "judge")
    g.add_conditional_edges(
        "judge",
        lambda s: "generate" if s["reliable"] else "refuse",
        {"generate": "generate", "refuse": "refuse"},
    )
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile()


_app = None


def get_agent():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def build_planner():
    """仅编排到 judge 的轻量图：rewrite -> retrieve -> judge -> END。

    用于流式主链路：用 LangGraph 真实驱动检索编排与「可靠/拒答」路由，
    生成阶段交由 _stream_generate 做 SSE token 流式，避免把 LLM 生成塞进图内
    （自定义 LLM 客户端无法被 LangGraph 的原生 token 流捕获）。
    """
    if StateGraph is None:
        raise RuntimeError("langgraph 未安装，请先 pip install langgraph")
    g = StateGraph(AgentState)
    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("judge", judge_node)
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "judge")
    g.add_edge("judge", END)
    return g.compile()


_planner = None


def get_planner():
    global _planner
    if _planner is None:
        _planner = build_planner()
    return _planner


def run(session_id: str, query: str) -> dict:
    """端到端执行一轮对话，返回答案、引用、可靠性与改写后的查询。"""
    agent = get_agent()
    mem = _get_memory(session_id)
    result = agent.invoke(AgentState(session_id=session_id, query=query, memory=mem))
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "reliable": result.get("reliable", False),
        "rewritten": result.get("rewritten", query),
    }


def _cite_chip(c: dict) -> dict:
    return {
        "id": c.get("id"),
        "title": c.get("title", ""),
        "heading": c.get("heading", ""),
        "label": c.get("label", ""),
    }


def _stream_generate(mem, query, rewritten, context, citations, reliable):
    """流式生成最终回答（真实 LLM token 流式），含拒答分支与引用角标校验。"""
    answer_parts: list[str] = []
    if reliable:
        try:
            llm = get_llm_client()
            history = mem.get_context_messages()
            for delta in llm.chat_stream(
                SYSTEM_PROMPT,
                history + [{"role": "user", "content": rewritten}],
                context=context,
            ):
                if delta:
                    answer_parts.append(delta)
                    yield {"type": "token", "delta": delta}
        except Exception as e:  # noqa: BLE001
            logger.exception("生成阶段失败")
            yield {"type": "error", "stage": "generate", "message": f"{type(e).__name__}: {e}"}
        answer = "".join(answer_parts)
        if not answer.strip():
            answer = "抱歉，生成环节出现异常，没能产出回答。请稍后重试。"
            yield {"type": "token", "delta": answer}
    else:
        answer = (
            "抱歉，关于这个问题我的知识库里暂时没有可靠的信息，无法准确回答。"
            "你可以补充相关素材，或换个角度提问。"
        )
        # 拒答也按字符切分，保持流式体感一致
        for ch in answer:
            yield {"type": "token", "delta": ch}
        citations = []  # 拒答不展示被判定为不可靠的来源

    # 引用角标校验：剔除 LLM 编造的越界 [n]，避免前端出现指向不存在的角标
    cleaned, dropped = _sanitize_citations(answer, len(citations))
    if dropped:
        uniq = sorted(set(dropped))
        logger.warning(f"剔除越界引用角标 {uniq}，实际来源数 {len(citations)}")
        yield {"type": "answer_replace", "answer": cleaned, "dropped": uniq}
    answer = cleaned

    # 持久化到多轮记忆
    mem.add("user", query)
    mem.add("assistant", answer)

    yield {"type": "citations", "items": citations}
    yield {"type": "done", "answer": answer}


def run_stream(session_id: str, query: str):
    """流式执行一轮对话，yield 结构化事件字典（供 SSE 推送）。

    编排层通过 LangGraph StateGraph（get_planner）真实驱动
    rewrite -> retrieve -> judge 并据 judge 结果做「可靠/拒答」路由；
    生成阶段复用同一检索上下文做 SSE token 流式推送（保留真实流式体验）。
    LangGraph 不可用或图执行异常时，自动降级为等价的人工节点驱动，保证服务不中断。
    """
    mem = _get_memory(session_id)
    state: AgentState = {
        "session_id": session_id,
        "query": query,
        "memory": mem,
        "rewritten": query,
        "retrieved": [],
        "context": "",
        "answer": "",
        "citations": [],
        "reliable": False,
    }

    reliable, citations, context, rewritten, chunks = False, [], "", query, []

    try:
        planner = get_planner()
        result = planner.invoke(state)
        rewritten = result.get("rewritten", query)
        chunks = result.get("retrieved") or []
        citations = result.get("citations") or []
        reliable = result.get("reliable", False)
        context = result.get("context", "")
        yield {"type": "step", "stage": "rewrite", "detail": rewritten}
        yield {
            "type": "step",
            "stage": "retrieve",
            "count": len(chunks),
            "chunks": [_cite_chip(c) for c in citations],
        }
        yield {"type": "step", "stage": "judge", "reliable": reliable}
    except Exception as e:  # noqa: BLE001
        logger.exception("LangGraph 规划链路异常，降级为人工节点驱动")
        yield {"type": "error", "stage": "graph", "message": f"{type(e).__name__}: {e}"}
        try:
            state.update(rewrite_node(state))
            rewritten = state.get("rewritten", query)
            yield {"type": "step", "stage": "rewrite", "detail": rewritten}
            state.update(retrieve_node(state))
            chunks = state.get("retrieved") or []
            citations = state.get("citations") or []
            yield {
                "type": "step",
                "stage": "retrieve",
                "count": len(chunks),
                "chunks": [_cite_chip(c) for c in citations],
            }
            state.update(judge_node(state))
            reliable = state.get("reliable", False)
            context = state.get("context", "")
            yield {"type": "step", "stage": "judge", "reliable": reliable}
        except Exception as e2:  # noqa: BLE001
            logger.exception("降级检索亦失败")
            reliable = False
            chunks, citations = [], []

    yield from _stream_generate(mem, query, rewritten, context, citations, reliable)
