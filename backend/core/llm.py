"""LLM 多模型抽象层（兼容 OpenAI 接口的 DeepSeek / Qwen / GLM）。

设计要点（对应简历「多模型抽象与容错降级」）：
- 统一接口 chat() / chat_stream()，业务代码不感知具体厂商。
- 未配置 API key 时自动降级为 DummyLLMClient：基于检索上下文做规则拼接，
  返回带引用的演示回答，保证整条 Agent 链路在无 key 时也能端到端跑通与演示。
- 配置 key 后 get_llm_client() 返回真实客户端，代码零改动。
"""

from __future__ import annotations

import httpx
from loguru import logger

from backend.config.settings import settings


class LLMClient:
    """生成环节统一接口。"""

    def chat(self, system: str, history: list[dict], context: str = "") -> str:
        raise NotImplementedError

    def chat_stream(self, system: str, history: list[dict], context: str = ""):
        # 默认用非流式实现兜底
        yield self.chat(system, history, context)


class OpenAICompatibleClient(LLMClient):
    """兼容 OpenAI 接口的各厂商客户端（DeepSeek / Qwen / GLM 均支持）。"""

    def __init__(self) -> None:
        from openai import OpenAI

        # default=llm_timeout 控制流式读取/首 token 等待；connect=5s 控制建连阶段。
        # 部署环境若无法访问 LLM 或网关超时，会快速失败并降级为规则生成，避免 502。
        timeout = httpx.Timeout(settings.llm_timeout, connect=5.0)
        self.client = OpenAI(
            api_key=settings.llm_api_key or "EMPTY",
            base_url=settings.llm_base_url,
            timeout=timeout,
        )
        self.model = settings.llm_model

    def _build_messages(self, system: str, history: list[dict], context: str) -> list[dict]:
        sys_content = system
        if context:
            sys_content += f"\n\n【参考资料（仅供回答，不要编造库外信息）】\n{context}"
        return [{"role": "system", "content": sys_content}, *history]

    def chat(self, system: str, history: list[dict], context: str = "") -> str:
        msgs = self._build_messages(system, history, context)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(self, system: str, history: list[dict], context: str = ""):
        msgs = self._build_messages(system, history, context)
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.warning(
                f"LLM 调用失败或超时（{settings.llm_timeout}s），降级为规则生成: {e}"
            )
            yield "（LLM 服务暂时不可用，已降级为规则生成）\n\n"
            for delta in DummyLLMClient().chat_stream(system, history, context):
                yield delta


class DummyLLMClient(LLMClient):
    """无 API key 时的降级客户端：基于检索上下文做规则拼接。"""

    def chat(self, system: str, history: list[dict], context: str = "") -> str:
        last_user = history[-1]["content"] if history else ""
        if not context:
            return (
                f"（演示模式·未配置 LLM key）关于「{last_user}」，知识库中暂未检索到可靠依据。"
                "配置 DeepSeek / Qwen / GLM 的 API key 后，我将生成完整回答。"
            )
        snippets = [s.strip() for s in context.split("\n\n") if s.strip()][:2]
        body = "\n".join(f"- {s[:200]}" for s in snippets)
        return f"（演示模式·规则生成）根据知识库检索到的信息：\n{body}"

    def chat_stream(self, system: str, history: list[dict], context: str = ""):
        """按字符切分输出，让 demo 模式也有真实流式体感。"""
        answer = self.chat(system, history, context)
        # 按字符逐个 yield，模拟真实 LLM token 间隔；
        # 真实 LLM 由 OpenAICompatibleClient 直接输出厂商原始 token。
        for ch in answer:
            yield ch


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        if settings.llm_api_key:
            logger.info(f"LLM 客户端：{settings.llm_provider}（{settings.llm_model}）")
            _client = OpenAICompatibleClient()
        else:
            logger.warning("未配置 LLM_API_KEY，降级为 DummyLLMClient（规则生成）")
            _client = DummyLLMClient()
    return _client
