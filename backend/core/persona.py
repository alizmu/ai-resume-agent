"""人格配置（Persona）：把"服务谁"这件事彻底从代码里抽出来，变成数据。

为什么需要它：
- 之前的实现把候选人姓名、项目白名单、系统提示词都写死在代码里，
  既泄漏隐私，也让系统只能服务固定一个人。
- 现在这些都放进 `knowledge/persona.json`（数据），代码只认"人格配置"这个抽象。
  换一个候选人，只改 persona.json + 替换 knowledge/raw/ 并重新入库，框架零改动。

这是把"我的个人小玩具"升级为"可复用 RAG 框架"的关键一步：
同一套代码，靠不同 persona 数据就能服务不同人 / 不同简历。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# backend/core/persona.py -> parents[2] = 项目根目录
PERSONA_PATH = Path(__file__).resolve().parents[2] / "knowledge" / "persona.json"


@dataclass
class Persona:
    """一个候选人的人格配置。全部来自数据，代码里不留任何个人信息。"""

    name: str = "候选人"  # 候选人姓名，用于拒答判定与提示词
    brand: str = "RAG 简历助手"  # 前端 / 接口展示名（可泛型）
    project_anchors: list[str] = field(
        default_factory=list
    )  # 已知项目清单（白名单），如 ["校园智能问答 RAG 系统"]
    project_keywords: list[str] = field(
        default_factory=list
    )  # 用于 multi-turn 指代消解的项目关键词，如 ["RAG", "智能问答"]


_persona: Persona | None = None


def get_persona() -> Persona:
    """读取 persona.json 并缓存。文件缺失时回退到中性默认值，保证服务可启动。"""
    global _persona
    if _persona is not None:
        return _persona
    try:
        with PERSONA_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        _persona = Persona(
            name=data.get("name", "候选人"),
            brand=data.get("brand", "RAG 简历助手"),
            project_anchors=data.get("project_anchors", []),
            project_keywords=data.get("project_keywords", []),
        )
    except FileNotFoundError:
        _persona = Persona()
    return _persona


def reload_persona() -> Persona:
    """丢弃缓存重新加载（换 persona 数据后调用）。

    同时刷新 settings.subject_name，保证 query_rewrite、retriever 等
    所有读取候选人姓名的链路都使用新 persona。
    """
    global _persona
    _persona = None
    persona = get_persona()
    # 保持所有 consumers 读取到的姓名一致
    from backend.config.settings import settings

    settings.subject_name = persona.name
    return persona
