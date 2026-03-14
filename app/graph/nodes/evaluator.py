from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from app.config import settings
from app.graph.state import ConversationState
from app.llm.structured_output import structured_llm_call
from app.models.scenario import load_scenario
from app.prompts.registry import get_prompt


def _get_last_user_message(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (
            isinstance(msg, dict) and msg.get("role") == "user"
        ):
            content = msg.content if isinstance(msg, HumanMessage) else msg["content"]
            return content
    return ""


def _get_recent_context(messages: list, max_pairs: int = 2) -> str:
    recent = messages[-(max_pairs * 2 + 1) : -1] if len(messages) > 1 else []
    lines = []
    for msg in recent:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
        else:
            role = "销售" if isinstance(msg, HumanMessage) else "客户"
            content = msg.content
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "（首轮对话）"


async def evaluator_node(state: ConversationState) -> dict:
    scenario = load_scenario(state["scenario_id"])
    last_msg = _get_last_user_message(state["messages"])
    context = _get_recent_context(state["messages"])

    coverage_text = json.dumps(state["semantic_coverage"], ensure_ascii=False)

    prompt_template = get_prompt("evaluator", settings.prompt_version)
    system_prompt = prompt_template.format(
        semantic_points_definitions=scenario.get_semantic_points_text(),
        current_coverage=coverage_text,
    )

    user_prompt = f"## 最近对话上下文\n{context}\n\n## 最新销售发言（请分析这条）\n{last_msg}"

    result = await structured_llm_call(
        system=system_prompt,
        user=user_prompt,
        temperature=settings.evaluator_temperature,
        model=settings.lite_model,
    )

    return {"current_analysis": result}
