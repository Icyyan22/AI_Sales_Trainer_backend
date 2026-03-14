from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.graph.state import ConversationState
from app.llm.structured_output import structured_llm_call
from app.models.scenario import load_scenario
from app.prompts.registry import get_prompt


DIFFICULTY_INSTRUCTIONS = {
    "easy": """### 难度：简单模式
客户态度友善，对新药持开放积极态度。
- 倾向使用 direct_question 和 acknowledge_and_pivot 策略
- 很少质疑数据，容易被说服
- 主动引导话题帮助销售覆盖语义点
- 态度渐变更快：1个语义点覆盖就转为 interested""",
    "normal": """### 难度：标准模式
客户态度审慎专业，需要看到证据才会认可。
- 可使用 clinical_scenario、acknowledge_and_pivot、challenge 策略
- 偶尔质疑数据，要求看到具体证据
- 用临床场景间接引导未覆盖语义点""",
    "hard": """### 难度：困难模式
客户态度挑剔，经验丰富，不轻易认可任何说法。
- 优先使用 challenge、competitor_comparison、data_deep_dive 策略
- 频繁质疑数据（"这是ITT分析还是PP分析？亚组数据呢？"）
- 拿竞品做对比（"隔壁XX药企也说他们效果好，你们有头对头研究吗？"）
- 追问细节，要求销售提供更深层的证据
- 态度渐变更慢：需要3个语义点全部覆盖才转为 convinced
- 即使认可某个卖点，也会提出额外顾虑""",
}

ATTITUDE_DESCRIPTIONS = {
    "cautious": "审慎质疑，对新药持开放但谨慎的态度，需要看到更多证据",
    "interested": "产生兴趣，已经认可部分卖点，但仍有疑问需要解答",
    "convinced": "基本认可，对产品整体印象良好，准备考虑在科室试用",
}


def _build_messages_for_llm(messages: list, max_recent_turns: int | None = None) -> list[dict]:
    if max_recent_turns is None:
        max_recent_turns = settings.max_context_turns

    all_msgs = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            all_msgs.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            all_msgs.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, dict):
            all_msgs.append(msg)

    max_messages = max_recent_turns * 2
    if len(all_msgs) <= max_messages + 1:
        return all_msgs

    # Keep opening message + most recent N turns
    return [all_msgs[0]] + all_msgs[-max_messages:]


def _format_coverage(coverage: dict, scenario) -> tuple[str, str]:
    covered = []
    uncovered = []
    sp_map = {sp.id: sp.name for sp in scenario.semantic_points}
    for pid, is_covered in coverage.items():
        label = f"{pid}({sp_map.get(pid, pid)})"
        if is_covered:
            covered.append(label)
        else:
            uncovered.append(label)
    return (
        ", ".join(covered) if covered else "无",
        ", ".join(uncovered) if uncovered else "无",
    )


STREAMING_OUTPUT_FORMAT = """
## 输出格式
请严格按照以下格式输出（先思考，后回复）：

<think>
{"target_point": "SP2", "strategy": "clinical_scenario", "attitude": "cautious"}
</think>
你作为客户的自然语言回复（直接输出文字，不要包裹任何标签）
"""


def build_persona_prompt(
    state: ConversationState, *, streaming: bool = False, memory_context: str = ""
) -> tuple[str, list[dict]]:
    """Build the system prompt and chat messages for the persona LLM call.

    Returns (system_prompt, chat_messages).
    """
    scenario = load_scenario(state["scenario_id"])
    cp = scenario.customer_profile
    difficulty = state.get("difficulty", "normal")
    phase = state["phase"]

    covered_text, uncovered_text = _format_coverage(
        state["semantic_coverage"], scenario
    )

    concerns_text = "\n".join(f"- {c}" for c in cp.concerns)
    analysis_text = json.dumps(
        state.get("current_analysis", {}), ensure_ascii=False, indent=2
    )

    wrap_up = ""
    if phase == "wrapping_up":
        wrap_up = "## 特别指令\n所有语义点已覆盖，请使用 summarize_and_close 策略，礼貌地表达对产品的初步认可并自然结束对话。"
    elif phase == "force_wrapping_up":
        wrap_up = "## 特别指令\n已达到最大轮次限制，请礼貌地结束对话。即使还有未覆盖的语义点，也需要收尾。"

    prompt_template = get_prompt("strategist_persona", settings.prompt_version)
    system_prompt = prompt_template.format(
        product_name=scenario.product.name,
        customer_name=cp.name,
        customer_role=cp.role,
        customer_organization=cp.organization,
        customer_background=cp.background,
        customer_speaking_style=cp.speaking_style,
        customer_concerns=concerns_text,
        covered_points=covered_text,
        uncovered_points=uncovered_text,
        turn_count=state["turn_count"],
        customer_attitude=state["customer_attitude"],
        phase=phase,
        stagnation_count=state.get("stagnation_count", 0),
        evaluator_result=analysis_text,
        difficulty_instruction=DIFFICULTY_INSTRUCTIONS.get(difficulty, ""),
        attitude_description=ATTITUDE_DESCRIPTIONS.get(
            state["customer_attitude"], ""
        ),
        wrap_up_instruction=wrap_up,
    )

    if memory_context:
        system_prompt += f"\n\n{memory_context}\n"

    # Replace output format section based on mode
    if streaming:
        system_prompt = system_prompt.replace(
            system_prompt[system_prompt.rfind("## 输出格式"):],
            STREAMING_OUTPUT_FORMAT,
        )

    chat_messages = _build_messages_for_llm(state["messages"])
    return system_prompt, chat_messages


async def strategist_persona_node(state: ConversationState) -> dict:
    memory_context = state.get("_memory_context", "")
    system_prompt, chat_messages = build_persona_prompt(state, memory_context=memory_context)

    result = await structured_llm_call(
        system=system_prompt,
        messages=chat_messages,
        temperature=settings.persona_temperature,
        default={
            "thinking": {"target_point": None, "strategy": "acknowledge_and_pivot", "attitude": state["customer_attitude"]},
            "response": "嗯，你说的有道理。还有其他方面要补充吗？",
        },
    )

    response_text = result.get("response", "嗯，你继续说。")
    thinking = result.get("thinking", {})

    update: dict = {
        "messages": [AIMessage(content=response_text)],
        "current_strategy": thinking,
    }

    if state["phase"] in ("wrapping_up", "force_wrapping_up"):
        update["phase"] = "completed"

    return update
