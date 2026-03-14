from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.graph.state import ConversationState
from app.llm.structured_output import structured_llm_call

logger = logging.getLogger(__name__)

ATTITUDE_JUDGE_PROMPT = """\
你是一位医药销售培训的评估专家。根据对话上下文和当前状态，判断客户当前的真实态度。

## 难度模式
{difficulty}

## 态度选项
- cautious: 审慎质疑，对新药持谨慎态度，需要更多证据
- interested: 产生兴趣，已认可部分卖点，但仍有疑问
- convinced: 基本认可，对产品整体印象良好，准备考虑试用

## 当前状态
- 已覆盖语义点: {covered_count}/{total_count}
- 当前轮次: {turn}
- 上一轮态度: {prev_attitude}
- 销售表达质量: {quality_summary}

## 判断规则
1. 态度只能渐进式变化（cautious→interested→convinced），不能跳跃或倒退
2. 仅凭语义点覆盖数量不够，还要看销售的表达质量（数据是否准确、是否回应了客户问题）
3. 如果销售用了"大概""好像"等模糊词汇，即使覆盖了语义点，也不应轻易提升态度
4. hard 模式下态度转变更慢，需要高质量的数据引用和专业表达
5. 重点关注客户最新回复的语气——是认可、质疑还是不满

## 最近对话
{recent_conversation}

## 输出（严格JSON）
{{"attitude": "cautious/interested/convinced", "reason": "一句话理由"}}
"""


def _get_recent_conversation(messages: list, max_turns: int = 3) -> str:
    recent = messages[-(max_turns * 2):] if len(messages) > max_turns * 2 else messages
    lines = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            lines.append(f"销售: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"客户: {msg.content}")
        elif isinstance(msg, dict):
            role = "销售" if msg.get("role") == "user" else "客户"
            lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines) if lines else "（无对话）"


def _quality_summary(analysis: dict) -> str:
    eq = analysis.get("expression_quality")
    if not eq or not isinstance(eq, dict):
        return "无评分"
    parts = []
    for key, label in [("data_citation", "数据引用"), ("interaction", "互动")]:
        dim = eq.get(key)
        if isinstance(dim, dict) and "score" in dim:
            parts.append(f"{label}{dim['score']}/5")
    return ", ".join(parts) if parts else "无评分"


async def state_updater_node(state: ConversationState) -> dict:
    analysis = state.get("current_analysis") or {}
    coverage = dict(state["semantic_coverage"])
    evidence = dict(state["semantic_evidence"])

    any_new = False
    for point in analysis.get("analysis", []):
        pid = point.get("point_id", "")
        confidence = point.get("confidence", 0.0)
        match_level = point.get("match_level", "none")
        newly_matched = point.get("newly_matched", False)

        if (
            newly_matched
            and confidence >= settings.confidence_threshold
            and match_level == "full"
            and not coverage.get(pid, False)
        ):
            coverage[pid] = True
            evidence[pid] = {
                "turn": state["turn_count"] + 1,
                "confidence": confidence,
                "evidence": point.get("evidence", ""),
                "reasoning": point.get("reasoning", ""),
            }
            any_new = True

    # Attitude: LLM-based judgment using lite model
    prev_attitude = state.get("customer_attitude", "cautious")
    covered_count = sum(coverage.values())
    total_count = len(coverage)
    turn = state["turn_count"] + 1

    try:
        prompt = ATTITUDE_JUDGE_PROMPT.format(
            difficulty=state.get("difficulty", "normal"),
            covered_count=covered_count,
            total_count=total_count,
            turn=turn,
            prev_attitude=prev_attitude,
            quality_summary=_quality_summary(analysis),
            recent_conversation=_get_recent_conversation(state["messages"]),
        )
        result = await structured_llm_call(
            system=prompt,
            user="请判断客户当前态度。",
            temperature=0.0,
            model=settings.lite_model,
            retries=1,
            default={"attitude": prev_attitude, "reason": "判断失败，保持上一轮态度"},
        )
        attitude = result.get("attitude", prev_attitude)
        # Validate: only allow valid values and no regression
        valid_attitudes = ["cautious", "interested", "convinced"]
        if attitude not in valid_attitudes:
            attitude = prev_attitude
        # Enforce no regression
        prev_idx = valid_attitudes.index(prev_attitude)
        new_idx = valid_attitudes.index(attitude)
        if new_idx < prev_idx:
            attitude = prev_attitude
        logger.info("Attitude judge: %s → %s (reason: %s)", prev_attitude, attitude, result.get("reason", ""))
    except Exception:
        logger.exception("Attitude LLM judge failed, falling back to rule-based")
        # Fallback to simple rule
        if covered_count >= 3:
            attitude = "convinced"
        elif covered_count >= 2:
            attitude = "interested"
        else:
            attitude = "cautious"

    # Stagnation tracking
    stagnation = state.get("stagnation_count", 0)
    if not any_new:
        stagnation += 1
    else:
        stagnation = 0

    # Phase determination
    max_turns = state.get("max_turns", settings.max_turns)

    if all(coverage.values()):
        phase = "wrapping_up"
    elif turn >= max_turns or stagnation >= 4:
        phase = "force_wrapping_up"
    else:
        phase = "active"

    return {
        "semantic_coverage": coverage,
        "semantic_evidence": evidence,
        "customer_attitude": attitude,
        "phase": phase,
        "turn_count": turn,
        "stagnation_count": stagnation,
    }
