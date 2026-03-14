"""LLM-based context compression for long conversations.

When conversation exceeds the sliding window, summarize earlier turns
into a compact context block. The summary is cached and only updated
incrementally when new messages fall outside the window.
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.llm.provider import llm_call

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """你是一个对话摘要助手。请将以下销售培训对话的早期内容压缩为一段简洁的摘要。

要求：
1. 保留关键事实：提到的数据、客户关心的问题、已讨论的要点
2. 保留客户态度变化
3. 不超过200字
4. 用第三人称客观描述

早期对话内容：
{conversation}

请直接输出摘要，不要加任何前缀。"""

INCREMENTAL_PROMPT = """你是一个对话摘要助手。请将已有摘要和新增对话内容合并为一段更新的摘要。

要求：
1. 保留已有摘要中的关键事实
2. 整合新增内容中的关键信息
3. 不超过200字
4. 用第三人称客观描述

已有摘要：
{existing_summary}

新增对话内容：
{new_conversation}

请直接输出更新后的摘要，不要加任何前缀。"""


def _parse_messages(messages: list) -> list[tuple[str, str]]:
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append(("user", msg.content))
        elif isinstance(msg, AIMessage):
            result.append(("assistant", msg.content))
        elif isinstance(msg, dict):
            result.append((msg.get("role", "user"), msg.get("content", "")))
    return result


def _format_msgs(msgs: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"{'销售' if role == 'user' else '客户'}: {content}"
        for role, content in msgs
    )


async def compress_context(
    messages: list,
    cached_summary: str = "",
    cached_early_count: int = 0,
    max_recent_turns: int | None = None,
) -> tuple[str, int]:
    """Summarize early turns outside the sliding window.

    Returns (summary_text, early_message_count) for caching in graph state.
    - Cache hit: early count unchanged → return cached summary (zero cost)
    - First compression: generate full summary
    - Incremental: merge new messages into existing summary
    """
    if max_recent_turns is None:
        max_recent_turns = settings.max_context_turns

    all_msgs = _parse_messages(messages)
    max_messages = max_recent_turns * 2
    # opening message at index 0 is always kept
    early_count = max(0, len(all_msgs) - 1 - max_messages)

    if early_count <= 0:
        return "", 0

    # Cache hit
    if cached_summary and early_count == cached_early_count:
        return cached_summary, early_count

    early_msgs = all_msgs[1 : 1 + early_count]

    try:
        if not cached_summary:
            # First compression: summarize all early messages
            result = await llm_call(
                system=SUMMARIZE_PROMPT.format(conversation=_format_msgs(early_msgs)),
                messages=[{"role": "user", "content": "请输出摘要。"}],
                temperature=0.0,
                model=settings.lite_model,
            )
            summary = f"[早期对话摘要]\n{result.strip()}"
            logger.info("Context compressed: %d early msgs -> %d chars", len(early_msgs), len(summary))
        else:
            # Incremental: only summarize newly slid-out messages
            new_msgs = early_msgs[cached_early_count:]
            raw_summary = cached_summary.replace("[早期对话摘要]\n", "")
            result = await llm_call(
                system=INCREMENTAL_PROMPT.format(
                    existing_summary=raw_summary,
                    new_conversation=_format_msgs(new_msgs),
                ),
                messages=[{"role": "user", "content": "请输出更新后的摘要。"}],
                temperature=0.0,
                model=settings.lite_model,
            )
            summary = f"[早期对话摘要]\n{result.strip()}"
            logger.info("Context incrementally updated: +%d msgs -> %d chars", len(new_msgs), len(summary))

        return summary, early_count
    except Exception:
        logger.exception("Context compression failed, keeping cached")
        return cached_summary, cached_early_count
