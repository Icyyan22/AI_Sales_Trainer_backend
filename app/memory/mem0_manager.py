from __future__ import annotations

import logging
import os

from mem0 import Memory

from app.config import settings

logger = logging.getLogger(__name__)

_memory_instance: Memory | None = None


def get_memory() -> Memory:
    """Singleton Memory instance using our existing LiteLLM config."""
    global _memory_instance
    if _memory_instance is not None:
        return _memory_instance

    # mem0's litellm provider reads API config from environment variables
    os.environ["OPENAI_API_KEY"] = settings.llm_api_key
    if settings.llm_api_base:
        os.environ["OPENAI_BASE_URL"] = settings.llm_api_base

    # Patch litellm to recognize our proxy model as supporting function calling
    import litellm
    _orig_supports_fc = litellm.supports_function_calling
    litellm.supports_function_calling = lambda model: (
        True if model.startswith("openai/gemini") else _orig_supports_fc(model)
    )

    config = {
        "llm": {
            "provider": "litellm",
            "config": {
                "model": "openai/gemini-2.5-flash",
                "temperature": 0.0,
                "max_tokens": 1000,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": settings.embedding_model,
                "embedding_dims": 3072,
                "api_key": settings.llm_api_key,
                "openai_base_url": settings.llm_api_base or "https://api.openai.com/v1",
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "embedding_model_dims": 3072,
                "path": "/tmp/qdrant_mem0",
            },
        },
    }
    _memory_instance = Memory.from_config(config)
    return _memory_instance


def _add_sync(memory: Memory, messages: list[dict], session_id: str) -> None:
    """Blocking mem0 add — runs in a thread pool."""
    try:
        memory.add(messages, run_id=session_id)
        logger.info("mem0: added %d messages for session %s", len(messages), session_id)
    except Exception:
        logger.exception("mem0: failed to add messages for session %s", session_id)


async def add_conversation_to_memory(session_id: str, messages: list[dict]) -> None:
    """Store latest conversation turn in mem0 (fire-and-forget in background thread)."""
    import asyncio
    memory = get_memory()
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _add_sync, memory, messages, session_id)


def _search_sync(memory: Memory, query: str, session_id: str, limit: int) -> str:
    """Blocking mem0 search — runs in a thread pool."""
    try:
        results = memory.search(query=query, run_id=session_id, limit=limit)
    except Exception:
        logger.exception("mem0: search failed for session %s", session_id)
        return ""

    memories = results.get("results", [])
    if not memories:
        return ""

    lines = ["[历史对话记忆]"]
    for entry in memories:
        lines.append(f"- {entry['memory']}")
    logger.info("mem0: found %d memories for session %s", len(memories), session_id)
    return "\n".join(lines)


async def search_relevant_memories(session_id: str, query: str, limit: int = 5) -> str:
    """Search mem0 for memories relevant to the query, formatted as text."""
    import asyncio
    memory = get_memory()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _search_sync, memory, query, session_id, limit)
