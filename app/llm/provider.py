from __future__ import annotations

import litellm

from app.config import settings


import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_SSL_VERIFY"] = "False"

litellm.drop_params = True
litellm.ssl_verify = False


async def llm_call(
    *,
    system: str,
    user: str | None = None,
    messages: list[dict] | None = None,
    temperature: float = 0.0,
    response_format: dict | None = None,
    model: str | None = None,
) -> str:
    call_messages = [{"role": "system", "content": system}]

    if messages:
        call_messages.extend(messages)

    if user:
        call_messages.append({"role": "user", "content": user})

    kwargs: dict = {
        "model": model or settings.llm_model,
        "messages": call_messages,
        "temperature": temperature,
        "api_key": settings.llm_api_key,
        "max_tokens": settings.llm_max_tokens,
    }

    if settings.llm_api_base:
        kwargs["api_base"] = settings.llm_api_base

    if response_format:
        kwargs["response_format"] = response_format

    response = await litellm.acompletion(**kwargs)
    return response.choices[0].message.content


async def llm_stream_call(
    *,
    system: str,
    user: str | None = None,
    messages: list[dict] | None = None,
    temperature: float = 0.0,
    chunk_size: int = 2,
):
    """Async generator that yields content in small chunks.

    Some API proxies don't support true streaming (return full response in 1 chunk).
    We simulate token-by-token output by splitting into small character groups.
    """
    import asyncio

    call_messages = [{"role": "system", "content": system}]

    if messages:
        call_messages.extend(messages)

    if user:
        call_messages.append({"role": "user", "content": user})

    kwargs: dict = {
        "model": settings.llm_model,
        "messages": call_messages,
        "temperature": temperature,
        "api_key": settings.llm_api_key,
        "max_tokens": settings.llm_max_tokens,
        "stream": True,
    }

    if settings.llm_api_base:
        kwargs["api_base"] = settings.llm_api_base

    # Collect all chunks from upstream
    full_text = ""
    response = await litellm.acompletion(**kwargs)
    async for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            full_text += delta.content

    # Yield in small chunks to simulate streaming
    for i in range(0, len(full_text), chunk_size):
        yield full_text[i : i + chunk_size]
        await asyncio.sleep(0.02)
