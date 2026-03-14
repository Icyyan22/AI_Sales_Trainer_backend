from __future__ import annotations

import json
import logging
import re

from app.llm.provider import llm_call

logger = logging.getLogger(__name__)


def _fix_common_json_errors(raw: str) -> str:
    # Strip markdown code fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)

    # Try to extract JSON object if surrounded by other text
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        raw = match.group()

    # Remove trailing commas before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    return raw


def _default_evaluator_result() -> dict:
    return {
        "analysis": [],
        "expression_quality": "fair",
        "quality_note": "评估失败，使用默认结果",
    }


async def structured_llm_call(
    *,
    system: str,
    user: str | None = None,
    messages: list[dict] | None = None,
    temperature: float = 0.0,
    retries: int = 2,
    default: dict | None = None,
    model: str | None = None,
) -> dict:
    for attempt in range(retries + 1):
        raw = await llm_call(
            system=system,
            user=user,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            model=model,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return json.loads(_fix_common_json_errors(raw))
            except json.JSONDecodeError:
                if attempt < retries:
                    logger.warning(
                        "JSON parse failed (attempt %d/%d), raw[:500]=%s, retrying",
                        attempt + 1,
                        retries + 1,
                        raw[:500],
                    )
                    continue
                logger.error("JSON parse failed after all retries, raw[:500]=%s, using default", raw[:500])
                return default if default is not None else _default_evaluator_result()
