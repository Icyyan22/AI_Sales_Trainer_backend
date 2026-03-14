from __future__ import annotations

import json

from app.config import settings
from app.graph.builder import get_graph
from app.llm.structured_output import structured_llm_call
from app.models.scenario import load_scenario
from app.prompts.registry import get_prompt
from app.services import session_service


def _compute_efficiency(coverage: dict, turns: int) -> float:
    covered = sum(coverage.values())
    total = len(coverage)
    if turns == 0 or total == 0:
        return 0.0
    return min(1.0, (covered / total) * (total / turns))


def _format_skill_radar_text(skill_radar: dict) -> str:
    labels = {
        "data_citation": "数据引用",
        "customer_relevance": "客户关联",
        "fab_structure": "FAB结构",
        "interaction": "互动技巧",
    }
    if not skill_radar or all(v == 0.0 for v in skill_radar.values()):
        return "（无评分数据）"
    lines = []
    for key, label in labels.items():
        score = skill_radar.get(key, 0.0)
        lines.append(f"- {label}: {score:.1f}/5")
    avg = sum(skill_radar.values()) / len(skill_radar) if skill_radar else 0
    lines.append(f"- 综合均分: {avg:.1f}/5")
    return "\n".join(lines)


def _format_per_turn_quality(messages: list[dict]) -> str:
    labels = {
        "data_citation": "数据引用",
        "customer_relevance": "客户关联",
        "fab_structure": "FAB结构",
        "interaction": "互动技巧",
    }
    lines = []
    for m in messages:
        if m["role"] != "user" or not m.get("analysis"):
            continue
        eq = m["analysis"].get("expression_quality", {})
        if not isinstance(eq, dict):
            continue
        turn = m.get("turn", "?")
        parts = []
        for key, label in labels.items():
            dim = eq.get(key)
            if isinstance(dim, dict):
                score = dim.get("score", "?")
                note = dim.get("note", "")
                parts.append(f"{label} {score}/5")
        quality_note = eq.get("quality_note", "")
        scores_str = ", ".join(parts) if parts else "无评分"
        line = f"第{turn}轮: {scores_str}"
        if quality_note:
            line += f" — {quality_note}"
        lines.append(line)
    return "\n".join(lines) if lines else "（无评分数据）"


async def generate_report(session_id: str) -> dict | None:
    graph = await get_graph()
    config = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(config)

    if not state or not state.values:
        return None

    sv = state.values
    scenario_id = sv.get("scenario_id", "")
    coverage = sv.get("semantic_coverage", {})
    evidence = sv.get("semantic_evidence", {})
    turn_count = sv.get("turn_count", 0)

    try:
        scenario = load_scenario(scenario_id)
    except FileNotFoundError:
        return None

    total = len(coverage) if coverage else 1
    covered = sum(coverage.values()) if coverage else 0
    coverage_rate = covered / total
    efficiency_score = _compute_efficiency(coverage, turn_count)

    # Build semantic detail
    sp_map = {sp.id: sp.name for sp in scenario.semantic_points}
    semantic_detail = []
    for pid, is_covered in coverage.items():
        ev = evidence.get(pid, {})
        semantic_detail.append({
            "point_id": pid,
            "name": sp_map.get(pid, pid),
            "covered": is_covered,
            "covered_at_turn": ev.get("turn"),
            "confidence": ev.get("confidence"),
            "evidence": ev.get("evidence"),
        })

    # Compute skill radar first (needed for overall_score)
    messages = await session_service.get_messages(session_id)
    skill_radar = _compute_skill_radar_from_messages(messages)

    # Overall score: coverage 40% + efficiency 20% + quality 40%
    avg_quality = (
        sum(skill_radar.values()) / len(skill_radar) / 5.0
        if skill_radar and any(v > 0 for v in skill_radar.values())
        else 0.5
    )
    overall_score = int(coverage_rate * 40 + efficiency_score * 20 + avg_quality * 40)

    # Get conversation history for LLM feedback
    conversation_lines = []
    for m in messages:
        role_label = "销售" if m["role"] == "user" else "客户"
        conversation_lines.append(f"{role_label}: {m['content']}")
    conversation_text = "\n".join(conversation_lines)

    detail_text = json.dumps(semantic_detail, ensure_ascii=False, indent=2)

    # Generate LLM feedback with enriched context
    prompt_template = get_prompt("report", settings.prompt_version)
    system_prompt = prompt_template.format(
        scenario_name=scenario.name,
        total_turns=turn_count,
        coverage_rate=f"{coverage_rate:.0%}",
        semantic_detail=detail_text,
        skill_radar_text=_format_skill_radar_text(skill_radar),
        per_turn_quality=_format_per_turn_quality(messages),
        conversation_history=conversation_text,
    )

    feedback = await structured_llm_call(
        system=system_prompt,
        user="请生成训练反馈报告。",
        temperature=0.3,
        default={
            "strengths": ["完成了基本的产品信息传递"],
            "improvements": ["建议更多使用具体数据支撑论点"],
            "overall": "基本完成训练目标，仍有提升空间。",
        },
    )

    return {
        "session_id": session_id,
        "summary": {
            "total_turns": turn_count,
            "coverage_rate": coverage_rate,
            "efficiency_score": efficiency_score,
            "overall_score": overall_score,
        },
        "semantic_detail": semantic_detail,
        "skill_radar": skill_radar,
        "feedback": feedback,
    }


def _compute_skill_radar_from_messages(messages: list[dict]) -> dict:
    """Average multi-dimensional scores across all user turns."""
    dimensions = ["data_citation", "customer_relevance", "fab_structure", "interaction"]
    totals = {d: [] for d in dimensions}

    for m in messages:
        if m["role"] != "user" or not m.get("analysis"):
            continue
        eq = m["analysis"].get("expression_quality", {})
        if not isinstance(eq, dict):
            continue
        for dim in dimensions:
            dim_data = eq.get(dim)
            if isinstance(dim_data, dict) and "score" in dim_data:
                totals[dim].append(dim_data["score"])

    return {
        dim: round(sum(scores) / len(scores), 1) if scores else 0.0
        for dim, scores in totals.items()
    }


async def _compute_skill_radar(session_id: str) -> dict:
    """Average multi-dimensional scores across all user turns."""
    messages = await session_service.get_messages(session_id)
    return _compute_skill_radar_from_messages(messages)
