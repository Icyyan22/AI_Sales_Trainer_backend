from __future__ import annotations

from app.graph.state import ConversationState


def route_by_coverage(state: ConversationState) -> str:
    phase = state.get("phase", "active")

    if phase == "completed":
        return "end"
    elif phase == "wrapping_up":
        return "wrap_up"
    elif phase == "force_wrapping_up":
        return "force_wrap_up"
    else:
        return "continue"
