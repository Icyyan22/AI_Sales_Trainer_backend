from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class ConversationState(TypedDict):
    session_id: str
    scenario_id: str

    messages: Annotated[list, add_messages]

    # DST core
    semantic_coverage: dict[str, bool]
    semantic_evidence: dict[str, dict]

    # Inter-agent communication
    current_analysis: dict | None
    current_strategy: dict | None

    # Session control
    phase: str  # active | wrapping_up | force_wrapping_up | completed
    turn_count: int
    customer_attitude: str  # cautious | interested | convinced
    stagnation_count: int
    max_turns: int
    difficulty: str  # easy | normal | hard
    _memory_context: str  # compressed early-conversation summary (cached)
    _summary_early_count: int  # number of early messages when summary was generated
