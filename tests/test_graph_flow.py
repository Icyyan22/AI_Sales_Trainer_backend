"""Integration tests for the LangGraph conversation flow."""

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.nodes.state_updater import state_updater_node
from app.graph.routing import route_by_coverage
from app.graph.state import ConversationState


def make_initial_state(**overrides) -> ConversationState:
    state = {
        "session_id": "test-001",
        "scenario_id": "diabetes_drug",
        "messages": [AIMessage(content="你好，我是张主任。你们有什么新药？")],
        "semantic_coverage": {"SP1": False, "SP2": False, "SP3": False},
        "semantic_evidence": {},
        "current_analysis": None,
        "current_strategy": None,
        "phase": "active",
        "turn_count": 0,
        "customer_attitude": "cautious",
        "stagnation_count": 0,
        "max_turns": 10,
        "difficulty": "normal",
    }
    state.update(overrides)
    return state


class TestRouting:
    def test_continue_when_active(self):
        state = make_initial_state(phase="active")
        assert route_by_coverage(state) == "continue"

    def test_wrap_up_when_all_covered(self):
        state = make_initial_state(phase="wrapping_up")
        assert route_by_coverage(state) == "wrap_up"

    def test_force_wrap_up(self):
        state = make_initial_state(phase="force_wrapping_up")
        assert route_by_coverage(state) == "force_wrap_up"

    def test_end_when_completed(self):
        state = make_initial_state(phase="completed")
        assert route_by_coverage(state) == "end"


class TestStateUpdater:
    async def test_updates_coverage_on_full_match(self):
        state = make_initial_state(
            current_analysis={
                "analysis": [
                    {
                        "point_id": "SP1",
                        "newly_matched": True,
                        "confidence": 0.95,
                        "match_level": "full",
                        "evidence": "HbA1c降幅1.5%",
                        "reasoning": "具体数据",
                    }
                ],
                "expression_quality": "good",
                "quality_note": "good",
            }
        )
        result = await state_updater_node(state)
        assert result["semantic_coverage"]["SP1"] is True
        assert result["semantic_coverage"]["SP2"] is False
        assert result["turn_count"] == 1
        assert result["stagnation_count"] == 0

    async def test_ignores_partial_match(self):
        state = make_initial_state(
            current_analysis={
                "analysis": [
                    {
                        "point_id": "SP1",
                        "newly_matched": True,
                        "confidence": 0.95,
                        "match_level": "partial",
                        "evidence": "效果好",
                        "reasoning": "模糊",
                    }
                ],
            }
        )
        result = await state_updater_node(state)
        assert result["semantic_coverage"]["SP1"] is False

    async def test_ignores_low_confidence(self):
        state = make_initial_state(
            current_analysis={
                "analysis": [
                    {
                        "point_id": "SP1",
                        "newly_matched": True,
                        "confidence": 0.5,
                        "match_level": "full",
                        "evidence": "something",
                        "reasoning": "not sure",
                    }
                ],
            }
        )
        result = await state_updater_node(state)
        assert result["semantic_coverage"]["SP1"] is False

    async def test_attitude_progression(self):
        state = make_initial_state(
            semantic_coverage={"SP1": True, "SP2": False, "SP3": False},
            current_analysis={
                "analysis": [
                    {
                        "point_id": "SP2",
                        "newly_matched": True,
                        "confidence": 0.9,
                        "match_level": "full",
                        "evidence": "低血糖1.8%",
                        "reasoning": "具体数据",
                    }
                ],
            },
        )
        result = await state_updater_node(state)
        assert result["customer_attitude"] == "interested"

    async def test_wrapping_up_when_all_covered(self):
        state = make_initial_state(
            semantic_coverage={"SP1": True, "SP2": True, "SP3": False},
            current_analysis={
                "analysis": [
                    {
                        "point_id": "SP3",
                        "newly_matched": True,
                        "confidence": 0.88,
                        "match_level": "full",
                        "evidence": "每周一次",
                        "reasoning": "便利性",
                    }
                ],
            },
        )
        result = await state_updater_node(state)
        assert result["phase"] == "wrapping_up"
        assert result["customer_attitude"] == "convinced"

    async def test_force_wrapping_up_on_max_turns(self):
        state = make_initial_state(
            turn_count=9,
            max_turns=10,
            current_analysis={"analysis": []},
        )
        result = await state_updater_node(state)
        assert result["phase"] == "force_wrapping_up"

    async def test_stagnation_tracking(self):
        state = make_initial_state(
            stagnation_count=1,
            current_analysis={"analysis": []},
        )
        result = await state_updater_node(state)
        assert result["stagnation_count"] == 2
