"""Evaluator accuracy regression tests.

These tests verify that the evaluator correctly identifies semantic point
coverage across a range of sales utterances, including edge cases like
wrong data, vague statements, and partial coverage.

Each test case provides a sales message and expected match outcomes,
run against the real evaluator node with LLM calls.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.nodes.evaluator import evaluator_node


def _make_state(user_message: str, coverage: dict | None = None):
    return {
        "session_id": "eval-test",
        "scenario_id": "diabetes_drug",
        "messages": [
            AIMessage(content="你好，我是张主任。你们有什么新药？"),
            HumanMessage(content=user_message),
        ],
        "semantic_coverage": coverage or {"SP1": False, "SP2": False, "SP3": False},
        "semantic_evidence": {},
        "current_analysis": None,
        "current_strategy": None,
        "phase": "active",
        "turn_count": 1,
        "customer_attitude": "cautious",
        "stagnation_count": 0,
        "max_turns": 10,
        "difficulty": "normal",
    }


# ── Positive cases: should match ────────────────────────────────────────

class TestPositiveMatches:
    """Sales utterances that should be recognized as covering semantic points."""

    @pytest.mark.asyncio
    async def test_sp1_exact_data(self):
        """Citing exact HbA1c data should match SP1 with high confidence."""
        state = _make_state(
            "我们这款药在III期临床试验中，HbA1c平均降幅达到1.5%，"
            "糖化血红蛋白达标率显著优于对照组。"
        )
        result = await evaluator_node(state)
        analysis = result["current_analysis"]["analysis"]
        sp1 = next((p for p in analysis if p["point_id"] == "SP1"), None)
        assert sp1 is not None
        assert sp1["newly_matched"] is True
        assert sp1["confidence"] >= 0.8
        assert sp1["match_level"] == "full"

    @pytest.mark.asyncio
    async def test_sp2_low_hypoglycemia(self):
        """Citing low hypoglycemia rate should match SP2."""
        state = _make_state(
            "安全性方面，低血糖事件发生率仅为1.8%，"
            "远低于磺脲类药物，老年亚组安全性表现尤其突出。"
        )
        result = await evaluator_node(state)
        analysis = result["current_analysis"]["analysis"]
        sp2 = next((p for p in analysis if p["point_id"] == "SP2"), None)
        assert sp2 is not None
        assert sp2["newly_matched"] is True
        assert sp2["confidence"] >= 0.8
        assert sp2["match_level"] == "full"

    @pytest.mark.asyncio
    async def test_sp3_weekly_injection(self):
        """Mentioning weekly injection should match SP3."""
        state = _make_state(
            "这款药每周只需皮下注射一次，大大减少注射负担，"
            "患者依从性提升了40%以上。"
        )
        result = await evaluator_node(state)
        analysis = result["current_analysis"]["analysis"]
        sp3 = next((p for p in analysis if p["point_id"] == "SP3"), None)
        assert sp3 is not None
        assert sp3["newly_matched"] is True
        assert sp3["confidence"] >= 0.8
        assert sp3["match_level"] == "full"


# ── Negative cases: should NOT match ────────────────────────────────────

class TestNegativeMatches:
    """Vague or irrelevant utterances that should NOT be judged as coverage."""

    @pytest.mark.asyncio
    async def test_vague_no_data(self):
        """Saying 'effect is good' without data should not match SP1."""
        state = _make_state("这个药效果特别好，很多医院都在用，反馈都不错。")
        result = await evaluator_node(state)
        analysis = result["current_analysis"]["analysis"]
        sp1 = next((p for p in analysis if p["point_id"] == "SP1"), None)
        if sp1 and sp1.get("newly_matched"):
            assert sp1["match_level"] != "full" or sp1["confidence"] < 0.7

    @pytest.mark.asyncio
    async def test_offtopic(self):
        """Off-topic messages should not match any points."""
        state = _make_state("张主任，最近天气真不错。您有没有兴趣打一场高尔夫？")
        result = await evaluator_node(state)
        analysis = result["current_analysis"]["analysis"]
        for point in analysis:
            assert not point.get("newly_matched") or point["match_level"] == "none"


# ── Edge cases: wrong data, partial ──────────────────────────────────────

class TestEdgeCases:
    """Tricky scenarios that test evaluator robustness."""

    @pytest.mark.asyncio
    async def test_wrong_hba1c_data(self):
        """Citing wrong HbA1c data (3% instead of 1.5%) should be partial, not full."""
        state = _make_state(
            "我们这个药非常强效，HbA1c能降3%，效果在同类药物中遥遥领先。"
        )
        result = await evaluator_node(state)
        analysis = result["current_analysis"]["analysis"]
        sp1 = next((p for p in analysis if p["point_id"] == "SP1"), None)
        if sp1 and sp1.get("newly_matched"):
            # Wrong data should NOT be judged as "full"
            assert sp1["match_level"] == "partial", (
                f"Wrong data should be partial, got {sp1['match_level']}"
            )

    @pytest.mark.asyncio
    async def test_partial_safety_mention(self):
        """Saying 'safe' without specifics should be partial at best."""
        state = _make_state("这个药副作用很小，安全性很好，基本没什么不良反应。")
        result = await evaluator_node(state)
        analysis = result["current_analysis"]["analysis"]
        sp2 = next((p for p in analysis if p["point_id"] == "SP2"), None)
        if sp2 and sp2.get("newly_matched"):
            assert sp2["match_level"] != "full" or sp2["confidence"] < 0.7

    @pytest.mark.asyncio
    async def test_already_covered_still_detected(self):
        """If SP1 is already covered, evaluator may still detect the match.

        The state_updater is responsible for filtering out already-covered
        points via `not coverage.get(pid, False)`. The evaluator's job is
        to detect semantic matches regardless of prior coverage.
        """
        state = _make_state(
            "我再强调一下，HbA1c降幅1.5%是非常显著的。",
            coverage={"SP1": True, "SP2": False, "SP3": False},
        )
        result = await evaluator_node(state)
        analysis = result["current_analysis"]["analysis"]
        sp1 = next((p for p in analysis if p["point_id"] == "SP1"), None)
        # Evaluator detects the semantic match; state_updater handles dedup
        assert sp1 is not None
        assert sp1["match_level"] == "full"
