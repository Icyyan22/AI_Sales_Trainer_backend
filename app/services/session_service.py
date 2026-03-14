from __future__ import annotations

import datetime
import json
import logging
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.graph.builder import get_graph
from app.graph.state import ConversationState
from app.models.db import (
    MessageRecord,
    SessionRecord,
    async_session_factory,
)
from app.models.scenario import load_scenario

from sqlalchemy import delete, select


async def create_session(
    scenario_id: str, difficulty: str = "normal", user_id: str | None = None
) -> dict:
    scenario = load_scenario(scenario_id)
    session_id = str(uuid.uuid4())

    # Build initial state
    initial_state: ConversationState = {
        "session_id": session_id,
        "scenario_id": scenario_id,
        "messages": [AIMessage(content=scenario.opening_message)],
        "semantic_coverage": scenario.get_coverage_init(),
        "semantic_evidence": {},
        "current_analysis": None,
        "current_strategy": None,
        "phase": "active",
        "turn_count": 0,
        "customer_attitude": "cautious",
        "stagnation_count": 0,
        "max_turns": settings.max_turns,
        "difficulty": difficulty,
    }

    # Initialize graph state via update (so checkpointer stores it)
    graph = await get_graph()
    config = {"configurable": {"thread_id": session_id}}
    await graph.aupdate_state(config, initial_state)

    # Persist to DB
    async with async_session_factory() as db:
        db_session = SessionRecord(
            id=session_id,
            scenario_id=scenario_id,
            user_id=user_id,
            difficulty=difficulty,
            status="active",
        )
        db.add(db_session)

        opening_msg = MessageRecord(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=scenario.opening_message,
            turn=0,
        )
        db.add(opening_msg)
        await db.commit()

    return {
        "session_id": session_id,
        "scenario": {
            "name": scenario.name,
            "customer_profile": scenario.customer_profile.model_dump(),
            "product": scenario.product.model_dump(),
            "semantic_points": [sp.model_dump() for sp in scenario.semantic_points],
        },
        "opening_message": {
            "role": "assistant",
            "content": scenario.opening_message,
        },
        "semantic_coverage": scenario.get_coverage_init(),
        "phase": "active",
    }


async def delete_session(session_id: str) -> bool:
    async with async_session_factory() as db:
        result = await db.execute(
            select(SessionRecord).where(SessionRecord.id == session_id)
        )
        session = result.scalar_one_or_none()
        if not session:
            return False
        await db.execute(
            delete(MessageRecord).where(MessageRecord.session_id == session_id)
        )
        await db.delete(session)
        await db.commit()
    return True


async def list_sessions(user_id: str | None = None) -> list[dict]:
    async with async_session_factory() as db:
        query = select(SessionRecord).order_by(SessionRecord.created_at.desc())
        if user_id:
            query = query.where(SessionRecord.user_id == user_id)
        result = await db.execute(query)
        sessions = result.scalars().all()
        return [
            {
                "session_id": s.id,
                "scenario_id": s.scenario_id,
                "status": s.status,
                "difficulty": s.difficulty,
                "created_at": s.created_at.isoformat() if s.created_at else "",
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            }
            for s in sessions
        ]


async def get_session_status(session_id: str) -> dict | None:
    async with async_session_factory() as db:
        result = await db.execute(
            select(SessionRecord).where(SessionRecord.id == session_id)
        )
        session = result.scalar_one_or_none()
        if not session:
            return None

        # Get graph state for live data
        graph = await get_graph()
        config = {"configurable": {"thread_id": session_id}}
        state = await graph.aget_state(config)

        state_values = state.values if state else {}

        return {
            "session_id": session.id,
            "scenario_id": session.scenario_id,
            "status": session.status,
            "difficulty": session.difficulty,
            "semantic_coverage": state_values.get("semantic_coverage"),
            "phase": state_values.get("phase"),
            "turn_count": state_values.get("turn_count"),
            "customer_attitude": state_values.get("customer_attitude"),
            "created_at": session.created_at.isoformat() if session.created_at else "",
        }


async def process_message(session_id: str, content: str) -> dict:
    import asyncio
    import logging
    import time

    from app.graph.nodes.evaluator import evaluator_node
    from app.graph.nodes.state_updater import state_updater_node
    from app.graph.nodes.strategist_persona import strategist_persona_node

    logger = logging.getLogger(__name__)

    graph = await get_graph()
    config = {"configurable": {"thread_id": session_id}}

    # Get current graph state and add user message
    snapshot = await graph.aget_state(config)
    sv = dict(snapshot.values)
    sv["messages"] = list(sv["messages"]) + [HumanMessage(content=content)]

    t_start = time.perf_counter()

    # Compress early context if conversation is long (with caching)
    from app.memory.context_compressor import compress_context
    memory_context, summary_early_count = await compress_context(
        sv["messages"],
        cached_summary=sv.get("_memory_context", ""),
        cached_early_count=sv.get("_summary_early_count", 0),
    )
    sv["_memory_context"] = memory_context
    sv["_summary_early_count"] = summary_early_count

    # Run evaluator and persona in parallel
    sv_for_persona = dict(sv)
    sv_for_persona["current_analysis"] = {}
    sv_for_persona["_memory_context"] = memory_context

    eval_task = asyncio.create_task(evaluator_node(sv))
    persona_task = asyncio.create_task(strategist_persona_node(sv_for_persona))

    eval_result, persona_result = await asyncio.gather(eval_task, persona_task)

    logger.info("⏱ Parallel eval+persona took %.1fs", time.perf_counter() - t_start)

    # Apply evaluator result → state updater
    sv.update(eval_result)
    analysis = sv.get("current_analysis", {})

    updater_result = await state_updater_node(sv)
    sv.update(updater_result)

    # Extract persona response
    ai_response = ""
    for msg in persona_result.get("messages", []):
        if isinstance(msg, AIMessage):
            ai_response = msg.content
            break
    strategy = persona_result.get("current_strategy", {})

    # Handle phase from persona (wrapping_up → completed)
    if persona_result.get("phase"):
        sv["phase"] = persona_result["phase"]

    turn = sv["turn_count"]
    coverage = sv["semantic_coverage"]

    # Persist messages to DB
    user_msg_id = str(uuid.uuid4())
    ai_msg_id = str(uuid.uuid4())

    async with async_session_factory() as db:
        db.add(MessageRecord(
            id=user_msg_id,
            session_id=session_id,
            role="user",
            content=content,
            analysis=analysis,
            turn=turn,
        ))
        db.add(MessageRecord(
            id=ai_msg_id,
            session_id=session_id,
            role="assistant",
            content=ai_response,
            strategy=strategy,
            turn=turn,
        ))

        # Update session status if completed
        if sv["phase"] == "completed":
            stmt = select(SessionRecord).where(SessionRecord.id == session_id)
            res = await db.execute(stmt)
            session = res.scalar_one_or_none()
            if session:
                session.status = "completed"
                session.final_coverage = coverage
                session.completed_at = datetime.datetime.now(datetime.UTC)

        await db.commit()

    # Update graph state so checkpointer stays in sync
    await graph.aupdate_state(
        config,
        {
            "messages": [HumanMessage(content=content), AIMessage(content=ai_response)],
            "current_analysis": analysis,
            "current_strategy": strategy,
            "semantic_coverage": coverage,
            "semantic_evidence": sv.get("semantic_evidence", {}),
            "customer_attitude": sv["customer_attitude"],
            "phase": sv["phase"],
            "turn_count": turn,
            "stagnation_count": sv.get("stagnation_count", 0),
            "_memory_context": sv.get("_memory_context", ""),
            "_summary_early_count": sv.get("_summary_early_count", 0),
        },
        as_node="strategist_persona",
    )

    total = len(coverage) if coverage else 1
    covered = sum(coverage.values()) if coverage else 0

    return {
        "user_message": {
            "id": user_msg_id,
            "role": "user",
            "content": content,
            "analysis": analysis,
        },
        "assistant_message": {
            "id": ai_msg_id,
            "role": "assistant",
            "content": ai_response,
        },
        "state": {
            "semantic_coverage": coverage,
            "coverage_rate": covered / total,
            "phase": sv["phase"],
            "customer_attitude": sv["customer_attitude"],
            "turn_count": turn,
            "strategy_used": strategy,
        },
    }


async def _apply_evaluator_result(sv, evaluator_task, t_start, logger):
    """Await evaluator result and apply state updates. Returns (analysis, coverage, turn)."""
    from app.graph.nodes.state_updater import state_updater_node

    eval_result = evaluator_task.result() if evaluator_task.done() else await evaluator_task
    t_eval = time.perf_counter() - t_start
    sv.update(eval_result)
    analysis = sv.get("current_analysis", {})

    updater_result = await state_updater_node(sv)
    sv.update(updater_result)

    logger.info("⏱ Evaluator finished at %.1fs", t_eval)
    return analysis, sv["semantic_coverage"], sv["turn_count"]


async def _persist_stream_results(
    session_id, content, response_text, analysis, strategy, coverage, turn, sv
):
    """Persist messages to DB, update graph state."""
    user_msg_id = str(uuid.uuid4())
    ai_msg_id = str(uuid.uuid4())

    if sv.get("phase") in ("wrapping_up", "force_wrapping_up"):
        sv["phase"] = "completed"

    async with async_session_factory() as db:
        db.add(MessageRecord(
            id=user_msg_id, session_id=session_id, role="user",
            content=content, analysis=analysis, turn=turn,
        ))
        db.add(MessageRecord(
            id=ai_msg_id, session_id=session_id, role="assistant",
            content=response_text.strip(), strategy=strategy, turn=turn,
        ))
        if sv["phase"] == "completed":
            stmt = select(SessionRecord).where(SessionRecord.id == session_id)
            res = await db.execute(stmt)
            session = res.scalar_one_or_none()
            if session:
                session.status = "completed"
                session.final_coverage = coverage
                session.completed_at = datetime.datetime.now(datetime.UTC)
        await db.commit()

    graph = await get_graph()
    config = {"configurable": {"thread_id": session_id}}
    await graph.aupdate_state(
        config,
        {
            "messages": [HumanMessage(content=content), AIMessage(content=response_text.strip())],
            "current_analysis": analysis,
            "current_strategy": strategy,
            "semantic_coverage": coverage,
            "semantic_evidence": sv.get("semantic_evidence", {}),
            "customer_attitude": sv["customer_attitude"],
            "phase": sv["phase"],
            "turn_count": turn,
            "stagnation_count": sv.get("stagnation_count", 0),
            "_memory_context": sv.get("_memory_context", ""),
            "_summary_early_count": sv.get("_summary_early_count", 0),
        },
        as_node="strategist_persona",
    )


async def stream_message(session_id: str, content: str):
    import asyncio
    import logging
    import re
    import time

    from app.graph.nodes.evaluator import evaluator_node
    from app.graph.nodes.strategist_persona import build_persona_prompt
    from app.llm.provider import llm_stream_call

    logger = logging.getLogger(__name__)

    graph = await get_graph()
    config = {"configurable": {"thread_id": session_id}}

    snapshot = await graph.aget_state(config)
    sv = dict(snapshot.values)
    sv["messages"] = list(sv["messages"]) + [HumanMessage(content=content)]

    coverage = sv.get("semantic_coverage", {})
    turn = sv.get("turn_count", 0)
    response_text = ""
    analysis = {}
    strategy = {}
    evaluator_done = False

    try:
        yield {
            "event": "thinking",
            "data": json.dumps({"step": "analyzing", "message": "正在思考..."}, ensure_ascii=False),
        }
        await asyncio.sleep(0.05)

        t_start = time.perf_counter()

        from app.memory.context_compressor import compress_context
        memory_context, summary_early_count = await compress_context(
            sv["messages"],
            cached_summary=sv.get("_memory_context", ""),
            cached_early_count=sv.get("_summary_early_count", 0),
        )
        sv["_memory_context"] = memory_context
        sv["_summary_early_count"] = summary_early_count
        logger.info("⏱ Context compression took %.1fs", time.perf_counter() - t_start)

        evaluator_task = asyncio.create_task(evaluator_node(sv))

        sv_for_persona = dict(sv)
        sv_for_persona["current_analysis"] = {}
        system_prompt, chat_messages = build_persona_prompt(
            sv_for_persona, streaming=True, memory_context=memory_context
        )

        # --- Stream persona tokens + parse <think> block ---
        full_text = ""
        thinking_emitted = False
        response_started = False

        async for token in llm_stream_call(
            system=system_prompt, messages=chat_messages,
            temperature=settings.persona_temperature,
        ):
            full_text += token

            if not thinking_emitted:
                close_idx = full_text.find("</think>")
                if close_idx != -1:
                    think_block = full_text[:close_idx]
                    json_match = re.search(r"\{[^}]+\}", think_block)
                    if json_match:
                        try:
                            strategy = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            pass
                    thinking_emitted = True
                    after = full_text[close_idx + len("</think>"):].lstrip("\n")
                    if after:
                        # Yield buffered post-think content in small chunks
                        for i in range(0, len(after), 2):
                            piece = after[i:i+2]
                            response_text += piece
                            response_started = True
                            yield {"event": "delta", "data": json.dumps({"content": piece}, ensure_ascii=False)}
                            await asyncio.sleep(0.02)
            else:
                chunk = token.lstrip("\n") if not response_started else token
                if chunk:
                    response_started = True
                    response_text += chunk
                    yield {"event": "delta", "data": json.dumps({"content": chunk}, ensure_ascii=False)}

            # Check if evaluator finished mid-stream
            if not evaluator_done and evaluator_task.done():
                try:
                    analysis, coverage, turn = await _apply_evaluator_result(sv, evaluator_task, t_start, logger)
                    yield {"event": "analysis", "data": json.dumps(analysis, ensure_ascii=False)}
                except Exception:
                    logger.exception("Evaluator task failed")
                evaluator_done = True

        logger.info("⏱ Persona streaming took %.1fs", time.perf_counter() - t_start)

        if not thinking_emitted:
            response_text = full_text.strip()
            for i in range(0, len(response_text), 2):
                piece = response_text[i:i+2]
                yield {"event": "delta", "data": json.dumps({"content": piece}, ensure_ascii=False)}
                await asyncio.sleep(0.02)

        # Await evaluator if still pending
        if not evaluator_done:
            try:
                analysis, coverage, turn = await _apply_evaluator_result(sv, evaluator_task, t_start, logger)
                yield {"event": "analysis", "data": json.dumps(analysis, ensure_ascii=False)}
            except Exception:
                logger.exception("Evaluator task failed")
            evaluator_done = True

        # Persist results
        await _persist_stream_results(
            session_id, content, response_text, analysis, strategy, coverage, turn, sv
        )

        logger.info("⏱ Total stream_message took %.1fs", time.perf_counter() - t_start)

    except Exception:
        logger.exception("Error in stream_message")
        if not evaluator_done and not evaluator_task.done():
            evaluator_task.cancel()
        if not response_text:
            response_text = "抱歉，处理出现问题，请重试。"
            yield {"event": "delta", "data": json.dumps({"content": response_text}, ensure_ascii=False)}

    total = len(coverage) if coverage else 1
    covered = sum(coverage.values()) if coverage else 0

    yield {
        "event": "metadata",
        "data": json.dumps({
            "analysis": analysis, "coverage": coverage,
            "coverage_rate": covered / total,
            "customer_attitude": sv.get("customer_attitude", "cautious"),
            "phase": sv.get("phase", "active"),
        }, ensure_ascii=False),
    }
    yield {"event": "done", "data": json.dumps({"turn": turn})}


async def get_messages(session_id: str) -> list[dict]:
    async with async_session_factory() as db:
        result = await db.execute(
            select(MessageRecord)
            .where(MessageRecord.session_id == session_id)
            .order_by(MessageRecord.created_at)
        )
        messages = result.scalars().all()
        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "turn": m.turn,
                "analysis": m.analysis,
                "strategy": m.strategy,
            }
            for m in messages
        ]


async def complete_session(session_id: str) -> dict | None:
    graph = await get_graph()
    config = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(config)
    if not state or not state.values:
        return None

    coverage = state.values.get("semantic_coverage", {})

    async with async_session_factory() as db:
        result = await db.execute(
            select(SessionRecord).where(SessionRecord.id == session_id)
        )
        session = result.scalar_one_or_none()
        if not session:
            return None

        session.status = "completed"
        session.final_coverage = coverage
        session.completed_at = datetime.datetime.now(datetime.UTC)
        await db.commit()

    return {
        "session_id": session_id,
        "status": "completed",
        "final_coverage": coverage,
    }
