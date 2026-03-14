from __future__ import annotations

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes.evaluator import evaluator_node
from app.graph.nodes.state_updater import state_updater_node
from app.graph.nodes.strategist_persona import strategist_persona_node
from app.graph.routing import route_by_coverage
from app.graph.state import ConversationState

DB_PATH = "checkpoints.db"


def _build_workflow() -> StateGraph:
    graph = StateGraph(ConversationState)

    graph.add_node("evaluator", evaluator_node)
    graph.add_node("state_updater", state_updater_node)
    graph.add_node("strategist_persona", strategist_persona_node)

    graph.add_edge(START, "evaluator")
    graph.add_edge("evaluator", "state_updater")
    graph.add_conditional_edges(
        "state_updater",
        route_by_coverage,
        {
            "continue": "strategist_persona",
            "wrap_up": "strategist_persona",
            "force_wrap_up": "strategist_persona",
            "end": END,
        },
    )
    graph.add_edge("strategist_persona", END)
    return graph


_graph = None
_conn: aiosqlite.Connection | None = None


async def get_graph():
    global _graph, _conn
    if _graph is None:
        _conn = await aiosqlite.connect(DB_PATH)
        checkpointer = AsyncSqliteSaver(_conn)
        await checkpointer.setup()
        workflow = _build_workflow()
        _graph = workflow.compile(checkpointer=checkpointer)
    return _graph
