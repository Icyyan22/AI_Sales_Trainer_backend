from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    MessageListResponse,
    SendMessageRequest,
    SendMessageResponse,
)
from app.services import session_service

router = APIRouter(prefix="/sessions", tags=["messages"])


@router.post("/{session_id}/messages", response_model=SendMessageResponse)
async def send_message(session_id: str, req: SendMessageRequest):
    status = await session_service.get_session_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    if status["status"] == "completed":
        raise HTTPException(status_code=400, detail="Session already completed")

    result = await session_service.process_message(session_id, req.content)
    return result


@router.post("/{session_id}/chat/stream")
async def stream_chat(session_id: str, req: SendMessageRequest):
    status = await session_service.get_session_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    if status["status"] == "completed":
        raise HTTPException(status_code=400, detail="Session already completed")

    return EventSourceResponse(
        session_service.stream_message(session_id, req.content)
    )


@router.get("/{session_id}/messages", response_model=MessageListResponse)
async def get_messages(session_id: str):
    messages = await session_service.get_messages(session_id)
    return {"session_id": session_id, "messages": messages}
