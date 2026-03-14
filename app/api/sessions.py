from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import require_user, get_current_user
from app.api.schemas import (
    CompleteSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    SessionListResponse,
    SessionStatusResponse,
)
from app.models.db import UserRecord, SessionRecord, async_session_factory
from app.services import session_service
from sqlalchemy import select

router = APIRouter(prefix="/sessions", tags=["sessions"])


async def _check_session_owner(session_id: str, user: UserRecord) -> None:
    """Verify that the user owns the session or is admin."""
    if user.role in ("admin", "super_admin"):
        return
    async with async_session_factory() as db:
        result = await db.execute(
            select(SessionRecord.user_id).where(SessionRecord.id == session_id)
        )
        row = result.first()
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        if row[0] and row[0] != user.id:
            raise HTTPException(status_code=403, detail="无权访问此会话")


@router.get("", response_model=SessionListResponse)
async def list_sessions(user: UserRecord = Depends(require_user)):
    if user.role in ("admin", "super_admin"):
        sessions = await session_service.list_sessions()
    else:
        sessions = await session_service.list_sessions(user_id=user.id)
    return {"sessions": sessions}


@router.post("", response_model=CreateSessionResponse, status_code=201)
async def create_session(req: CreateSessionRequest, user: UserRecord = Depends(require_user)):
    try:
        result = await session_service.create_session(req.scenario_id, req.difficulty, user_id=user.id)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' not found")


@router.get("/{session_id}", response_model=SessionStatusResponse)
async def get_session(session_id: str, user: UserRecord = Depends(require_user)):
    await _check_session_owner(session_id, user)
    result = await session_service.get_session_status(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str, user: UserRecord = Depends(require_user)):
    await _check_session_owner(session_id, user)
    deleted = await session_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/{session_id}/complete", response_model=CompleteSessionResponse)
async def complete_session(session_id: str, user: UserRecord = Depends(require_user)):
    await _check_session_owner(session_id, user)
    result = await session_service.complete_session(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    return result
