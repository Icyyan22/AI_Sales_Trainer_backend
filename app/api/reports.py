from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import ReportResponse
from app.services.report_service import generate_report

router = APIRouter(prefix="/sessions", tags=["reports"])


@router.get("/{session_id}/report", response_model=ReportResponse)
async def get_report(session_id: str):
    report = await generate_report(session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Session not found or no data available")
    return report
