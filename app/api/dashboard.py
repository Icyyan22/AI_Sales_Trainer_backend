from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.auth import require_user, require_admin
from app.models.db import UserRecord
from app.services.dashboard_service import get_personal_stats, get_admin_stats

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/me")
async def personal_dashboard(
    user: UserRecord = Depends(require_user),
    days: int | None = Query(None, description="Filter by recent N days"),
    scenario_id: str | None = Query(None, description="Filter by scenario"),
    difficulty: str | None = Query(None, description="Filter by difficulty"),
):
    return await get_personal_stats(
        user_id=user.id,
        days=days,
        scenario_id=scenario_id,
        difficulty=difficulty,
    )


@router.get("/admin")
async def admin_dashboard(user: UserRecord = Depends(require_admin)):
    return await get_admin_stats()


@router.get("/admin/users/{user_id}")
async def admin_user_detail(
    user_id: str,
    _: UserRecord = Depends(require_admin),
    days: int | None = Query(None),
    scenario_id: str | None = Query(None),
    difficulty: str | None = Query(None),
):
    return await get_personal_stats(
        user_id=user_id,
        days=days,
        scenario_id=scenario_id,
        difficulty=difficulty,
    )
