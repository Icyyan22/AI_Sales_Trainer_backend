from __future__ import annotations

import datetime
import json

from sqlalchemy import select, func, text, and_, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import SessionRecord, UserRecord, MessageRecord, async_session_factory
from app.models.scenario import list_scenarios


async def get_personal_stats(
    user_id: str,
    days: int | None = None,
    scenario_id: str | None = None,
    difficulty: str | None = None,
) -> dict:
    async with async_session_factory() as db:
        # Build base filter
        filters = [SessionRecord.user_id == user_id]
        if days:
            cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
            filters.append(SessionRecord.created_at >= cutoff)
        if scenario_id:
            filters.append(SessionRecord.scenario_id == scenario_id)
        if difficulty:
            filters.append(SessionRecord.difficulty == difficulty)

        base = and_(*filters)

        # Stats
        result = await db.execute(
            select(
                func.count().label("total"),
                func.sum(case((SessionRecord.status == "completed", 1), else_=0)).label("completed"),
            ).where(base)
        )
        row = result.first()
        total_sessions = row.total or 0
        completed_sessions = row.completed or 0

        # Scores from completed sessions with reports
        score_result = await db.execute(
            select(SessionRecord.final_report)
            .where(and_(base, SessionRecord.status == "completed", SessionRecord.final_report.isnot(None)))
        )
        reports = score_result.scalars().all()

        scores = []
        for report in reports:
            if isinstance(report, dict):
                s = report.get("summary", {}).get("overall_score")
                if s is not None:
                    scores.append(s)

        avg_score = round(sum(scores) / len(scores)) if scores else 0
        best_score = max(scores) if scores else 0
        completion_rate = round(completed_sessions / total_sessions, 2) if total_sessions else 0

        # Skill trend (completed sessions with reports, ordered by date)
        trend_result = await db.execute(
            select(
                SessionRecord.id,
                SessionRecord.scenario_id,
                SessionRecord.created_at,
                SessionRecord.final_report,
            )
            .where(and_(base, SessionRecord.status == "completed", SessionRecord.final_report.isnot(None)))
            .order_by(SessionRecord.created_at.asc())
        )
        trend_rows = trend_result.all()

        scenario_map = {s.id: s.name for s in list_scenarios()}
        skill_trend = []
        for r in trend_rows:
            report = r.final_report if isinstance(r.final_report, dict) else {}
            summary = report.get("summary", {})
            radar = report.get("skill_radar", {})
            skill_trend.append({
                "date": r.created_at.strftime("%Y-%m-%d") if r.created_at else "",
                "session_id": r.id,
                "scenario_name": scenario_map.get(r.scenario_id, r.scenario_id),
                "overall_score": summary.get("overall_score", 0),
                "data_citation": radar.get("data_citation", 0),
                "customer_relevance": radar.get("customer_relevance", 0),
                "fab_structure": radar.get("fab_structure", 0),
                "interaction": radar.get("interaction", 0),
            })

        # Weak areas: scenarios with low avg scores
        scenario_scores: dict[str, list[int]] = {}
        for r in trend_rows:
            report = r.final_report if isinstance(r.final_report, dict) else {}
            s = report.get("summary", {}).get("overall_score")
            if s is not None:
                scenario_scores.setdefault(r.scenario_id, []).append(s)

        weak_scenarios = sorted(
            [
                {
                    "scenario_id": sid,
                    "name": scenario_map.get(sid, sid),
                    "avg_score": round(sum(ss) / len(ss)),
                    "session_count": len(ss),
                }
                for sid, ss in scenario_scores.items()
            ],
            key=lambda x: x["avg_score"],
        )[:5]

        # Weak dimensions
        dim_labels = {
            "data_citation": "数据引用",
            "customer_relevance": "客户关联",
            "fab_structure": "FAB结构",
            "interaction": "互动技巧",
        }
        dim_totals: dict[str, list[float]] = {d: [] for d in dim_labels}
        for r in trend_rows:
            report = r.final_report if isinstance(r.final_report, dict) else {}
            radar = report.get("skill_radar", {})
            for dim in dim_labels:
                v = radar.get(dim)
                if v is not None and v > 0:
                    dim_totals[dim].append(v)

        weak_dimensions = sorted(
            [
                {
                    "dimension": dim,
                    "label": dim_labels[dim],
                    "avg_score": round(sum(vals) / len(vals), 1) if vals else 0,
                }
                for dim, vals in dim_totals.items()
            ],
            key=lambda x: x["avg_score"],
        )

        # Recent sessions
        recent_result = await db.execute(
            select(SessionRecord)
            .where(base)
            .order_by(SessionRecord.created_at.desc())
            .limit(10)
        )
        recent_sessions = [
            {
                "session_id": s.id,
                "scenario_id": s.scenario_id,
                "scenario_name": scenario_map.get(s.scenario_id, s.scenario_id),
                "overall_score": (
                    s.final_report.get("summary", {}).get("overall_score", 0)
                    if isinstance(s.final_report, dict)
                    else 0
                ),
                "coverage_rate": (
                    s.final_report.get("summary", {}).get("coverage_rate", 0)
                    if isinstance(s.final_report, dict)
                    else 0
                ),
                "difficulty": s.difficulty,
                "status": s.status,
                "created_at": s.created_at.isoformat() if s.created_at else "",
            }
            for s in recent_result.scalars().all()
        ]

        return {
            "stats": {
                "total_sessions": total_sessions,
                "completed_sessions": completed_sessions,
                "avg_score": avg_score,
                "best_score": best_score,
                "completion_rate": completion_rate,
            },
            "skill_trend": skill_trend,
            "weak_areas": {
                "weak_scenarios": weak_scenarios,
                "weak_dimensions": weak_dimensions,
            },
            "recent_sessions": recent_sessions,
        }


async def get_admin_stats() -> dict:
    async with async_session_factory() as db:
        # Overall stats
        user_count = (await db.execute(select(func.count()).select_from(UserRecord))).scalar() or 0

        total_sessions = (await db.execute(select(func.count()).select_from(SessionRecord))).scalar() or 0

        # Avg score across completed sessions with reports
        report_result = await db.execute(
            select(SessionRecord.final_report)
            .where(SessionRecord.status == "completed", SessionRecord.final_report.isnot(None))
        )
        all_scores = []
        for report in report_result.scalars().all():
            if isinstance(report, dict):
                s = report.get("summary", {}).get("overall_score")
                if s is not None:
                    all_scores.append(s)

        avg_score = round(sum(all_scores) / len(all_scores)) if all_scores else 0

        # Active today
        today_start = datetime.datetime.now(datetime.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        active_today = (await db.execute(
            select(func.count(func.distinct(SessionRecord.user_id)))
            .where(SessionRecord.created_at >= today_start)
        )).scalar() or 0

        # Per-user stats
        users_result = await db.execute(
            select(UserRecord).order_by(UserRecord.created_at.asc())
        )
        users = users_result.scalars().all()

        scenario_map = {s.id: s.name for s in list_scenarios()}
        user_stats = []
        for u in users:
            sess_result = await db.execute(
                select(
                    func.count().label("total"),
                    func.sum(case((SessionRecord.status == "completed", 1), else_=0)).label("completed"),
                    func.max(SessionRecord.created_at).label("last_active"),
                ).where(SessionRecord.user_id == u.id)
            )
            sr = sess_result.first()
            u_total = sr.total or 0
            u_completed = sr.completed or 0

            # Get user's scores
            u_reports = await db.execute(
                select(SessionRecord.final_report)
                .where(
                    SessionRecord.user_id == u.id,
                    SessionRecord.status == "completed",
                    SessionRecord.final_report.isnot(None),
                )
            )
            u_scores = []
            for report in u_reports.scalars().all():
                if isinstance(report, dict):
                    s = report.get("summary", {}).get("overall_score")
                    if s is not None:
                        u_scores.append(s)

            user_stats.append({
                "user_id": u.id,
                "username": u.username,
                "display_name": u.display_name or u.username,
                "role": u.role or "user",
                "total_sessions": u_total,
                "avg_score": round(sum(u_scores) / len(u_scores)) if u_scores else 0,
                "completion_rate": round(u_completed / u_total, 2) if u_total else 0,
                "last_active": sr.last_active.isoformat() if sr.last_active else None,
            })

        # Per-scenario stats
        scenario_result = await db.execute(
            select(
                SessionRecord.scenario_id,
                func.count().label("usage_count"),
                func.sum(case((SessionRecord.status == "completed", 1), else_=0)).label("completed_count"),
            )
            .group_by(SessionRecord.scenario_id)
        )
        scenario_stats = []
        for sr in scenario_result.all():
            # Get scores for this scenario
            sc_reports = await db.execute(
                select(SessionRecord.final_report)
                .where(
                    SessionRecord.scenario_id == sr.scenario_id,
                    SessionRecord.status == "completed",
                    SessionRecord.final_report.isnot(None),
                )
            )
            sc_scores = []
            for report in sc_reports.scalars().all():
                if isinstance(report, dict):
                    s = report.get("summary", {}).get("overall_score")
                    if s is not None:
                        sc_scores.append(s)

            scenario_stats.append({
                "scenario_id": sr.scenario_id,
                "name": scenario_map.get(sr.scenario_id, sr.scenario_id),
                "usage_count": sr.usage_count or 0,
                "avg_score": round(sum(sc_scores) / len(sc_scores)) if sc_scores else 0,
                "completion_rate": round((sr.completed_count or 0) / sr.usage_count, 2) if sr.usage_count else 0,
            })

        return {
            "overall": {
                "total_users": user_count,
                "total_sessions": total_sessions,
                "avg_score": avg_score,
                "active_today": active_today,
            },
            "user_stats": user_stats,
            "scenario_stats": scenario_stats,
        }
