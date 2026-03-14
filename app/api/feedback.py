from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.schemas import (
    EvaluatorMetricsResponse,
    SubmitFeedbackRequest,
    SubmitFeedbackResponse,
)
from app.models.db import FeedbackRecord, MessageRecord, async_session_factory

router = APIRouter(tags=["feedback"])


@router.post(
    "/sessions/{session_id}/messages/{message_id}/feedback",
    response_model=SubmitFeedbackResponse,
)
async def submit_feedback(
    session_id: str, message_id: str, req: SubmitFeedbackRequest
):
    async with async_session_factory() as db:
        # Get the message and its AI analysis
        result = await db.execute(
            select(MessageRecord).where(
                MessageRecord.id == message_id,
                MessageRecord.session_id == session_id,
            )
        )
        message = result.scalar_one_or_none()
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        if message.role != "user":
            raise HTTPException(
                status_code=400, detail="Feedback can only be submitted for user messages"
            )

        # Extract AI labels from analysis
        ai_analysis = message.analysis or {}
        ai_labels = {}
        for point in ai_analysis.get("analysis", []):
            pid = point.get("point_id", "")
            ai_labels[pid] = {
                "covered": (
                    point.get("newly_matched", False)
                    and point.get("match_level") == "full"
                    and point.get("confidence", 0) >= 0.7
                ),
                "confidence": point.get("confidence", 0),
                "match_level": point.get("match_level", "none"),
            }

        # Compare human vs AI
        human_labels_dict = {
            pid: label.model_dump() for pid, label in req.human_labels.items()
        }
        discrepancies = []
        agree_count = 0
        total_count = 0

        for pid, human in req.human_labels.items():
            total_count += 1
            ai = ai_labels.get(pid, {"covered": False, "confidence": 0})
            if human.covered == ai.get("covered", False):
                agree_count += 1
            else:
                discrepancies.append({
                    "point_id": pid,
                    "ai_judgment": ai.get("covered", False),
                    "human_judgment": human.covered,
                    "ai_confidence": ai.get("confidence", 0),
                    "human_comment": human.comment,
                })

        agreement_rate = agree_count / total_count if total_count > 0 else 1.0

        # Persist feedback
        feedback = FeedbackRecord(
            id=str(uuid.uuid4()),
            session_id=session_id,
            message_id=message_id,
            human_labels=human_labels_dict,
            ai_labels=ai_labels,
            agreement_rate=agreement_rate,
        )
        db.add(feedback)
        await db.commit()

    return {
        "agreement_rate": agreement_rate,
        "discrepancies": discrepancies,
    }


@router.get("/admin/evaluator-metrics", response_model=EvaluatorMetricsResponse)
async def get_evaluator_metrics():
    async with async_session_factory() as db:
        result = await db.execute(select(FeedbackRecord))
        feedbacks = result.scalars().all()

    if not feedbacks:
        return {
            "total_labeled_turns": 0,
            "accuracy": 0.0,
            "precision_by_point": {},
            "common_discrepancies": [],
        }

    # Aggregate metrics
    total_turns = len(feedbacks)
    total_agree = 0
    total_comparisons = 0

    # Per-point tracking: tp, fp, fn, tn
    point_stats: dict[str, dict] = {}
    discrepancy_details: list[dict] = []

    for fb in feedbacks:
        human = fb.human_labels or {}
        ai = fb.ai_labels or {}

        for pid, h_label in human.items():
            if pid not in point_stats:
                point_stats[pid] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

            h_covered = h_label.get("covered", False)
            a_info = ai.get(pid, {})
            a_covered = a_info.get("covered", False)

            total_comparisons += 1
            if h_covered == a_covered:
                total_agree += 1
                if h_covered:
                    point_stats[pid]["tp"] += 1
                else:
                    point_stats[pid]["tn"] += 1
            else:
                if a_covered and not h_covered:
                    point_stats[pid]["fp"] += 1
                    discrepancy_details.append({
                        "point_id": pid,
                        "type": "false_positive",
                        "ai_confidence": a_info.get("confidence", 0),
                        "human_comment": h_label.get("comment", ""),
                    })
                else:
                    point_stats[pid]["fn"] += 1
                    discrepancy_details.append({
                        "point_id": pid,
                        "type": "false_negative",
                        "ai_confidence": a_info.get("confidence", 0),
                        "human_comment": h_label.get("comment", ""),
                    })

    accuracy = total_agree / total_comparisons if total_comparisons > 0 else 0.0

    precision_by_point = {}
    for pid, stats in point_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_by_point[pid] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "total_samples": tp + fp + fn + stats["tn"],
        }

    # Summarize common discrepancies
    from collections import Counter
    fp_counter = Counter()
    fn_counter = Counter()
    for d in discrepancy_details:
        if d["type"] == "false_positive":
            fp_counter[d["point_id"]] += 1
        else:
            fn_counter[d["point_id"]] += 1

    common = []
    for pid, count in fp_counter.most_common(5):
        common.append({
            "point_id": pid,
            "issue": "false_positive",
            "count": count,
            "description": f"{pid} 被AI误判为已覆盖 ({count}次)",
        })
    for pid, count in fn_counter.most_common(5):
        common.append({
            "point_id": pid,
            "issue": "false_negative",
            "count": count,
            "description": f"{pid} 被AI漏判为未覆盖 ({count}次)",
        })

    return {
        "total_labeled_turns": total_turns,
        "accuracy": round(accuracy, 3),
        "precision_by_point": precision_by_point,
        "common_discrepancies": common,
    }
