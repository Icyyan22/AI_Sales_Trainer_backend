from __future__ import annotations

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    scenario_id: str = "diabetes_drug"
    difficulty: str = Field(default="normal", pattern="^(easy|normal|hard)$")


class CreateSessionResponse(BaseModel):
    session_id: str
    scenario: dict
    opening_message: dict
    semantic_coverage: dict[str, bool]
    phase: str


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000)


class SendMessageResponse(BaseModel):
    user_message: dict
    assistant_message: dict
    state: dict


class SessionStatusResponse(BaseModel):
    session_id: str
    scenario_id: str
    status: str
    difficulty: str
    semantic_coverage: dict[str, bool] | None = None
    phase: str | None = None
    turn_count: int | None = None
    customer_attitude: str | None = None
    created_at: str


class MessageItem(BaseModel):
    id: str
    role: str
    content: str
    turn: int
    analysis: dict | None = None
    strategy: dict | None = None


class MessageListResponse(BaseModel):
    session_id: str
    messages: list[MessageItem]


class CompleteSessionResponse(BaseModel):
    session_id: str
    status: str
    final_coverage: dict[str, bool]


class ReportResponse(BaseModel):
    session_id: str
    summary: dict
    semantic_detail: list[dict]
    skill_radar: dict
    feedback: dict


class SessionListItem(BaseModel):
    session_id: str
    scenario_id: str
    status: str
    difficulty: str
    created_at: str
    completed_at: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionListItem]


class ScenarioListResponse(BaseModel):
    scenarios: list[dict]


# --- Phase 4-5: Feedback & Admin ---

class PointFeedback(BaseModel):
    covered: bool
    comment: str | None = None


class SubmitFeedbackRequest(BaseModel):
    human_labels: dict[str, PointFeedback]


class SubmitFeedbackResponse(BaseModel):
    agreement_rate: float
    discrepancies: list[dict]


class EvaluatorMetricsResponse(BaseModel):
    total_labeled_turns: int
    accuracy: float
    precision_by_point: dict[str, dict]
    common_discrepancies: list[dict]
