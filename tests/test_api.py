"""API integration tests (no LLM calls - tests route structure only)."""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.db import init_db


@pytest_asyncio.fixture
async def client():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealthCheck:
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "database" in data


class TestScenarios:
    async def test_list_scenarios(self, client):
        resp = await client.get("/api/v1/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert "scenarios" in data
        assert len(data["scenarios"]) >= 1
        assert data["scenarios"][0]["id"] == "diabetes_drug"


class TestSessions:
    async def test_create_session(self, client):
        resp = await client.post(
            "/api/v1/sessions",
            json={"scenario_id": "diabetes_drug"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert data["phase"] == "active"
        assert data["semantic_coverage"] == {"SP1": False, "SP2": False, "SP3": False}

    async def test_create_session_not_found(self, client):
        resp = await client.post(
            "/api/v1/sessions",
            json={"scenario_id": "nonexistent"},
        )
        assert resp.status_code == 404

    async def test_get_session(self, client):
        create_resp = await client.post(
            "/api/v1/sessions",
            json={"scenario_id": "diabetes_drug"},
        )
        session_id = create_resp.json()["session_id"]

        resp = await client.get(f"/api/v1/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["status"] == "active"

    async def test_get_messages(self, client):
        create_resp = await client.post(
            "/api/v1/sessions",
            json={"scenario_id": "diabetes_drug"},
        )
        session_id = create_resp.json()["session_id"]

        resp = await client.get(f"/api/v1/sessions/{session_id}/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "assistant"

    async def test_create_session_with_difficulty(self, client):
        resp = await client.post(
            "/api/v1/sessions",
            json={"scenario_id": "diabetes_drug", "difficulty": "hard"},
        )
        assert resp.status_code == 201

    async def test_create_session_invalid_difficulty(self, client):
        resp = await client.post(
            "/api/v1/sessions",
            json={"scenario_id": "diabetes_drug", "difficulty": "impossible"},
        )
        assert resp.status_code == 422


class TestFeedback:
    async def test_feedback_message_not_found(self, client):
        resp = await client.post(
            "/api/v1/sessions/fake-session/messages/fake-msg/feedback",
            json={"human_labels": {"SP1": {"covered": True}}},
        )
        assert resp.status_code == 404

    async def test_evaluator_metrics_empty(self, client):
        resp = await client.get("/api/v1/admin/evaluator-metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_labeled_turns"] == 0
        assert data["accuracy"] == 0.0
