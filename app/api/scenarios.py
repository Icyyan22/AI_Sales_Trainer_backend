from __future__ import annotations

import json
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.auth import require_admin
from app.api.schemas import ScenarioListResponse
from app.models.db import UserRecord
from app.models.scenario import SCENARIOS_DIR, Scenario, list_scenarios, load_scenario

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


class CreateScenarioRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    customer_name: str = Field(..., min_length=1)
    customer_role: str = Field(..., min_length=1)
    customer_hospital: str = ""
    customer_background: str = ""
    customer_concerns: list[str] = []
    customer_speaking_style: str = ""
    product_name: str = Field(..., min_length=1)
    product_selling_points: list[str] = []
    semantic_points: list[dict] = Field(..., min_length=1)
    opening_message: str = Field(..., min_length=1)


@router.get("", response_model=ScenarioListResponse)
async def get_scenarios():
    scenarios = list_scenarios()
    return {
        "scenarios": [
            {
                "id": s.id,
                "name": s.name,
                "stage": s.stage,
                "customer_profile": s.customer_profile.model_dump(),
                "product": s.product.model_dump(),
                "semantic_points": [sp.model_dump() for sp in s.semantic_points],
                "opening_message": s.opening_message,
            }
            for s in scenarios
        ]
    }


@router.post("", status_code=201)
async def create_scenario(req: CreateScenarioRequest, _: UserRecord = Depends(require_admin)):
    # Generate a safe ID from the name
    scenario_id = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", req.name).strip("_").lower()
    if not scenario_id:
        scenario_id = "custom_scenario"

    # Ensure unique
    path = SCENARIOS_DIR / f"{scenario_id}.json"
    suffix = 1
    while path.exists():
        scenario_id = f"{scenario_id}_{suffix}"
        path = SCENARIOS_DIR / f"{scenario_id}.json"
        suffix += 1

    data = {
        "id": scenario_id,
        "name": req.name,
        "stage": "value_delivery",
        "customer_profile": {
            "name": req.customer_name,
            "role": req.customer_role,
            "hospital": req.customer_hospital,
            "background": req.customer_background,
            "concerns": req.customer_concerns,
            "speaking_style": req.customer_speaking_style,
        },
        "product": {
            "name": req.product_name,
            "selling_points": req.product_selling_points,
        },
        "semantic_points": [
            {
                "id": sp.get("id", f"SP{i+1}"),
                "name": sp.get("name", f"语义点{i+1}"),
                "description": sp.get("description", ""),
                "match_examples": sp.get("match_examples", []),
                "non_match_examples": sp.get("non_match_examples", []),
            }
            for i, sp in enumerate(req.semantic_points)
        ],
        "opening_message": req.opening_message,
    }

    # Validate via Scenario model
    try:
        scenario = Scenario(**data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Write to disk
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Clear lru_cache so new scenario is discoverable
    load_scenario.cache_clear()

    return {"id": scenario.id, "name": scenario.name}
