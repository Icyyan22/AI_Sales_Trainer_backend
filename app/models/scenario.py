from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class SemanticPoint(BaseModel):
    id: str
    name: str
    description: str
    match_examples: list[str] = []
    non_match_examples: list[str] = []


class CustomerProfile(BaseModel):
    name: str
    role: str
    hospital: str
    background: str
    concerns: list[str] = []
    speaking_style: str = ""


class ProductInfo(BaseModel):
    name: str
    selling_points: list[str] = []


class Scenario(BaseModel):
    id: str
    name: str
    stage: str = "value_delivery"
    customer_profile: CustomerProfile
    product: ProductInfo
    semantic_points: list[SemanticPoint]
    opening_message: str

    def get_coverage_init(self) -> dict[str, bool]:
        return {sp.id: False for sp in self.semantic_points}

    def get_semantic_points_text(self) -> str:
        lines = []
        for sp in self.semantic_points:
            lines.append(f"- {sp.id} ({sp.name}): {sp.description}")
            if sp.match_examples:
                lines.append(f"  匹配示例: {'; '.join(sp.match_examples)}")
            if sp.non_match_examples:
                lines.append(f"  不匹配示例: {'; '.join(sp.non_match_examples)}")
        return "\n".join(lines)


SCENARIOS_DIR = Path(__file__).parent.parent / "data" / "scenarios"


@lru_cache(maxsize=32)
def load_scenario(scenario_id: str) -> Scenario:
    path = SCENARIOS_DIR / f"{scenario_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Scenario(**data)


def list_scenarios() -> list[Scenario]:
    scenarios = []
    for path in SCENARIOS_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        scenarios.append(Scenario(**data))
    return scenarios
