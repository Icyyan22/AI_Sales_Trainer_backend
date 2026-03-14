from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _load_metadata() -> dict:
    path = PROMPTS_DIR / "metadata.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_latest_version() -> str:
    return _load_metadata()["latest_version"]


@lru_cache(maxsize=32)
def get_prompt(agent: str, version: str | None = None) -> str:
    if version is None:
        version = get_latest_version()
    path = PROMPTS_DIR / version / f"{agent}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")
