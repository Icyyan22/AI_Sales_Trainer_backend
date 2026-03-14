from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "TRAINER_", "env_file": ".env"}

    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_api_base: str | None = None

    # Lightweight model for evaluator / state_updater (fast structured tasks)
    lite_model: str | None = None

    evaluator_temperature: float = 0.0
    persona_temperature: float = 0.7
    llm_max_tokens: int = 8192

    confidence_threshold: float = 0.7
    max_turns: int = 10
    max_context_turns: int = 4

    database_url: str = "sqlite+aiosqlite:///./trainer.db"
    prompt_version: str = "v1"

    super_admin_username: str | None = None  # First registered user auto-becomes super_admin if not set


settings = Settings()
