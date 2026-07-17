from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor .env resolution to the repo layout, not the process CWD: the shared .env
# lives at the repo root, and a CWD-relative "env_file='.env'" silently
# loads nothing when uvicorn/pytest run from backend/. Later entries win, so a
# backend-local .env can override the root one.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_ENV_FILES = (_BACKEND_DIR.parent / ".env", _BACKEND_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES, env_file_encoding="utf-8", extra="ignore"
    )

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/gym_coach"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Auth
    # ≥32 bytes: PyJWT warns on short HS256 keys
    JWT_SECRET: str = "dev-secret-change-me-not-for-production-0123456789"
    JWT_ACCESS_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_EXPIRE_DAYS: int = 7

    # LLM / AI
    ANTHROPIC_API_KEY: str = ""
    VOYAGE_API_KEY: str = ""
    SONNET_MODEL_ID: str = "claude-sonnet-4-6"
    HAIKU_MODEL_ID: str = "claude-haiku-4-5-20251001"
    EMBED_MODEL_ID: str = "voyage-4-lite"

    # CORS / cookies
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]
    COOKIE_SAMESITE: str = "lax"
    COOKIE_SECURE: bool = False

    # Episodic memory pipeline
    # Distinct conversations a (user, topic_key) must appear in before its
    # observations are consolidated into a durable user_memories row.
    MEMORY_CONSOLIDATION_THRESHOLD: int = 3
    # Max consolidated memories injected into the system prompt per turn.
    MEMORY_MAX_INJECTED: int = 15


settings = Settings()
