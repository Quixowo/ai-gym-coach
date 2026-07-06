from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/gym_coach"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Auth
    JWT_SECRET: str = "dev-secret-change-me-not-for-production-0123456789"  # ≥32 bytes: PyJWT warns on short HS256 keys
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


settings = Settings()
