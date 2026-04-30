from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(alias="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    admin_key: str = Field(alias="CONDUCT_ADMIN_KEY")

    default_model: str = Field(default="llama3.3:70b", alias="DEFAULT_MODEL")
    default_sensitive_model: str = Field(default="llama3.3:70b", alias="DEFAULT_SENSITIVE_MODEL")

    otel_endpoint: str = Field(default="http://localhost:4317", alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str = Field(default="conduct", alias="OTEL_SERVICE_NAME")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
