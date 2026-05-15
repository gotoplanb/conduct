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

    # Comma-separated Ollama models that the worker pins on startup and
    # promises never to evict. The API can call these directly without going
    # through the worker queue, enabling parallel fan-out for real-time eval.
    resident_models_raw: str = Field(default="", alias="RESIDENT_MODELS")

    # TTS — directory holding Piper voice files (.onnx + .onnx.json) and the
    # output directory where generated MP3s land. Both default to repo-relative
    # paths suitable for local dev; in containers, mount these as volumes.
    tts_voices_dir: str = Field(default="./voices", alias="TTS_VOICES_DIR")
    tts_default_voice: str = Field(default="en_US-amy-medium", alias="TTS_DEFAULT_VOICE")
    tts_output_dir: str = Field(default="./output", alias="TTS_OUTPUT_DIR")
    tts_max_chars: int = Field(default=10_000, alias="TTS_MAX_CHARS")

    otel_endpoint: str = Field(default="http://localhost:4317", alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str = Field(default="conduct", alias="OTEL_SERVICE_NAME")

    # Grafana base URL used by the UI to build deep-links into Tempo for a
    # specific job's trace. Set to empty to suppress the link.
    grafana_base_url: str = Field(default="http://localhost:3000", alias="GRAFANA_BASE_URL")

    # Set to true when the UI is served over HTTPS so the admin session cookie
    # carries the Secure flag. Local HTTP development leaves this false.
    ui_cookie_secure: bool = Field(default=False, alias="UI_COOKIE_SECURE")

    @property
    def resident_models(self) -> list[str]:
        return [m.strip() for m in self.resident_models_raw.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
