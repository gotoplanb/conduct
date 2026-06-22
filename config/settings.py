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

    # Master key (Fernet) for encrypting per-client secrets at rest. Generate
    # once with `python -c "from cryptography.fernet import Fernet;
    # print(Fernet.generate_key().decode())"`. Required to set or use any
    # per-client API key (e.g. a client's own Anthropic key); without it the
    # secrets layer raises on encrypt/decrypt.
    secrets_key: str | None = Field(default=None, alias="CONDUCT_SECRETS_KEY")

    # Public HTTPS origin Conduct is reachable at (the tunnel / reverse proxy).
    # Used as the OAuth issuer and to build the authorize/token URLs that go
    # into a Claude custom connector. Must match what external clients hit, so
    # set it to the public URL in prod; localhost is fine for local dev.
    public_base_url: str = Field(default="http://localhost:8000", alias="CONDUCT_PUBLIC_URL")

    # TTL for single-use job eval tokens (the credential-less scoring link).
    eval_token_ttl_days: int = Field(default=7, alias="EVAL_TOKEN_TTL_DAYS")

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

    # Watchtower's local-only rust-build sandbox (gotoplanb/watchtower#1) — the
    # toolchain runner for the code-gen eval flywheel (#24). host.docker.internal
    # because it's a separate compose project, same as the ComfyUI provider.
    # Code generation is local-only, so there is no cloud value for this.
    rust_build_url: str = Field(
        default="http://host.docker.internal:8055", alias="RUST_BUILD_URL"
    )

    # The external MLX DPO training sidecar (#45) — a native-macOS daemon on the
    # M5 wrapping mlx-tune. host.docker.internal because, like ComfyUI/ACE-Step,
    # it runs native (MLX needs Metal) beside the worker, not in Docker. Local-
    # only: a cloud worker won't have this and dpo_fine_tune fails cleanly.
    dpo_train_url: str = Field(
        default="http://host.docker.internal:8077", alias="DPO_TRAIN_URL"
    )

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
