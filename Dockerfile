# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:0.5-python${PYTHON_VERSION}-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Baked at build time so containerized prompts can still report a git SHA in
# job metadata (the .git directory is excluded from the image). Pass via:
#   docker build --build-arg GIT_SHA=$(git rev-parse HEAD) ...
ARG GIT_SHA=""
ENV CONDUCT_GIT_SHA=$GIT_SHA \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ffmpeg — needed by the TTS path to convert Piper's WAV output to MP3
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1001 conduct \
 && useradd --system --uid 1001 --gid conduct --no-create-home conduct

WORKDIR /app

COPY --from=builder --chown=conduct:conduct /app /app

USER conduct

EXPOSE 8000 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
