"""AWS Bedrock provider using the Converse API.

Bedrock's Converse API is a unified surface that maps a single
messages/system/inference-config shape to each underlying foundation model's
native protocol. One provider class works for Anthropic-on-Bedrock, Llama,
Mistral, Cohere, Nova, and friends without per-family adapters.

Credentials are passed in by the registry on a per-call basis (decrypted
from the owning ClientApp's `bedrock_creds_encrypted` blob). Two auth
styles are supported:

  - **Long-term API key (bearer token).** Generated in the Bedrock console;
    sent as `Authorization: Bearer <token>` via a botocore event hook so
    we avoid the process-global `AWS_BEARER_TOKEN_BEDROCK` env var (unsafe
    under concurrent multi-tenant shadow fan-out).
  - **IAM access key + secret.** Traditional SigV4 signing via boto3's
    default credential chain.

There is no host-IAM fallback by design — a client without its own creds
simply can't route to Bedrock, mirroring the per-client Anthropic-key
rule.
"""

from __future__ import annotations

import time

import aioboto3
from botocore.exceptions import ClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.pricing import get_pricing
from observability.metrics import record_retry
from providers.base import (
    BaseProvider,
    ProviderError,
    ProviderRateLimit,
    ProviderResponse,
)


def _record_bedrock_retry(retry_state) -> None:  # type: ignore[no-untyped-def]
    record_retry(provider="bedrock", reason="throttling")


class BedrockProvider(BaseProvider):
    name = "bedrock"

    def __init__(
        self,
        *,
        region: str,
        access_key_id: str = "",
        secret_access_key: str = "",
        bearer_token: str = "",
    ) -> None:
        has_pair = bool(access_key_id and secret_access_key)
        has_bearer = bool(bearer_token)
        if has_pair and has_bearer:
            raise ValueError(
                "BedrockProvider takes either (access_key_id + secret_access_key) "
                "OR bearer_token, not both"
            )
        if not has_pair and not has_bearer:
            raise ValueError(
                "BedrockProvider requires either (access_key_id + secret_access_key) "
                "or bearer_token"
            )
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._bearer_token = bearer_token
        self._region = region
        # aioboto3 sessions are cheap; we hold one per provider instance so
        # the underlying httpx/aiohttp client is reused across calls within
        # the same request.
        self._session = aioboto3.Session()

    @retry(
        retry=retry_if_exception_type(ProviderRateLimit),
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
        before_sleep=_record_bedrock_retry,
    )
    async def complete(
        self,
        prompt: str,
        model: str,
        system_prompt: str = "",
        max_tokens: int = 1000,
        temperature: float | None = None,
        seed: int | None = None,
        **kwargs: object,
    ) -> ProviderResponse:
        inference_config: dict = {"maxTokens": max_tokens}
        # Bedrock Converse has no seed param — `seed` is ignored; a
        # "deterministic" task falls back to temperature 0 (best-effort, no
        # reproducibility guarantee).
        if temperature is not None:
            inference_config["temperature"] = temperature
        converse_kwargs: dict = {
            "modelId": model,
            "messages": [
                {"role": "user", "content": [{"text": prompt}]},
            ],
            "inferenceConfig": inference_config,
        }
        if system_prompt:
            converse_kwargs["system"] = [{"text": system_prompt}]

        # Use real AKID+secret if we have them, otherwise placeholder strings
        # are enough to satisfy boto3's session construction — the actual
        # Authorization header gets overwritten by the bearer-token event
        # hook below before the request goes out.
        ak = self._access_key_id or "conduct-bearer-placeholder"
        sk = self._secret_access_key or "conduct-bearer-placeholder"
        started = time.perf_counter()
        async with self._session.client(
            "bedrock-runtime",
            region_name=self._region,
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
        ) as client:
            if self._bearer_token:
                self._install_bearer_auth(client, self._bearer_token)
            try:
                resp = await client.converse(**converse_kwargs)
            except ClientError as e:
                err_code = e.response.get("Error", {}).get("Code", "")
                # Bedrock surfaces rate limits as ThrottlingException; everything
                # else is a non-retryable provider error from Conduct's POV.
                if err_code in {"ThrottlingException", "TooManyRequestsException"}:
                    raise ProviderRateLimit(str(e)) from e
                raise ProviderError(f"bedrock {err_code}: {e}") from e
        latency_ms = int((time.perf_counter() - started) * 1000)

        # Converse responses follow a stable envelope regardless of model:
        # output.message.content is a list of content blocks; the text we
        # want lives in the `text` field of each block.
        message = resp.get("output", {}).get("message", {})
        text_parts = [
            block["text"]
            for block in message.get("content", [])
            if "text" in block
        ]
        text = "".join(text_parts)
        usage = resp.get("usage", {})
        tokens_in = int(usage.get("inputTokens", 0))
        tokens_out = int(usage.get("outputTokens", 0))

        return ProviderResponse(
            response=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=get_pricing().cost(self.name, model, tokens_in, tokens_out),
            latency_ms=latency_ms,
            model_used=model,
            provider=self.name,
        )

    @staticmethod
    def _install_bearer_auth(client, token: str) -> None:
        """Register a botocore event hook that overwrites the Authorization
        header with a bearer token after SigV4 signing but before the request
        is sent. This is per-client (the hook closes over `token`), so it's
        safe under concurrent shadow fan-out — unlike the
        `AWS_BEARER_TOKEN_BEDROCK` env var, which is process-global."""

        def _override_auth(request, **_kwargs):
            # `request` is a botocore AWSPreparedRequest at before-send time;
            # mutating headers here replaces SigV4's Authorization with ours.
            request.headers["Authorization"] = f"Bearer {token}"
            # SigV4 also stamps these; bearer auth doesn't use them and a
            # few Bedrock endpoints reject a mix of signing styles.
            for stale in ("X-Amz-Date", "X-Amz-Security-Token", "X-Amz-Content-SHA256"):
                request.headers.pop(stale, None)

        client.meta.events.register("before-send.bedrock-runtime.*", _override_auth)
