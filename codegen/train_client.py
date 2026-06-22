"""Client for the external MLX DPO training sidecar (#45).

Transport + contract only: ship a base-model name + preference pairs to the
local training sidecar (a native-macOS daemon on the M5 wrapping mlx-tune) and
return the resulting tag + provenance. The heavy ML stack lives entirely in the
sidecar — Conduct carries no training dependencies, exactly like the media
providers (ComfyUI/ACE-Step) and the rust-build sandbox.

Local-only: the sidecar runs on owned hardware; there is no cloud equivalent
(the code-gen flywheel is local-only). On a cloud worker `dpo_train_url` won't
resolve and a dpo_fine_tune job fails cleanly with :class:`TrainServiceError`.

Authoritative contract the sidecar implements:

    POST /train
      {"base_model", "pairs": [{prompt, system, chosen, rejected}, ...],
       "training": {epochs, lora_rank, lora_alpha, beta, learning_rate}?}
    -> {"tag", "artifact_path", "pairs_consumed", "training_time_s", "dataset_sha"}
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

# Training is slow (~30-60 min for a 4B model on an M-series Mac); the client
# waits generously. The sidecar caps the run on its side.
DEFAULT_TRAIN_TIMEOUT_S = 7200.0


@dataclass
class TrainResult:
    tag: str
    artifact_path: str
    pairs_consumed: int
    training_time_s: float
    dataset_sha: str

    def as_metadata(self) -> dict:
        """The provenance block stored on the job (metadata.training)."""
        return {
            "tag": self.tag, "artifact_path": self.artifact_path,
            "pairs_consumed": self.pairs_consumed,
            "training_time_s": self.training_time_s, "dataset_sha": self.dataset_sha,
        }


class TrainServiceError(RuntimeError):
    """The training sidecar was unreachable or returned a non-200 envelope. The
    dpo_fine_tune job should fail cleanly rather than register a phantom tag."""


class DpoTrainClient:
    """Thin httpx client for the DPO training sidecar. Mirrors RustBuildClient
    (base_url + timeout, injectable transport for tests)."""

    def __init__(
        self, base_url: str, *, timeout_s: float = DEFAULT_TRAIN_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    async def train(
        self, *, base_model: str, pairs: list[dict], training: dict | None = None
    ) -> TrainResult:
        payload: dict = {"base_model": base_model, "pairs": pairs}
        if training:
            payload["training"] = training
        try:
            async with httpx.AsyncClient(
                base_url=self._base, timeout=self._timeout_s, transport=self._transport
            ) as client:
                resp = await client.post("/train", json=payload)
        except httpx.HTTPError as e:
            raise TrainServiceError(f"training sidecar unreachable: {e}") from e
        if resp.status_code != 200:
            raise TrainServiceError(
                f"training sidecar error {resp.status_code}: {resp.text[:300]}"
            )
        return _parse_result(resp.json())


def _parse_result(body: dict) -> TrainResult:
    tag = body.get("tag")
    if not tag:
        raise TrainServiceError(f"training sidecar returned no tag: {body!r}"[:300])
    return TrainResult(
        tag=str(tag),
        artifact_path=str(body.get("artifact_path") or ""),
        pairs_consumed=int(body.get("pairs_consumed") or 0),
        training_time_s=float(body.get("training_time_s") or 0.0),
        dataset_sha=str(body.get("dataset_sha") or ""),
    )
