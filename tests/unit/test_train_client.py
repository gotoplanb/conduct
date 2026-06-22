"""Unit tests for codegen.train_client — the DPO training sidecar transport
(#45). Stubs the HTTP layer with httpx.MockTransport; live mlx-tune training is
validated against the real sidecar separately."""

from __future__ import annotations

import json

import httpx
import pytest

from codegen.train_client import DpoTrainClient, TrainServiceError


def _client(handler) -> DpoTrainClient:
    return DpoTrainClient("http://train", transport=httpx.MockTransport(handler))


_OK = {
    "tag": "gemma4-e4b-dpo-abc", "artifact_path": "/m5/x.gguf",
    "pairs_consumed": 12, "training_time_s": 99.5, "dataset_sha": "deadbeef",
}


async def test_train_parses_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_OK)

    r = await _client(handler).train(base_model="gemma4:e4b", pairs=[{"chosen": "a"}])
    assert r.tag == "gemma4-e4b-dpo-abc" and r.pairs_consumed == 12
    assert r.training_time_s == 99.5 and r.dataset_sha == "deadbeef"
    assert r.as_metadata()["tag"] == "gemma4-e4b-dpo-abc"


async def test_train_sends_base_model_pairs_and_training() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_OK)

    await _client(handler).train(
        base_model="gemma4:e4b", pairs=[{"chosen": "a", "rejected": "b"}],
        training={"epochs": 1, "beta": 0.1},
    )
    assert seen["base_model"] == "gemma4:e4b"
    assert seen["pairs"] == [{"chosen": "a", "rejected": "b"}]
    assert seen["training"] == {"epochs": 1, "beta": 0.1}


async def test_train_omits_training_when_none() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_OK)

    await _client(handler).train(base_model="m", pairs=[])
    assert "training" not in seen


async def test_train_non_200_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="cuda oom")

    with pytest.raises(TrainServiceError, match="500"):
        await _client(handler).train(base_model="m", pairs=[])


async def test_train_unreachable_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(TrainServiceError, match="unreachable"):
        await _client(handler).train(base_model="m", pairs=[])


async def test_train_missing_tag_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"artifact_path": "/x"})  # no tag

    with pytest.raises(TrainServiceError, match="no tag"):
        await _client(handler).train(base_model="m", pairs=[])
