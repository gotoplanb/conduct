"""Tests for should_fan_out_shadows — the fan-out gate (#36).

A bench run (force_shadows) must fan out across the fleet even when the primary
fails, so a flaky primary doesn't take the whole comparison with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eval.shadow_runner import should_fan_out_shadows
from models.types import JobStatus


@dataclass
class _Job:
    status: str
    job_metadata: dict = field(default_factory=dict)


def test_complete_primary_fans_out() -> None:
    assert should_fan_out_shadows(_Job(JobStatus.COMPLETE.value)) is True


def test_failed_primary_without_force_does_not_fan_out() -> None:
    assert should_fan_out_shadows(_Job(JobStatus.FAILED.value)) is False


def test_failed_primary_with_force_fans_out() -> None:
    # The #36 case: a force_shadows bench primary that failed (e.g. malformed
    # artifact) still fans out so the other models are compared.
    assert should_fan_out_shadows(
        _Job(JobStatus.FAILED.value, {"force_shadows": True})
    ) is True


def test_complete_with_force_fans_out() -> None:
    assert should_fan_out_shadows(
        _Job(JobStatus.COMPLETE.value, {"force_shadows": True})
    ) is True
