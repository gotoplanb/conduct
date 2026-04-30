from enum import StrEnum


class Sensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


# Higher value = stricter. Used to take the max of (job, rule) sensitivities so
# rules can act as a floor that clients cannot relax.
_LEVEL = {
    Sensitivity.PUBLIC: 0,
    Sensitivity.INTERNAL: 1,
    Sensitivity.CONFIDENTIAL: 2,
}


def stricter(a: Sensitivity, b: Sensitivity) -> Sensitivity:
    return a if _LEVEL[a] >= _LEVEL[b] else b


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
