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


class MediaKind(StrEnum):
    """Declared by RoutingRule. Tells the worker which dispatch path to take:
    `text` (the default) uses the existing BaseProvider.complete; everything
    else routes through BaseMediaProvider.produce and writes Job.media_url.
    `mux` is the ffmpeg composition primitive — no model, just shells out."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MUX = "mux"
