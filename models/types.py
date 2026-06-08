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


class Sampling(StrEnum):
    """Per-task generation profile, declared by RoutingRule (overridable
    per-request). Bundles temperature + seed policy so the two knobs stay
    coherent — exposing raw temperature alone invites "fixed seed + high temp =
    the same 'random' output forever". The engine maps each profile to concrete
    params (see routing.engine.sampling_params):

      deterministic — temperature 0.0 + a seed derived from the input, so the
        same prompt+context yields the same output (a bio generator). Real
        reproducibility is local-model only; cloud (Anthropic/Bedrock) has no
        seed param, so it's temperature 0 best-effort.
      balanced      — temperature 0.7, random seed. The default.
      creative      — temperature 1.0, random seed: same prompt, different
        output each call (a dad-joke generator).
    """

    DETERMINISTIC = "deterministic"
    BALANCED = "balanced"
    CREATIVE = "creative"


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
