"""Shared set-entry validation.

These rules are enforced in BOTH the Pydantic request schema and the service
layer. The service copy is not redundant: the Phase-4 agent tool path builds
``log_set`` calls straight from untrusted LLM tool input and never passes through
the REST request model, so the schema validation would be skipped there. Keeping
the canonical rules here (raising :class:`ValidationError`) makes the service the
single source of truth; the schema mirrors these bounds for early 422s on the
REST path.
"""

from __future__ import annotations

from app.services.errors import ValidationError

MIN_REPS = 1
MAX_REPS = 100
MIN_RIR = 0.0
MAX_RIR = 10.0
# RIR is recorded in half-rep increments ("0.5 increments allowed").
RIR_INCREMENT = 0.5


def validate_set_fields(weight: float, reps: int, rir: float | None) -> None:
    """Validate a single set's weight / reps / rir.

    Raises :class:`ValidationError` (which routes map to HTTP 422) on any breach:
    ``weight >= 0``; ``1 <= reps <= 100``; ``rir`` either ``None`` or within
    ``[0, 10]`` in 0.5 increments. Values above ~5 RIR are unusual but valid (a
    warmup set can have high RIR), so only the bounds and increment are checked.
    """
    if weight < 0:
        raise ValidationError("weight must be >= 0")
    if not (MIN_REPS <= reps <= MAX_REPS):
        raise ValidationError(f"reps must be between {MIN_REPS} and {MAX_REPS}")
    if rir is not None:
        if not (MIN_RIR <= rir <= MAX_RIR):
            raise ValidationError(f"rir must be between {MIN_RIR} and {MAX_RIR}")
        # Guard against float dust (e.g. 0.1 + 0.2) before the increment check.
        remainder = round(rir / RIR_INCREMENT, 6)
        if remainder != round(remainder):
            raise ValidationError("rir must be in 0.5 increments")
