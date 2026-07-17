"""Domain-exception hierarchy for the service layer (Phase 3a).

The service functions are the single source of truth shared by two callers with
very different error conventions:

- the REST routes (Phase 3), which need HTTP status codes, and
- the agent tool handlers (Phase 4), which must return a plain
  ``{"error": "..."}`` dict and never raise.

To keep the services free of any HTTP coupling, they raise these plain domain
exceptions. Routes catch them and translate to the appropriate status code
(``ValidationError``/``LoadJumpCapError`` -> 422, ``NotFoundError`` -> 404,
``ConflictError`` -> 409); the Phase-4 tool handlers will catch the same base
class and turn ``str(exc)`` into the ``{"error": ...}`` payload. Neither concern
leaks into the other.
"""

from __future__ import annotations


class ServiceError(Exception):
    """Base class for all service-layer domain errors.

    A Phase-4 tool handler can catch this single type and render
    ``{"error": str(exc)}`` without caring which subclass was raised.
    """


class ValidationError(ServiceError):
    """Input failed a service-layer validation rule.

    Routes translate this to HTTP 422. Raised in addition to (not instead of)
    the mirrored Pydantic constraints, because the agent tool path constructs
    service calls from raw tool input and bypasses the REST request schema.
    """


class NotFoundError(ServiceError):
    """A referenced row does not exist, or is not owned by the current user.

    Cross-user access is deliberately reported as "not found" rather than
    "forbidden" so ownership is never leaked (application-level access control).
    Routes translate this to HTTP 404.
    """


class ConflictError(ServiceError):
    """The request conflicts with current state (e.g. an open session exists).

    Routes translate this to HTTP 409.
    """


class LoadJumpCapError(ValidationError):
    """A program update would raise a target weight beyond the 10% cap.

    Subclasses :class:`ValidationError` so it maps to HTTP 422 like any other
    rejected write. Carries the exercise / prior / requested values so both the
    route detail and a Phase-4 tool's ``{"error": ...}`` string can name them.
    """

    def __init__(self, exercise_id: str, prior: float, requested: float) -> None:
        self.exercise_id = exercise_id
        self.prior = prior
        self.requested = requested
        super().__init__(
            f"Requested weight increase for exercise {exercise_id} exceeds the "
            f"10% safety cap (prior: {prior}, requested: {requested})."
        )
