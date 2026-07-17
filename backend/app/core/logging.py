"""Minimal structured (JSON) logging to stdout with in-code secret redaction.

This is a deliberately small slice of full observability — the rest
(request IDs, latency, tool-trace correlation) lands in Phase 6. What matters
*now* is that credentials never reach a log sink: register/login events log
identifying fields (email, user_id) but any field named like a secret is
replaced with ``"[REDACTED]"`` **before serialization**, enforced in a logging
``Filter`` so it can't be forgotten at a call site.

Usage::

    from app.core.logging import get_logger
    log = get_logger(__name__)
    log.info("user_registered", extra={"email": email, "user_id": str(uid)})

Never pass a plaintext (or hashed) password in ``extra`` and rely on redaction
as a safety net — but the redaction is there so an accidental one is scrubbed.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

# Field/argument names whose values must never appear in logs. Compared
# case-insensitively against both ``extra`` keys and ``%``-style args dict keys.
_REDACT_KEYS = frozenset(
    {
        "password",
        "hashed_password",
        "access_token",
        "refresh_token",
        "cookie",
        "authorization",
    }
)

_REDACTED = "[REDACTED]"

# Standard LogRecord attributes we don't want to re-emit as structured fields;
# anything else attached to the record is treated as caller-supplied context.
_RESERVED_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
    }
)


def _redact(key: str, value: Any) -> Any:
    return _REDACTED if key.lower() in _REDACT_KEYS else value


class RedactionFilter(logging.Filter):
    """Scrub secret-named fields on the record before any formatter sees them.

    Runs as a ``logging.Filter`` (not just inside the formatter) so redaction is
    enforced structurally: it applies to every handler on the logger regardless
    of formatter, and returns ``True`` to keep the record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Redact structured context attached via ``extra=...``.
        for attr in list(record.__dict__.keys()):
            if attr in _RESERVED_ATTRS or attr.startswith("_"):
                continue
            record.__dict__[attr] = _redact(attr, record.__dict__[attr])

        # Redact %-style dict args, e.g. log.info("...", {"password": ...}).
        if isinstance(record.args, dict):
            record.args = {k: _redact(k, v) for k, v in record.args.items()}
        return True


class JsonFormatter(logging.Formatter):
    """Serialize a LogRecord to a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        # Merge any caller-supplied structured context (already redacted).
        for key, value in record.__dict__.items():
            if key in _RESERVED_ATTRS or key.startswith("_") or key == "message":
                continue
            payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure_root() -> None:
    """Attach a single stdout JSON handler + redaction filter to the app logger.

    Idempotent — guarded so repeated imports / test reloads don't stack
    handlers and duplicate every line.
    """
    app_logger = logging.getLogger("app")
    if getattr(app_logger, "_gym_coach_configured", False):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactionFilter())

    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(handler)
    # Don't also emit through the root logger's default (non-JSON) handler.
    app_logger.propagate = False
    app_logger._gym_coach_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """Return a JSON-emitting, redaction-filtered logger under the ``app`` tree."""
    _configure_root()
    # Namespacing under "app" so the single configured handler/filter applies.
    if name == "app" or name.startswith("app."):
        return logging.getLogger(name)
    return logging.getLogger(f"app.{name}")
