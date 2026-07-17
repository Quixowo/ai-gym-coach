"""Injury red-flag classifier.

A single cheap Haiku call, run in the chat route *before* the agent loop, that
decides whether the incoming message describes a potential acute injury (needs
professional care) versus routine soreness / a general question. On a flag the
route short-circuits to :data:`FIXED_INJURY_REDIRECT_RESPONSE` — a hard-coded
string, never an LLM generation, so the wording can't drift under adversarial
phrasing.

Strictness + fail-open tradeoff (documented for review): the classifier requires
an exact ``ACUTE_INJURY`` token; anything else — ``ROUTINE``, unexpected wording,
whitespace, or an API exception — returns ``False`` (treat as routine, let the
main agent handle it). This is a deliberate *availability-over-caution* choice:
Haiku being down must not block every chat message, and the system prompt +
Coach's persona still steer injury language toward professional care as a second
layer. The cost is that a genuine acute-injury message slips through if the
classifier is unavailable — accepted, but flagged here so the orchestrator can
revisit if it prefers fail-closed.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.client import get_anthropic_client

log = get_logger(__name__)

_ACUTE_LABEL = "ACUTE_INJURY"

_CLASSIFIER_PROMPT = (
    "Classify whether this message describes a potential ACUTE injury "
    "(something that may need professional medical attention) as opposed "
    "to normal training soreness, fatigue, or a general question.\n\n"
    'Message: "{message}"\n\n'
    "Respond with exactly one word: ACUTE_INJURY or ROUTINE."
)

FIXED_INJURY_REDIRECT_RESPONSE = (
    "That sounds like it could be more than routine training soreness. I'm not "
    "able to help troubleshoot potential injuries — please see a doctor or "
    "physical therapist to get it properly evaluated. I'm glad to help with "
    "training and nutrition once you're cleared to train normally again."
)


async def classify_acute_injury(message: str) -> bool:
    """Return True iff Haiku labels ``message`` as a potential acute injury.

    Strict one-word check: only an exact ``ACUTE_INJURY`` (after strip) returns
    True. Any other response, or any exception (API down, unexpected shape), is
    logged and returns ``False`` — fail-open to ROUTINE (see module docstring).
    """
    try:
        client = get_anthropic_client()
        response = await client.messages.create(
            model=settings.HAIKU_MODEL_ID,
            max_tokens=10,
            messages=[{"role": "user", "content": _CLASSIFIER_PROMPT.format(message=message)}],
        )
        text = response.content[0].text.strip()
    except Exception:  # noqa: BLE001 — availability over caution: never block chat
        log.exception("injury_classifier_error")
        return False

    if text == _ACUTE_LABEL:
        return True
    if text != "ROUTINE":
        # Unexpected label — log it (helps tune the prompt) but treat as routine.
        log.warning("injury_classifier_unexpected_label", extra={"label": text})
    return False
