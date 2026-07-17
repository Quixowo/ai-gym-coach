"""SSE stream event vocabulary.

The orchestrator and chat route yield these events; the chat route serializes each
to one ``data: <json>\\n\\n`` SSE frame. The frontend trace panel switches on the
``type`` field. Payload keys match exactly:

- ``text_delta``          -> ``{"type", "text"}``
- ``tool_call_started``   -> ``{"type", "tools"}``
- ``tool_call_completed`` -> ``{"type", "tool", "latency_ms", "result_summary"}``
- ``turn_complete``       -> ``{"type", "iterations", "total_latency_ms"}``
- ``error``               -> ``{"type", "message"}``

Kept as small dataclasses with an ``as_dict()`` so the wire shape is defined in one
place; the route calls ``json.dumps(event.as_dict())``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextDeltaEvent:
    text: str
    type: str = "text_delta"

    def as_dict(self) -> dict:
        return {"type": self.type, "text": self.text}


@dataclass
class ToolCallStartedEvent:
    tools: list[str] = field(default_factory=list)
    type: str = "tool_call_started"

    def as_dict(self) -> dict:
        return {"type": self.type, "tools": self.tools}


@dataclass
class ToolCallCompletedEvent:
    tool: str
    latency_ms: int
    result_summary: str
    type: str = "tool_call_completed"

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "tool": self.tool,
            "latency_ms": self.latency_ms,
            "result_summary": self.result_summary,
        }


@dataclass
class TurnCompleteEvent:
    iterations: int
    total_latency_ms: int
    type: str = "turn_complete"

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "iterations": self.iterations,
            "total_latency_ms": self.total_latency_ms,
        }


@dataclass
class ErrorEvent:
    message: str
    type: str = "error"

    def as_dict(self) -> dict:
        return {"type": self.type, "message": self.message}
