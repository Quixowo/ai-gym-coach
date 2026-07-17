"""Replay harness shared by the three eval suites.

The eval suites consume ONLY committed JSON fixtures under ``claude_responses/``
(CI never calls a live model). This module turns those
recordings back into the fake SDK surfaces the production code expects, so the
*real* orchestrator / classifier / knowledge service run against recorded model
output:

- :class:`FakeAnthropicClient` — replays recorded ``messages.create`` responses
  (classifier verdicts, RAG synthesis + groundedness) and recorded
  ``messages.stream`` iterations (agent turns), popping one per call in order.
- :class:`FakeVoyageClient` — returns a canned query embedding (no live Voyage).
- :func:`load_fixture` — read a committed fixture JSON by name.
- :func:`rank_embedding` — synthetic 1024-dim unit vectors whose cosine
  similarity to the rank-0 query vector strictly decreases with rank, so
  recorded chunks re-inserted into pgvector retrieve in the recorded order
  without storing real embeddings in the fixtures.

The stream fakes mirror ``tests/test_orchestrator.py``'s pattern (an async
context manager that is also an async iterator of text deltas and exposes
``get_final_message``); here the content blocks come from a recording instead
of being hand-built. ``messages`` kwargs are snapshotted at
call time because the orchestrator mutates one list across iterations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "claude_responses"

EMBED_DIM = 1024  # voyage-4-lite output dimension (matches the DB column)


def load_fixture(name: str) -> dict:
    """Load a committed fixture JSON (``claude_responses/{name}.json``)."""
    path = FIXTURES_DIR / f"{name}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def query_embedding() -> list[float]:
    """The synthetic query vector every groundedness replay embeds to."""
    v = [0.0] * EMBED_DIM
    v[0] = 1.0
    return v


def rank_embedding(rank: int) -> list[float]:
    """A unit vector whose cosine similarity to :func:`query_embedding` is
    ``1 - 0.05 * rank`` — strictly decreasing in ``rank``, so pgvector's
    cosine-distance ordering reproduces the recorded retrieval order exactly.
    """
    similarity = 1.0 - 0.05 * rank
    v = [0.0] * EMBED_DIM
    v[0] = similarity
    v[rank + 1] = math.sqrt(1.0 - similarity * similarity)
    return v


# --------------------------------------------------------------------------- #
# Content-block objects: rebuild the SDK's attribute-access shape from recorded
# JSON dicts so production code (which reads ``block.type`` / ``block.input`` /
# ``block.text``) works unchanged.
# --------------------------------------------------------------------------- #
class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    def __init__(self, block_id: str, name: str, tool_input: dict) -> None:
        self.type = "tool_use"
        self.id = block_id
        self.name = name
        self.input = tool_input


def _rebuild_blocks(recorded_blocks: list[dict]) -> list:
    """Turn recorded content-block dicts into attribute-access block objects."""
    out: list = []
    for block in recorded_blocks:
        if block["type"] == "tool_use":
            out.append(_ToolUseBlock(block["id"], block["name"], block.get("input", {})))
        else:
            out.append(_TextBlock(block.get("text", "")))
    return out


class _FinalMessage:
    def __init__(self, content: list) -> None:
        self.content = content


class _Delta:
    def __init__(self, text: str) -> None:
        self.type = "text_delta"
        self.text = text


class _ContentBlockDeltaEvent:
    def __init__(self, text: str) -> None:
        self.type = "content_block_delta"
        self.delta = _Delta(text)


class _ReplayStream:
    """Async context manager + iterator standing in for ``messages.stream(...)``.

    Emits the recorded assistant turn's text as one ``text_delta`` event, then
    finalizes to a message carrying the recorded content blocks (text + any
    tool_use) — exactly what the orchestrator's loop consumes.
    """

    def __init__(self, content_blocks: list[dict]) -> None:
        self._blocks = _rebuild_blocks(content_blocks)
        self._text = "".join(b.text for b in self._blocks if b.type == "text")

    async def __aenter__(self) -> _ReplayStream:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def __aiter__(self):
        if self._text:
            yield _ContentBlockDeltaEvent(self._text)

    async def get_final_message(self) -> _FinalMessage:
        return _FinalMessage(list(self._blocks))


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_TextBlock(text)]


class _FakeMessages:
    """Replays recorded model outputs, popping one per call in recorded order.

    Supports both surfaces the production code uses:

    - ``create(**kwargs)`` -> pops the next scripted text as a response object
      (classifier: one call; RAG: synthesis then groundedness verdict).
    - ``stream(**kwargs)`` -> pops the next recorded iteration's content blocks
      as a :class:`_ReplayStream` (agent loop).

    Running out of scripted responses raises — a replay that makes more model
    calls than the recording did is a real regression, not something to paper
    over with a default response.
    """

    def __init__(
        self,
        create_texts: list[str] | None = None,
        stream_iterations: list[list[dict]] | None = None,
    ) -> None:
        self._create_texts = list(create_texts or [])
        self._stream_iterations = list(stream_iterations or [])
        self.create_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def create(self, **kwargs):
        snapshot = dict(kwargs)
        snapshot["messages"] = [dict(m) for m in kwargs.get("messages", [])]
        self.create_calls.append(snapshot)
        if not self._create_texts:
            raise AssertionError("no recorded create() response left to replay")
        return _FakeResponse(self._create_texts.pop(0))

    def stream(self, **kwargs):
        # Snapshot messages at call time — the orchestrator reuses one mutable
        # list and appends after each call.
        snapshot = dict(kwargs)
        snapshot["messages"] = list(kwargs.get("messages", []))
        self.stream_calls.append(snapshot)
        if not self._stream_iterations:
            raise AssertionError("no recorded stream() iteration left to replay")
        return _ReplayStream(self._stream_iterations.pop(0))


class FakeAnthropicClient:
    """Drop-in for the shared Anthropic client, driven by recorded fixtures."""

    def __init__(
        self,
        create_texts: list[str] | None = None,
        stream_iterations: list[list[dict]] | None = None,
    ) -> None:
        self.messages = _FakeMessages(create_texts, stream_iterations)


class _FakeEmbedResult:
    def __init__(self, embedding: list[float]) -> None:
        self.embeddings = [embedding]


class FakeVoyageClient:
    """Returns one canned query embedding per ``embed`` call (no live Voyage)."""

    def __init__(self, embedding: list[float] | None = None) -> None:
        self._embedding = embedding if embedding is not None else query_embedding()
        self.embed_calls: list[dict] = []

    async def embed(self, texts, **kwargs):
        self.embed_calls.append({"texts": list(texts), **dict(kwargs)})
        return _FakeEmbedResult(self._embedding)


# --------------------------------------------------------------------------- #
# Fixture-driven DB setup + id rewriting for tool-correctness replay
# --------------------------------------------------------------------------- #
def rewrite_ids(obj: object, id_map: dict[str, str]) -> object:
    """Deep-replace recorded exercise UUID strings with test-DB ones.

    Recorded tool_use inputs contain exercise ids resolved against the DB the
    recording ran on; those rows don't exist (or have different ids) in the test
    DB. The fixture carries ``id_map`` (recorded id -> exercise NAME); the test
    resolves each name in its own DB and calls this to rewrite every occurrence,
    so the replayed model messages reference rows that actually exist.
    """
    if isinstance(obj, str):
        for old, new in id_map.items():
            obj = obj.replace(old, new)
        return obj
    if isinstance(obj, list):
        return [rewrite_ids(item, id_map) for item in obj]
    if isinstance(obj, dict):
        return {k: rewrite_ids(v, id_map) for k, v in obj.items()}
    return obj
