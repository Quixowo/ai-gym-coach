"""RAG retrieval/synthesis tests (spec §9.2/§10.3) — fully mocked, no live API.

Both provider factories are patched at their module of use
(``app.services.knowledge_service``): a fake Voyage client returns a canned query
embedding, and a fake Anthropic client pops scripted responses (synthesis first,
groundedness second). Retrieval itself runs against the real Postgres+pgvector
through the NullPool test session maker — NEVER the app's pooled engine
(LESSONS.md closed-loop family) — with deterministic 1024-dim unit vectors so
cosine ranking is predictable.

Also covers the ``search_knowledge_base`` tool handler (CLAUDE.md rule 4: a
provider outage must become ``{"error": ...}``, never an exception into the SSE
stream). Per-call kwargs are snapshotted in the fakes before storage — mutable
kwargs alias otherwise (LESSONS.md stream-kwargs entry).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

import app.services.knowledge_service as knowledge_service_module
from app.agent.tools.handlers import execute_tool
from app.core.config import settings
from app.models.knowledge_chunk import KnowledgeChunk
from app.services.knowledge_service import search_knowledge_base
from tests.conftest import test_session_maker as session_maker

EMBED_DIM = 1024


def _vec(hot_index: int) -> list[float]:
    """A deterministic 1024-dim unit vector; distinct indexes are orthogonal."""
    v = [0.0] * EMBED_DIM
    v[hot_index] = 1.0
    return v


# --------------------------------------------------------------------------- #
# Fake provider clients (patched at knowledge_service's import boundary)
# --------------------------------------------------------------------------- #
class _FakeEmbedResult:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class _FakeVoyage:
    """Returns one canned query embedding per call, or raises."""

    def __init__(
        self, embedding: list[float] | None = None, raises: Exception | None = None
    ) -> None:
        self._embedding = embedding if embedding is not None else _vec(0)
        self._raises = raises
        self.embed_calls: list[dict] = []

    async def embed(self, texts, **kwargs):
        if self._raises is not None:
            raise self._raises
        # Snapshot mutable args at call time (LESSONS.md stream-kwargs entry).
        self.embed_calls.append({"texts": list(texts), **dict(kwargs)})
        return _FakeEmbedResult([self._embedding])


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    """Pops scripted response texts in order: synthesis first, groundedness second."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.create_calls: list[dict] = []

    async def create(self, **kwargs):
        snapshot = dict(kwargs)
        snapshot["messages"] = [dict(m) for m in kwargs.get("messages", [])]
        self.create_calls.append(snapshot)
        return _FakeResponse(self._responses.pop(0))


class _FakeAnthropic:
    def __init__(self, responses: list[str]) -> None:
        self.messages = _FakeMessages(responses)


def _patch_clients(
    monkeypatch: pytest.MonkeyPatch, voyage: _FakeVoyage, anthropic_client: _FakeAnthropic
) -> None:
    monkeypatch.setattr(knowledge_service_module, "get_voyage_client", lambda: voyage)
    monkeypatch.setattr(knowledge_service_module, "get_anthropic_client", lambda: anthropic_client)


# --------------------------------------------------------------------------- #
# DB fixtures/helpers — NullPool test session maker only (LESSONS.md)
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
async def _empty_knowledge_chunks():
    """Start every test from an empty knowledge_chunks table (shared dev DB)."""
    async with session_maker() as db:
        await db.execute(delete(KnowledgeChunk))
        await db.commit()
    yield


async def _insert_chunk(
    db: AsyncSession,
    *,
    title: str,
    text: str,
    embedding: list[float],
    category: str = "training",
    index: int = 0,
    citation: str | None = None,
) -> None:
    db.add(
        KnowledgeChunk(
            document_title=title,
            category=category,
            chunk_text=text,
            chunk_index=index,
            source_citation=citation,
            embedding=embedding,
        )
    )
    await db.commit()


# --------------------------------------------------------------------------- #
# Service — happy path
# --------------------------------------------------------------------------- #
async def test_happy_path_returns_grounded_answer_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voyage = _FakeVoyage(embedding=_vec(0))
    anthropic_client = _FakeAnthropic(["Protein: aim for 1.6-2.2 g/kg.", "GROUNDED"])
    _patch_clients(monkeypatch, voyage, anthropic_client)

    async with session_maker() as db:
        await _insert_chunk(
            db,
            title="Protein Intake for Muscle Growth",
            text="# Protein Intake for Muscle Growth\n## How Much\n1.6-2.2 g/kg per day.",
            embedding=_vec(0),
            category="nutrition",
        )
        result = await search_knowledge_base(db, "how much protein should I eat")

    assert result == {
        "answer": "Protein: aim for 1.6-2.2 g/kg.",
        "sources": [
            {
                "document_title": "Protein Intake for Muscle Growth",
                "category": "nutrition",
                "source_citation": None,
            }
        ],
        "groundedness_passed": True,
    }

    # Query embedded with input_type="query" and the configured model (§9.1/§9.2).
    assert voyage.embed_calls == [
        {
            "texts": ["how much protein should I eat"],
            "model": settings.EMBED_MODEL_ID,
            "input_type": "query",
        }
    ]
    # Two Haiku calls: synthesis (400) then groundedness (10) — §9.2/§10.3.
    synthesis_call, groundedness_call = anthropic_client.messages.create_calls
    assert synthesis_call["max_tokens"] == 400
    assert synthesis_call["model"] == settings.HAIKU_MODEL_ID
    assert groundedness_call["max_tokens"] == 10
    assert "GROUNDED or UNGROUNDED" in groundedness_call["messages"][0]["content"]


async def test_synthesis_prompt_contains_numbered_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voyage = _FakeVoyage(embedding=_vec(0))
    anthropic_client = _FakeAnthropic(["Answer.", "GROUNDED"])
    _patch_clients(monkeypatch, voyage, anthropic_client)

    nearest_text = "# Doc A\n## Section\nNearest chunk body."
    farther_text = "# Doc B\n## Section\nFarther chunk body."
    async with session_maker() as db:
        # _vec(0) is distance 0 from the query; _vec(1) is orthogonal — deterministic rank.
        await _insert_chunk(db, title="Doc A", text=nearest_text, embedding=_vec(0))
        await _insert_chunk(db, title="Doc B", text=farther_text, embedding=_vec(1))
        await search_knowledge_base(db, "what is progressive overload")

    prompt = anthropic_client.messages.create_calls[0]["messages"][0]["content"]
    assert f"[1] {nearest_text}" in prompt
    assert f"[2] {farther_text}" in prompt
    assert "Question: what is progressive overload" in prompt
    assert "ONLY the reference material" in prompt


# --------------------------------------------------------------------------- #
# Service — groundedness failure keeps sources (§9.2 fallback)
# --------------------------------------------------------------------------- #
async def test_groundedness_failure_returns_fallback_with_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voyage = _FakeVoyage(embedding=_vec(0))
    anthropic_client = _FakeAnthropic(["A made-up claim.", "UNGROUNDED"])
    _patch_clients(monkeypatch, voyage, anthropic_client)

    async with session_maker() as db:
        await _insert_chunk(
            db,
            title="Common Shoulder Injuries in Lifters",
            text="# Common Shoulder Injuries in Lifters\n## Strains\nBody.",
            embedding=_vec(0),
            category="injury_prevention",
            citation="Shoulder pain overview, Mayo Clinic — https://example.org/shoulder",
        )
        result = await search_knowledge_base(db, "shoulder hurts when pressing")

    assert result["groundedness_passed"] is False
    # Conservative answer text, but retrieval is NOT thrown away — sources with the
    # citation survive so the user can read the material directly.
    assert "couldn't fully confirm" in result["answer"]
    assert result["sources"] == [
        {
            "document_title": "Common Shoulder Injuries in Lifters",
            "category": "injury_prevention",
            "source_citation": "Shoulder pain overview, Mayo Clinic — https://example.org/shoulder",
        }
    ]


async def test_groundedness_check_is_strict_equality(monkeypatch: pytest.MonkeyPatch) -> None:
    # Anything other than exactly "GROUNDED" (after strip) fails the check —
    # same strict one-word convention as the injury classifier.
    voyage = _FakeVoyage(embedding=_vec(0))
    anthropic_client = _FakeAnthropic(["Answer.", "GROUNDED, mostly"])
    _patch_clients(monkeypatch, voyage, anthropic_client)

    async with session_maker() as db:
        await _insert_chunk(db, title="Doc", text="# Doc\n## S\nBody.", embedding=_vec(0))
        result = await search_knowledge_base(db, "q")

    assert result["groundedness_passed"] is False


# --------------------------------------------------------------------------- #
# Service — empty retrieval and source dedup
# --------------------------------------------------------------------------- #
async def test_empty_table_returns_no_results_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    voyage = _FakeVoyage(embedding=_vec(0))
    anthropic_client = _FakeAnthropic([])  # must never be called
    _patch_clients(monkeypatch, voyage, anthropic_client)

    async with session_maker() as db:
        result = await search_knowledge_base(db, "anything at all")

    assert result == {
        "answer": "No relevant information found in the knowledge base.",
        "sources": [],
        "groundedness_passed": True,
    }
    assert anthropic_client.messages.create_calls == []


async def test_sources_deduped_by_document_title(monkeypatch: pytest.MonkeyPatch) -> None:
    voyage = _FakeVoyage(embedding=_vec(0))
    anthropic_client = _FakeAnthropic(["Answer.", "GROUNDED"])
    _patch_clients(monkeypatch, voyage, anthropic_client)

    async with session_maker() as db:
        # Two chunks from the same document + one from another.
        await _insert_chunk(
            db, title="Deload Protocols", text="# D\n## When\nBody.", embedding=_vec(0), index=0
        )
        await _insert_chunk(
            db, title="Deload Protocols", text="# D\n## How\nBody.", embedding=_vec(1), index=1
        )
        await _insert_chunk(
            db, title="RIR-Based Autoregulation", text="# R\n## Why\nBody.", embedding=_vec(2)
        )
        result = await search_knowledge_base(db, "when should I deload")

    titles = [s["document_title"] for s in result["sources"]]
    assert sorted(titles) == ["Deload Protocols", "RIR-Based Autoregulation"]
    assert len(titles) == len(set(titles))  # no document listed twice


# --------------------------------------------------------------------------- #
# Tool handler — passthrough, input validation, provider outage (CLAUDE.md rule 4)
# --------------------------------------------------------------------------- #
async def test_handler_returns_service_dict_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    voyage = _FakeVoyage(embedding=_vec(0))
    anthropic_client = _FakeAnthropic(["Grounded answer.", "GROUNDED"])
    _patch_clients(monkeypatch, voyage, anthropic_client)

    async with session_maker() as db:
        await _insert_chunk(db, title="Doc", text="# Doc\n## S\nBody.", embedding=_vec(0))
        result = await execute_tool(
            "search_knowledge_base", {"query": "how to warm up"}, uuid.uuid4(), db
        )

    assert result["answer"] == "Grounded answer."
    assert result["groundedness_passed"] is True
    assert result["sources"][0]["document_title"] == "Doc"


@pytest.mark.parametrize("tool_input", [{}, {"query": ""}, {"query": "   "}])
async def test_handler_missing_or_blank_query_returns_error(
    monkeypatch: pytest.MonkeyPatch, tool_input: dict
) -> None:
    # Clients patched to fail loudly if reached — validation must short-circuit.
    _patch_clients(monkeypatch, _FakeVoyage(raises=AssertionError("no call")), _FakeAnthropic([]))

    async with session_maker() as db:
        result = await execute_tool("search_knowledge_base", tool_input, uuid.uuid4(), db)

    assert result == {"error": "query is required"}


async def test_handler_voyage_outage_returns_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An embedding-provider exception must become {"error": ...}, never propagate
    # into the SSE stream (CLAUDE.md rule 4).
    _patch_clients(monkeypatch, _FakeVoyage(raises=RuntimeError("voyage down")), _FakeAnthropic([]))

    async with session_maker() as db:
        result = await execute_tool("search_knowledge_base", {"query": "protein"}, uuid.uuid4(), db)

    assert result == {"error": "internal error executing search_knowledge_base"}
