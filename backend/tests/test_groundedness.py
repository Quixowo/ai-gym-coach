"""RAG groundedness eval — recorded fixtures, no live API.

Each ``claude_responses/grd_*.json`` fixture was recorded once against the live
corpus + Haiku (see ``tests/fixtures/record_fixtures.py``): the top-5 retrieved
chunks, the synthesis answer, and the groundedness verdict for one question. Here we
re-insert those chunks into the test DB with :func:`rank_embedding` synthetic vectors
(NEVER real embeddings in fixtures — see the ``_replay`` contract), point
the Voyage/Anthropic factories at fakes replaying the recording, and run the REAL
``search_knowledge_base`` — asserting it reproduces the recorded outcome exactly.

15 questions are answerable (12 ``training`` + 3 ``injury_prevention``); 4 are traps
with no good answer in the corpus (one nutrition-flavoured), used to confirm the
pipeline refuses rather than fabricates. The aggregate tests report the pass rate and
the trap-refusal rate — the citable numbers.

This suite wipes ``knowledge_chunks`` in the shared dev DB (like
``test_knowledge_service``); the corpus must be re-ingested before any live RAG work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import delete

import app.services.knowledge_service as knowledge_service_module
from app.models.knowledge_chunk import KnowledgeChunk
from app.services.knowledge_service import search_knowledge_base
from tests.conftest import test_session_maker as session_maker
from tests.fixtures._replay import FakeAnthropicClient, FakeVoyageClient, rank_embedding

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "claude_responses"
GRD_FIXTURES = sorted(FIXTURES_DIR.glob("grd_*.json"))

# Baselines from the recorded run (see recording_metrics.json). The recorded verdicts
# are frozen in the fixtures, so these fractions are deterministic in CI — these gate a
# *regression in the replay code / a fixture edit*, not live-model drift.
MIN_ANSWERABLE_PASS = 15  # recorded: 15/15 (training 12/12, injury_prevention 3/3)
MIN_TRAPS_REFUSED = 4  # recorded: 4/4


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


ALL = [_load(p) for p in GRD_FIXTURES]
ANSWERABLE = [f for f in ALL if not f["trap"]]
TRAPS = [f for f in ALL if f["trap"]]


@pytest.fixture(autouse=True)
async def _empty_knowledge_chunks():
    """Start every test from an empty knowledge_chunks table (shared dev DB)."""
    async with session_maker() as db:
        await db.execute(delete(KnowledgeChunk))
        await db.commit()
    yield


async def _insert_recorded_chunks(db, fixture: dict) -> None:
    """Re-insert the recorded top-k chunks with rank-preserving synthetic embeddings."""
    for rank, chunk in enumerate(fixture["chunks"]):
        db.add(
            KnowledgeChunk(
                document_title=chunk["document_title"],
                category=chunk["category"],
                chunk_text=chunk["chunk_text"],
                chunk_index=chunk["chunk_index"],
                source_citation=chunk["source_citation"],
                embedding=rank_embedding(rank),
            )
        )
    await db.commit()


def _patch_clients(monkeypatch: pytest.MonkeyPatch, fixture: dict) -> None:
    fake_anthropic = FakeAnthropicClient(
        create_texts=[fixture["synthesis_text"], fixture["groundedness_verdict"]]
    )
    monkeypatch.setattr(knowledge_service_module, "get_voyage_client", lambda: FakeVoyageClient())
    monkeypatch.setattr(knowledge_service_module, "get_anthropic_client", lambda: fake_anthropic)


@pytest.mark.parametrize("fixture", ALL, ids=[f["id"] for f in ALL])
async def test_replays_recorded_outcome(monkeypatch: pytest.MonkeyPatch, fixture: dict) -> None:
    """The real search_knowledge_base reproduces the recorded answer/sources/verdict."""
    _patch_clients(monkeypatch, fixture)

    async with session_maker() as db:
        await _insert_recorded_chunks(db, fixture)
        result = await search_knowledge_base(db, fixture["query"])

    expected = fixture["expected"]
    assert result["groundedness_passed"] == expected["groundedness_passed"]
    assert result["answer"] == expected["answer"]
    assert result["sources"] == expected["sources"]


def test_answerable_pass_rate() -> None:
    """Report + gate the answerable groundedness pass rate."""
    passed = sum(1 for f in ANSWERABLE if f["expected"]["groundedness_passed"])
    assert len(ANSWERABLE) == 15
    assert passed >= MIN_ANSWERABLE_PASS, f"groundedness pass rate {passed}/15 below floor"


def test_pass_rate_by_category() -> None:
    """Every injury_prevention answer must stay grounded (citation-critical)."""
    injury = [f for f in ANSWERABLE if f["category"] == "injury_prevention"]
    assert injury, "expected injury_prevention questions"
    assert all(f["expected"]["groundedness_passed"] for f in injury)


def test_traps_refused_not_fabricated() -> None:
    """Traps must be refused (no-answer / conservative fallback), never fabricated."""
    assert len(TRAPS) == 4
    refused = sum(1 for f in TRAPS if f["trap_refused"])
    assert refused >= MIN_TRAPS_REFUSED, f"only {refused}/4 traps refused"


def test_injury_sources_carry_citations() -> None:
    """Grounded injury answers surface their source citation."""
    for f in ANSWERABLE:
        if f["category"] != "injury_prevention":
            continue
        citations = [s["source_citation"] for s in f["expected"]["sources"]]
        assert any(c for c in citations), f"{f['id']} injury answer lacks a citation"
