"""Memory-extraction correctness eval (4th eval suite) — recorded fixtures, no live API.

Mirrors the Phase 7 eval suites (``test_tool_correctness`` / ``test_groundedness`` /
``test_red_flag_recall``): committed JSON fixtures under ``claude_responses/`` carry
the raw model reply, the REAL production code runs against a fake client replaying it,
and CI never touches a live API.

The extraction step calls ``client.messages.create`` and reads ``resp.content[0].text``
(a JSON array), exactly like the classifier/RAG surfaces. So each ``mem_*.json`` fixture
stores that raw reply text in ``recorded_reply`` — the same field a live recorder would
overwrite with ``resp.content[0].text`` — plus the ``expected`` observations DERIVED by
running the production ``_parse_observations`` over that reply. Deriving ``expected``
that way makes the fixture self-consistent and gives a live gate a deterministic
recompute path (see ``tests/fixtures/README_memory_fixtures.md``).

Each fixture declares its recording status via ``hand_authored`` (presence asserted in
:func:`test_all_fixtures_declare_recording_status`): ``true`` = hand-authored pending a
live recording, ``false`` = recorded from a real Haiku reply by the memory phase gate
(which drops the real reply into ``recorded_reply`` and regenerates ``expected`` — same
shape, no test changes needed). Adversarial fixtures stay hand-authored by design.

``get_anthropic_client`` is patched at the ``memory_service`` module boundary (it is
imported there specifically so tests patch it here), so no live Haiku call is made.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select

import app.services.memory_service as memory_service
from app.models.memory_observation import MemoryObservation
from app.models.user_memory import UserMemory
from app.services.memory_service import _parse_observations, process_turn
from tests.conftest import test_session_maker as session_maker
from tests.fixtures._replay import FakeAnthropicClient
from tests.helpers import create_db_user

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "claude_responses"
MEM_FIXTURES = sorted(FIXTURES_DIR.glob("mem_*.json"))


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _sort_key(obs: dict) -> tuple[str, str, str]:
    return (obs["category"], obs["topic_key"], obs["content"])


def _canonical(observations: list[dict]) -> list[dict]:
    return sorted(
        (
            {"category": o["category"], "topic_key": o["topic_key"], "content": o["content"]}
            for o in observations
        ),
        key=_sort_key,
    )


async def _seed_existing_keys(db, user_id: uuid.UUID, keys: list[str]) -> None:
    """One prior observation per existing key, each in its OWN conversation."""
    for i, key in enumerate(keys):
        db.add(
            MemoryObservation(
                user_id=user_id,
                conversation_id=uuid.uuid4(),
                category="preferences",
                topic_key=key,
                content=f"Prior observation about {key}.",
            )
        )
    await db.commit()


# --------------------------------------------------------------------------- #
# Fixture integrity — expected is exactly what the parser yields on the raw reply
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", MEM_FIXTURES, ids=lambda p: p.stem)
def test_expected_matches_parser(path: Path) -> None:
    """``expected`` must equal ``_parse_observations(recorded_reply)`` (recompute path)."""
    fixture = _load(path)
    assert _parse_observations(fixture["recorded_reply"]) == fixture["expected"]


def test_all_fixtures_declare_recording_status() -> None:
    """Every mem_ fixture must carry the ``hand_authored`` status flag.

    ``True`` = hand-authored pending live recording; ``False`` = a live gate wrote a
    real Haiku reply into ``recorded_reply``. Adversarial fixtures (malformed model
    output a live model wouldn't reproduce) stay ``True`` permanently by design, so
    only the flag's presence — not its value — is an invariant.
    """
    assert MEM_FIXTURES, "no mem_*.json fixtures found"
    for path in MEM_FIXTURES:
        assert isinstance(_load(path).get("hand_authored"), bool), (
            f"{path.stem} missing hand_authored flag"
        )


# --------------------------------------------------------------------------- #
# End-to-end: extraction reply -> stored observation rows (right cat/key/content)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", MEM_FIXTURES, ids=lambda p: p.stem)
async def test_extraction_stores_expected_rows(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """``process_turn`` stores exactly the fixture's ``expected`` observations.

    Covers the rich-facts multi-item case, the []-empty case, and every guardrail
    fixture (invalid category / empty topic_key / non-dict rows / >300-char content /
    prose+code-fence wrapping / unparseable text): invalid rows dropped, valid rows
    kept with normalized keys and truncated content, and it never raises.
    """
    fixture = _load(path)

    async with session_maker() as db:
        user_id = await create_db_user(db)
        await _seed_existing_keys(db, user_id, fixture.get("existing_keys", []))

        fake = FakeAnthropicClient(create_texts=[fixture["recorded_reply"]])
        monkeypatch.setattr(memory_service, "get_anthropic_client", lambda: fake)

        conversation_id = uuid.uuid4()
        await process_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=fixture["user_message"],
            assistant_reply=fixture["assistant_reply"],
            db=db,
        )

        # Only observations from THIS turn's conversation (excludes any pre-seeded keys).
        rows = (
            (
                await db.execute(
                    select(MemoryObservation).where(
                        MemoryObservation.user_id == user_id,
                        MemoryObservation.conversation_id == conversation_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        stored = _canonical(
            [{"category": r.category, "topic_key": r.topic_key, "content": r.content} for r in rows]
        )
        assert stored == _canonical(fixture["expected"])

        # Fresh users on a single conversation never cross the threshold, so extraction
        # is the ONLY model call — no consolidation attempted.
        assert len(fake.messages.create_calls) == 1


# --------------------------------------------------------------------------- #
# Topic-key reuse: existing keys are fed into the prompt, a reused key accumulates
# --------------------------------------------------------------------------- #
async def test_topic_key_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing keys reach the extraction prompt; a reused key accumulates under it."""
    fixture = _load(FIXTURES_DIR / "mem_02_topic_reuse.json")
    existing_keys = fixture["existing_keys"]
    reused_key = existing_keys[0]

    async with session_maker() as db:
        user_id = await create_db_user(db)
        await _seed_existing_keys(db, user_id, existing_keys)

        fake = FakeAnthropicClient(create_texts=[fixture["recorded_reply"]])
        monkeypatch.setattr(memory_service, "get_anthropic_client", lambda: fake)

        await process_turn(
            user_id=user_id,
            conversation_id=uuid.uuid4(),
            user_message=fixture["user_message"],
            assistant_reply=fixture["assistant_reply"],
            db=db,
        )

        # The prompt sent to the mock listed the user's existing topic_keys for reuse.
        # Snapshot the request kwargs at call time (the mock copies messages to avoid
        # aliasing the orchestrator's mutable list), so this reads what was actually
        # sent on that call.
        assert len(fake.messages.create_calls) == 1
        prompt = fake.messages.create_calls[0]["messages"][0]["content"]
        assert reused_key in prompt

        # The reply reused the key, so its observations accumulate (seed + this turn).
        total_for_key = (
            await db.execute(
                select(func.count(MemoryObservation.id)).where(
                    MemoryObservation.user_id == user_id,
                    MemoryObservation.topic_key == reused_key,
                )
            )
        ).scalar_one()
        assert total_for_key == 2


# --------------------------------------------------------------------------- #
# Small talk: [] -> nothing stored, no consolidation attempted
# --------------------------------------------------------------------------- #
async def test_small_talk_stores_nothing_and_skips_consolidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _load(FIXTURES_DIR / "mem_03_small_talk.json")

    async with session_maker() as db:
        user_id = await create_db_user(db)

        fake = FakeAnthropicClient(create_texts=[fixture["recorded_reply"]])
        monkeypatch.setattr(memory_service, "get_anthropic_client", lambda: fake)

        await process_turn(
            user_id=user_id,
            conversation_id=uuid.uuid4(),
            user_message=fixture["user_message"],
            assistant_reply=fixture["assistant_reply"],
            db=db,
        )

        obs = (
            await db.execute(
                select(func.count(MemoryObservation.id)).where(MemoryObservation.user_id == user_id)
            )
        ).scalar_one()
        mems = (
            await db.execute(select(func.count(UserMemory.id)).where(UserMemory.user_id == user_id))
        ).scalar_one()

    assert obs == 0
    assert mems == 0
    # Exactly one model call (extraction). An empty extract returns before any
    # consolidation pass, so the fake is never asked for a second reply.
    assert len(fake.messages.create_calls) == 1


# --------------------------------------------------------------------------- #
# Provider failure: create() raises -> process_turn swallows, stores nothing
# --------------------------------------------------------------------------- #
async def test_provider_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising extraction client must not propagate or store anything (chat unaffected)."""

    class _RaisingMessages:
        def __init__(self) -> None:
            self.create_calls: list[dict] = []

        async def create(self, **kwargs):
            self.create_calls.append(dict(kwargs))
            raise RuntimeError("simulated Anthropic outage")

    class _RaisingClient:
        def __init__(self) -> None:
            self.messages = _RaisingMessages()

    fake = _RaisingClient()
    monkeypatch.setattr(memory_service, "get_anthropic_client", lambda: fake)

    async with session_maker() as db:
        user_id = await create_db_user(db)

        # Must NOT raise despite the provider blowing up.
        await process_turn(
            user_id=user_id,
            conversation_id=uuid.uuid4(),
            user_message="I train five days a week at a commercial gym.",
            assistant_reply="Great, five days gives us room for a full split.",
            db=db,
        )

        obs = (
            await db.execute(
                select(func.count(MemoryObservation.id)).where(MemoryObservation.user_id == user_id)
            )
        ).scalar_one()

    assert obs == 0
    assert len(fake.messages.create_calls) == 1
