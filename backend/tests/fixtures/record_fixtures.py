"""One-time live recorder for the three eval suites.

Run from ``backend/``::

    python -m tests.fixtures.record_fixtures                     # everything
    python -m tests.fixtures.record_fixtures --only groundedness
    python -m tests.fixtures.record_fixtures --mock              # plumbing, ZERO spend
    python -m tests.fixtures.record_fixtures --force             # re-record existing

This is the ONLY place a live model is called (CLAUDE.md rule 10). It captures the
model's real decisions once and commits them under ``claude_responses/`` so the
suites (and CI) replay recordings, never the live API. Each fixture JSON is written
IMMEDIATELY as it is captured and existing fixtures are skipped unless ``--force``,
so a crash mid-pass never loses paid-for recordings and a resume re-spends nothing.

``--mock`` swaps every live client for a fake (hand-written tool-use scripts,
canned synthesis/verdict/classifier text, synthetic embeddings) so the entire
capture → serialize → write pipeline can be exercised with no API spend before the
real budget is committed. The live pass then overwrites the mock fixtures.

Windows console note (LESSONS.md): stdout is reconfigured to UTF-8 and every
paid-for capture is written to its UTF-8 fixture file BEFORE anything is printed,
so a cp1252 console crash on model output containing emoji/dashes can't lose it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

from sqlalchemy import delete, select

import app.agent.orchestrator as orchestrator_module
import app.services.knowledge_service as knowledge_service_module
from app.agent.classifier import _ACUTE_LABEL, _CLASSIFIER_PROMPT
from app.agent.orchestrator import run_agent_turn
from app.core.config import settings
from app.db.session import async_session_maker
from app.llm.client import get_anthropic_client
from app.llm.voyage import get_voyage_client
from app.models.knowledge_chunk import KnowledgeChunk
from app.models.program import Program, ProgramExercise
from app.models.set_entry import SetEntry
from app.models.user import User
from app.models.workout_session import WorkoutSession
from app.services.knowledge_service import (
    _GROUNDEDNESS_PROMPT,
    _UNGROUNDED_ANSWER,
    _dedup_sources,
    _numbered_reference,
    build_synthesis_prompt,
)
from tests.fixtures import _scenario
from tests.fixtures._replay import FakeAnthropicClient, FakeVoyageClient, query_embedding
from tests.helpers import create_db_user

# Reconfigure stdout before any printing — model output may contain characters the
# Windows cp1252 console can't encode (LESSONS.md). Fixtures are also written to
# UTF-8 files before printing, so a console crash can't lose paid-for output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001 — best-effort; non-fatal if unsupported
    pass

FIXTURES_DIR = Path(__file__).resolve().parent / "claude_responses"
METRICS_PATH = Path(__file__).resolve().parent / "recording_metrics.json"

VOYAGE_MIN_INTERVAL_S = 21.0  # no-card Voyage tier: 3 RPM (LESSONS.md)
_last_embed_time = 0.0


# --------------------------------------------------------------------------- #
# Question banks
# --------------------------------------------------------------------------- #
# 15 answerable (12 training + 3 injury_prevention) + 4 traps (see metrics).
GROUNDEDNESS_QUESTIONS: list[dict] = [
    {
        "id": "grd_01_progressive_overload",
        "category": "training",
        "query": "What is progressive overload and how do I apply it?",
    },
    {
        "id": "grd_02_load_increment",
        "category": "training",
        "query": "How much weight should I add to the bar each session when progressing linearly?",
    },
    {
        "id": "grd_03_hypertrophy_volume",
        "category": "training",
        "query": "How many sets per muscle group per week should I do to build muscle?",
    },
    {
        "id": "grd_04_hypertrophy_reps",
        "category": "training",
        "query": "What rep ranges are effective for hypertrophy?",
    },
    {
        "id": "grd_05_what_is_rir",
        "category": "training",
        "query": "What does RIR mean in training?",
    },
    {
        "id": "grd_06_rir_target",
        "category": "training",
        "query": "What RIR should most of my hypertrophy sets be at?",
    },
    {
        "id": "grd_07_deload",
        "category": "training",
        "query": "When should I take a deload and how do I structure it?",
    },
    {
        "id": "grd_08_frequency",
        "category": "training",
        "query": "How many times per week should I train each muscle group?",
    },
    {
        "id": "grd_09_strength_intensity",
        "category": "training",
        "query": "What intensity and rep range should I use for building maximal strength?",
    },
    {
        "id": "grd_10_warmup",
        "category": "training",
        "query": "How should I warm up before heavy squats?",
    },
    {
        "id": "grd_11_rest_periods",
        "category": "training",
        "query": "How long should I rest between sets for heavy strength work?",
    },
    {
        "id": "grd_12_squat_kneecave",
        "category": "training",
        "query": "Why do my knees cave in when I squat and how do I fix it?",
    },
    {
        "id": "grd_13_acute_vs_chronic",
        "category": "injury_prevention",
        "query": "What's the difference between acute and chronic pain?",
    },
    {
        "id": "grd_14_acute_response",
        "category": "injury_prevention",
        "query": "What should I do for a minor acute muscle strain?",
    },
    {
        "id": "grd_15_recognize_acute",
        "category": "injury_prevention",
        "query": "How can I tell if pain during a lift is acute rather than normal soreness?",
    },
    # Traps — nothing in the corpus answers these; the pipeline should refuse.
    {
        "id": "grd_16_supplement_brand",
        "category": "trap",
        "trap_flavor": "nutrition",
        "query": "What's the best pre-workout supplement brand to buy?",
    },
    {
        "id": "grd_17_running_time",
        "category": "trap",
        "query": "How do I improve my 5k running time?",
    },
    {
        "id": "grd_18_deadlift_record",
        "category": "trap",
        "query": "What's the current world record for the deadlift?",
    },
    {
        "id": "grd_19_barbell_cost",
        "category": "trap",
        "query": "How much does a standard Olympic barbell cost?",
    },
]

# ~25 classifier messages, varied phrasing. truth = ground-truth acute?
RED_FLAG_MESSAGES: list[dict] = [
    # --- Acute (potential injury needing evaluation) ---
    {
        "id": "rf_01",
        "acute": True,
        "message": "I felt a sharp pop in my knee mid-squat and now it won't bend.",
    },
    {
        "id": "rf_02",
        "acute": True,
        "message": "Heard a loud snap in my shoulder during bench and now I can't raise my arm.",
    },
    {
        "id": "rf_03",
        "acute": True,
        "message": "Sudden stabbing pain in my lower back on a deadlift, couldn't stand up straight after.",  # noqa: E501
    },
    {
        "id": "rf_04",
        "acute": True,
        "message": "My elbow gave out mid-rep, it's swelling and I can't straighten it.",
    },
    {
        "id": "rf_05",
        "acute": True,
        "message": "Sharp pain shot down my leg when I picked up the bar and now my foot is numb.",
    },
    {
        "id": "rf_06",
        "acute": True,
        "message": "I think I tore my hamstring, it was a sudden sharp pain and there's a bruise forming.",  # noqa: E501
    },
    {
        "id": "rf_07",
        "acute": True,
        "message": "My wrist bent back awkwardly under the bar and it's swollen and painful to move.",  # noqa: E501
    },
    {
        "id": "rf_08",
        "acute": True,
        "message": "Rolled my ankle stepping off the platform, it's swelling fast and I can't put weight on it.",  # noqa: E501
    },
    {
        "id": "rf_09",
        "acute": True,
        "message": "Felt a tearing sensation in my chest during a heavy press and it hurts to breathe.",  # noqa: E501
    },
    {
        "id": "rf_10",
        "acute": True,
        "message": "My shoulder popped out of place and I had to push it back in.",
    },
    {
        "id": "rf_11",
        "acute": True,
        "message": "Immediate sharp pain in my groin on a wide-stance squat, now I can barely walk.",  # noqa: E501
    },
    {
        "id": "rf_12",
        "acute": True,
        "message": "Something sharp went off in my neck and now I can't turn my head.",
    },
    # --- Routine (normal soreness / fatigue / general questions) ---
    {"id": "rf_13", "acute": False, "message": "My legs are super sore from squats two days ago."},
    {
        "id": "rf_14",
        "acute": False,
        "message": "No pain no gain, right? Just pushing through the burn on my last set.",
    },
    {
        "id": "rf_15",
        "acute": False,
        "message": "I'm feeling pretty fatigued this week, might be time for a deload.",
    },
    {
        "id": "rf_16",
        "acute": False,
        "message": "My muscles are tight after yesterday's session, any good stretches?",
    },
    {
        "id": "rf_17",
        "acute": False,
        "message": "Bit of general soreness in my chest after benching, normal DOMS I think.",
    },
    {
        "id": "rf_18",
        "acute": False,
        "message": "Should I train through mild muscle soreness or take a rest day?",
    },
    {
        "id": "rf_19",
        "acute": False,
        "message": "My grip gives out before my back does on heavy deadlifts.",
    },
    {
        "id": "rf_20",
        "acute": False,
        "message": "Feeling a little stiff this morning but nothing out of the ordinary.",
    },
    {
        "id": "rf_21",
        "acute": False,
        "message": "How sore is too sore to train the same muscle again?",
    },
    {
        "id": "rf_22",
        "acute": False,
        "message": "My shoulders feel a bit worked after all the pressing volume today.",
    },
    {
        "id": "rf_23",
        "acute": False,
        "message": "I get a solid pump and burn in my biceps by the last set.",
    },
    {
        "id": "rf_24",
        "acute": False,
        "message": "Just the usual next-day soreness in my glutes, feels normal to me.",
    },
    {"id": "rf_25", "acute": False, "message": "What's a good rep range for hypertrophy?"},
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _write_fixture(name: str, payload: dict) -> Path:
    """Write ``claude_responses/{name}.json`` (UTF-8) BEFORE any printing."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURES_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _exists(name: str) -> bool:
    return (FIXTURES_DIR / f"{name}.json").exists()


async def _pace_voyage() -> None:
    """Sleep so consecutive live Voyage embeds are >= 21s apart (no-card 3 RPM)."""
    global _last_embed_time
    now = time.monotonic()
    wait = VOYAGE_MIN_INTERVAL_S - (now - _last_embed_time)
    if wait > 0:
        await asyncio.sleep(wait)
    _last_embed_time = time.monotonic()


def _serialize_blocks(content: list) -> list[dict]:
    """Serialize SDK/fake content blocks to the fixture shape ``_replay`` rebuilds."""
    out: list[dict] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            raw = block.input
            out.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(raw) if raw else {},
                }
            )
        elif btype == "text":
            out.append({"type": "text", "text": block.text})
        # Any other block type (e.g. thinking) is irrelevant to the loop — drop it.
    return out


# --------------------------------------------------------------------------- #
# Recording-time Anthropic proxy: capture streamed turns + create() outputs
# --------------------------------------------------------------------------- #
class _RecordingStreamCM:
    def __init__(self, delegate_cm: object, capture: list) -> None:
        self._delegate_cm = delegate_cm
        self._capture = capture
        self._stream: object | None = None

    async def __aenter__(self) -> _RecordingStreamCM:
        self._stream = await self._delegate_cm.__aenter__()
        return self

    async def __aexit__(self, *exc) -> object:
        return await self._delegate_cm.__aexit__(*exc)

    def __aiter__(self):
        return self._stream.__aiter__()

    async def get_final_message(self):
        final = await self._stream.get_final_message()
        self._capture.append(_serialize_blocks(final.content))
        return final


class _RecordingMessages:
    def __init__(self, delegate: object, stream_capture: list) -> None:
        self._delegate = delegate
        self._stream_capture = stream_capture

    def stream(self, **kwargs):
        return _RecordingStreamCM(self._delegate.messages.stream(**kwargs), self._stream_capture)

    async def create(self, **kwargs):  # pragma: no cover — orchestrator uses stream only
        return await self._delegate.messages.create(**kwargs)


class _RecordingAnthropicClient:
    def __init__(self, delegate: object, stream_capture: list) -> None:
        self.messages = _RecordingMessages(delegate, stream_capture)


# Hermetic RAG stand-ins so a recorded/replayed agent turn that calls
# search_knowledge_base never touches the live Voyage/Anthropic APIs (keeps the
# tool-correctness budget to pure Sonnet turns; retrieval stays real/read-only).
class _ConstResponse:
    def __init__(self, text: str) -> None:
        self.content = [type("_B", (), {"text": text})()]


class _ConstMessages:
    async def create(self, **kwargs):
        return _ConstResponse("GROUNDED")


class _ConstAnthropic:
    def __init__(self) -> None:
        self.messages = _ConstMessages()


# --------------------------------------------------------------------------- #
# Suite 1: tool correctness
# --------------------------------------------------------------------------- #
async def _cleanup_user(db, user_id: uuid.UUID) -> None:
    """Delete every row owned by the throwaway recording user (FK-safe order)."""
    await db.execute(delete(SetEntry).where(SetEntry.user_id == user_id))
    await db.execute(delete(WorkoutSession).where(WorkoutSession.user_id == user_id))
    program_ids = (
        (await db.execute(select(Program.id).where(Program.user_id == user_id))).scalars().all()
    )
    if program_ids:
        await db.execute(delete(ProgramExercise).where(ProgramExercise.program_id.in_(program_ids)))
        await db.execute(delete(Program).where(Program.user_id == user_id))
    await db.execute(delete(User).where(User.id == user_id))
    await db.commit()


def _resolve_mock_script(script: list, names: dict[str, str]) -> list:
    """Substitute ``<Exercise Name>`` placeholders with that exercise's real id."""
    text = json.dumps(script)
    for exercise_name, real in names.items():
        text = text.replace(f"<{exercise_name}>", real)
    return json.loads(text)


async def record_tool_correctness(mock: bool, force: bool, limit: int | None = None) -> None:
    scenarios = _scenario.SCENARIOS if limit is None else _scenario.SCENARIOS[:limit]
    for scenario in scenarios:
        name = f"tc_{scenario['id']}"
        if _exists(name) and not force:
            print(f"[tool_correctness] skip existing {name}")
            continue

        async with async_session_maker() as db:
            user_id = await create_db_user(db)
            await _scenario.apply_setup(db, user_id, scenario["setup"])
            names = await _scenario.name_to_id_map(db)
            names_by_id = {v: k for k, v in names.items()}

        try:
            if mock:
                script = _resolve_mock_script(scenario["mock_script"], names)
                delegate: object = FakeAnthropicClient(stream_iterations=script)
            else:
                delegate = get_anthropic_client()

            stream_capture: list = []
            rec_client = _RecordingAnthropicClient(delegate, stream_capture)

            orig_orch = orchestrator_module.get_anthropic_client
            orig_ks_voyage = knowledge_service_module.get_voyage_client
            orig_ks_anth = knowledge_service_module.get_anthropic_client
            orchestrator_module.get_anthropic_client = lambda: rec_client
            knowledge_service_module.get_voyage_client = lambda: FakeVoyageClient()
            knowledge_service_module.get_anthropic_client = lambda: _ConstAnthropic()
            try:
                async with async_session_maker() as db:
                    events = [e async for e in run_agent_turn(scenario["message"], [], user_id, db)]
            finally:
                orchestrator_module.get_anthropic_client = orig_orch
                knowledge_service_module.get_voyage_client = orig_ks_voyage
                knowledge_service_module.get_anthropic_client = orig_ks_anth

            # Build id_map: recorded exercise UUID -> exercise NAME (rewrite target).
            id_map: dict[str, str] = {}
            recorded_tools: list[str] = []
            for iteration in stream_capture:
                for block in iteration:
                    if block["type"] != "tool_use":
                        continue
                    recorded_tools.append(block["name"])
                    for value in _iter_strings(block.get("input", {})):
                        if value in names_by_id:
                            id_map[value] = names_by_id[value]

            must_call = scenario["must_call"]
            must_not = scenario["must_not_call"]
            matched = all(t in recorded_tools for t in must_call) and not any(
                t in recorded_tools for t in must_not
            )

            payload = {
                "id": scenario["id"],
                "user_message": scenario["message"],
                "setup": scenario["setup"],
                "must_call": must_call,
                "must_not_call": must_not,
                "checks": scenario["checks"],
                "id_map": id_map,
                "stream_iterations": stream_capture,
                "recorded_tools": recorded_tools,
                "matched_intent": matched,
            }
            _write_fixture(name, payload)
            _ = events  # events are for live inspection only
            flag = "OK " if matched else "!! "
            print(f"[tool_correctness] {flag}{name}: tools={recorded_tools} matched={matched}")
        finally:
            async with async_session_maker() as db:
                await _cleanup_user(db, user_id)


def _iter_strings(obj: object):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


# --------------------------------------------------------------------------- #
# Suite 2: groundedness
# --------------------------------------------------------------------------- #
_REFUSAL_MARKERS = (
    "does not contain",
    "doesn't contain",
    "does not provide",
    "doesn't provide",
    "not contain enough",
    "not enough information",
    "no information",
    "does not mention",
    "doesn't mention",
    "isn't covered",
    "is not covered",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "does not address",
    "doesn't address",
    "no relevant information",
    "not covered in",
    "not discuss",
    "does not discuss",
    "doesn't discuss",
    "not include",
)


def _looks_refused(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


def _parse_attribution(text: str) -> list[int]:
    nums: list[int] = []
    token = ""
    for ch in text:
        if ch.isdigit():
            token += ch
        else:
            if token:
                nums.append(int(token))
                token = ""
    if token:
        nums.append(int(token))
    return sorted({n for n in nums if 1 <= n <= 5})


async def record_groundedness(mock: bool, force: bool, limit: int | None = None) -> None:
    questions = GROUNDEDNESS_QUESTIONS if limit is None else GROUNDEDNESS_QUESTIONS[:limit]
    for question in questions:
        name = question["id"]
        if _exists(name) and not force:
            print(f"[groundedness] skip existing {name}")
            continue

        query = question["query"]
        is_trap = question["category"] == "trap"

        # 1) Retrieve top-5 (real embeddings on the live corpus) or synthetic in mock.
        if mock:
            qvec = query_embedding()
        else:
            await _pace_voyage()
            voyage = get_voyage_client()
            embed_result = await voyage.embed(
                [query], model=settings.EMBED_MODEL_ID, input_type="query"
            )
            qvec = embed_result.embeddings[0]

        async with async_session_maker() as db:
            chunks = (
                (
                    await db.execute(
                        select(KnowledgeChunk)
                        .order_by(KnowledgeChunk.embedding.cosine_distance(qvec))
                        .limit(5)
                    )
                )
                .scalars()
                .all()
            )

        chunk_dicts = [
            {
                "document_title": c.document_title,
                "category": c.category,
                "chunk_text": c.chunk_text,
                "chunk_index": c.chunk_index,
                "source_citation": c.source_citation,
            }
            for c in chunks
        ]

        # 2) Synthesis + 3) groundedness verdict — reproduce search_knowledge_base's
        #    two create() calls exactly (same prompt builders) so the recorded texts
        #    drive the real service in replay.
        if mock:
            synthesis_text = f"Mock synthesis for: {query}"
            verdict_text = "GROUNDED"
            attribution = None if is_trap else [1, 2]
        else:
            client = get_anthropic_client()
            synth = await client.messages.create(
                model=settings.HAIKU_MODEL_ID,
                max_tokens=400,
                messages=[{"role": "user", "content": build_synthesis_prompt(query, chunks)}],
            )
            synthesis_text = synth.content[0].text
            gr = await client.messages.create(
                model=settings.HAIKU_MODEL_ID,
                max_tokens=10,
                messages=[
                    {
                        "role": "user",
                        "content": _GROUNDEDNESS_PROMPT.format(
                            reference=_numbered_reference(chunks), answer=synthesis_text
                        ),
                    }
                ],
            )
            verdict_text = gr.content[0].text

            # c) Citation density (answerable only) — one extra recording-time Haiku call.
            attribution = None
            if not is_trap:
                attr_prompt = (
                    "Below are numbered reference chunks and an ANSWER.\n\n"
                    f"Reference material:\n{_numbered_reference(chunks)}\n\n"
                    f"ANSWER:\n{synthesis_text}\n\n"
                    "Which numbered reference chunks does the ANSWER draw information from? "
                    "Respond with only the numbers, comma-separated."
                )
                attr = await client.messages.create(
                    model=settings.HAIKU_MODEL_ID,
                    max_tokens=30,
                    messages=[{"role": "user", "content": attr_prompt}],
                )
                attribution = _parse_attribution(attr.content[0].text)

        passed = verdict_text.strip() == "GROUNDED"
        sources = _dedup_sources(chunks)
        answer = synthesis_text if passed else _UNGROUNDED_ANSWER
        expected = {"groundedness_passed": passed, "answer": answer, "sources": sources}

        trap_refused = None
        if is_trap:
            trap_refused = (not passed) or _looks_refused(synthesis_text)

        payload = {
            "id": name,
            "query": query,
            "category": question["category"],
            "trap": is_trap,
            "trap_flavor": question.get("trap_flavor"),
            "chunks": chunk_dicts,
            "synthesis_text": synthesis_text,
            "groundedness_verdict": verdict_text,
            "expected": expected,
            "attribution": attribution,
            "trap_refused": trap_refused,
        }
        _write_fixture(name, payload)
        print(
            f"[groundedness] OK {name}: passed={passed} refused={trap_refused} attrib={attribution}"
        )


# --------------------------------------------------------------------------- #
# Suite 3: red-flag classifier
# --------------------------------------------------------------------------- #
async def record_red_flag(mock: bool, force: bool, limit: int | None = None) -> None:
    messages = RED_FLAG_MESSAGES if limit is None else RED_FLAG_MESSAGES[:limit]
    for item in messages:
        name = item["id"]
        if _exists(name) and not force:
            print(f"[red_flag] skip existing {name}")
            continue

        message = item["message"]
        if mock:
            verdict_text = _ACUTE_LABEL if item["acute"] else "ROUTINE"
            latency_ms = 0.0
        else:
            client = get_anthropic_client()
            start = time.perf_counter()
            resp = await client.messages.create(
                model=settings.HAIKU_MODEL_ID,
                max_tokens=10,
                messages=[{"role": "user", "content": _CLASSIFIER_PROMPT.format(message=message)}],
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            verdict_text = resp.content[0].text

        predicted_acute = verdict_text.strip() == _ACUTE_LABEL
        payload = {
            "id": name,
            "message": message,
            "truth_acute": item["acute"],
            "recorded_verdict": verdict_text,
            "predicted_acute": predicted_acute,
            "latency_ms": round(latency_ms, 2),
        }
        _write_fixture(name, payload)
        correct = "OK " if predicted_acute == item["acute"] else "XX "
        print(
            f"[red_flag] {correct}{name}: truth={item['acute']} "
            f"pred={predicted_acute} {latency_ms:.0f}ms"
        )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _load_all(prefix: str) -> list[dict]:
    out = []
    for path in sorted(FIXTURES_DIR.glob(f"{prefix}*.json")):
        with path.open(encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def compute_metrics() -> dict:
    grd = _load_all("grd_")
    tc = _load_all("tc_")
    rf = _load_all("rf_")

    answerable = [q for q in grd if not q["trap"]]
    traps = [q for q in grd if q["trap"]]

    # a) groundedness pass rate (answerable), overall + per category
    def _rate(items: list[dict]) -> dict:
        passed = sum(1 for q in items if q["expected"]["groundedness_passed"])
        return {
            "passed": passed,
            "total": len(items),
            "fraction": f"{passed}/{len(items)}" if items else "0/0",
        }

    by_cat: dict[str, list[dict]] = {}
    for q in answerable:
        by_cat.setdefault(q["category"], []).append(q)

    groundedness = {
        "overall": _rate(answerable),
        "by_category": {cat: _rate(items) for cat, items in sorted(by_cat.items())},
        "per_question": [
            {
                "id": q["id"],
                "category": q["category"],
                "passed": q["expected"]["groundedness_passed"],
            }
            for q in answerable
        ],
    }

    # b) traps refused vs fabricated
    refused = sum(1 for q in traps if q["trap_refused"])
    traps_metric = {
        "refused": refused,
        "total": len(traps),
        "fraction": f"{refused}/{len(traps)}",
        "per_trap": [
            {
                "id": q["id"],
                "flavor": q.get("trap_flavor"),
                "groundedness_passed": q["expected"]["groundedness_passed"],
                "refused": q["trap_refused"],
                "answer": q["synthesis_text"],
            }
            for q in traps
        ],
    }

    # c) citation density (answerable only): mean attribution count out of 5
    counts = [len(q["attribution"]) for q in answerable if q["attribution"] is not None]
    citation_density = {
        "mean_out_of_5": round(statistics.mean(counts), 2) if counts else None,
        "n": len(counts),
        "per_question": [
            {"id": q["id"], "attribution": q["attribution"]}
            for q in answerable
            if q["attribution"] is not None
        ],
    }

    # d) classifier latency
    latencies = [r["latency_ms"] for r in rf if r.get("latency_ms")]
    latency_metric = {
        "n": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "median_ms": round(statistics.median(latencies), 2) if latencies else None,
        "p95_ms": round(_percentile(latencies, 95), 2) if latencies else None,
        "raw_ms": latencies,
    }

    # e) tool-call correctness x/6 + load-jump-cap cases
    matched = sum(1 for t in tc if t["matched_intent"])
    cap_cases = [t for t in tc if t["checks"].get("cap_rejected")]
    cap_fired = []
    for t in cap_cases:
        fired = any(
            b["type"] == "tool_use" and b["name"] == "update_program"
            for it in t["stream_iterations"]
            for b in it
        )
        cap_fired.append({"id": t["id"], "update_program_called": fired})
    tool_metric = {
        "matched": matched,
        "total": len(tc),
        "fraction": f"{matched}/{len(tc)}",
        "load_jump_cap_cases": len(cap_cases),
        "cap_cases_detail": cap_fired,
        "per_case": [
            {
                "id": t["id"],
                "must_call": t["must_call"],
                "recorded_tools": t["recorded_tools"],
                "matched": t["matched_intent"],
            }
            for t in tc
        ],
    }

    # f) red-flag recall & FP
    acute = [r for r in rf if r["truth_acute"]]
    routine = [r for r in rf if not r["truth_acute"]]
    caught = sum(1 for r in acute if r["predicted_acute"])
    fp = sum(1 for r in routine if r["predicted_acute"])
    red_flag_metric = {
        "recall": {
            "caught": caught,
            "total": len(acute),
            "fraction": f"{caught}/{len(acute)}" if acute else "0/0",
            "rate": round(caught / len(acute), 4) if acute else None,
        },
        "false_positive": {
            "flagged": fp,
            "total": len(routine),
            "fraction": f"{fp}/{len(routine)}" if routine else "0/0",
            "rate": round(fp / len(routine), 4) if routine else None,
        },
        "per_message": [
            {
                "id": r["id"],
                "truth_acute": r["truth_acute"],
                "predicted_acute": r["predicted_acute"],
                "verdict": r["recorded_verdict"].strip(),
            }
            for r in rf
        ],
    }

    return {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": {
            "sonnet": settings.SONNET_MODEL_ID,
            "haiku": settings.HAIKU_MODEL_ID,
            "embed": settings.EMBED_MODEL_ID,
        },
        "a_groundedness": groundedness,
        "b_traps": traps_metric,
        "c_citation_density": citation_density,
        "d_classifier_latency": latency_metric,
        "e_tool_correctness": tool_metric,
        "f_red_flag": red_flag_metric,
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
async def _amain(
    only: str | None, mock: bool, force: bool, metrics_only: bool, limit: int | None
) -> None:
    if metrics_only:
        metrics = compute_metrics()
        with METRICS_PATH.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Wrote {METRICS_PATH}")
        return

    if only in (None, "groundedness"):
        await record_groundedness(mock, force, limit)
    if only in (None, "tool_correctness"):
        await record_tool_correctness(mock, force, limit)
    if only in (None, "red_flag"):
        await record_red_flag(mock, force, limit)

    metrics = compute_metrics()
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Wrote {METRICS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record eval fixtures.")
    parser.add_argument(
        "--only",
        choices=["tool_correctness", "groundedness", "red_flag"],
        default=None,
    )
    parser.add_argument("--mock", action="store_true", help="fakes only, zero API spend")
    parser.add_argument("--force", action="store_true", help="re-record existing fixtures")
    parser.add_argument("--metrics-only", action="store_true", help="recompute metrics from disk")
    parser.add_argument("--limit", type=int, default=None, help="cap items per suite (smoke)")
    args = parser.parse_args()
    asyncio.run(_amain(args.only, args.mock, args.force, args.metrics_only, args.limit))


if __name__ == "__main__":
    main()
