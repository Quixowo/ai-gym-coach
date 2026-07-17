"""Injury red-flag recall eval — recorded fixtures, no live API.

Each ``claude_responses/rf_*.json`` fixture holds one classifier message, its
ground-truth label (acute injury vs routine), and the raw Haiku verdict recorded
once against the live model (see ``tests/fixtures/record_fixtures.py``). Here the
REAL ``classify_acute_injury`` runs against :class:`FakeAnthropicClient` replaying
that verdict, so CI verifies the strict one-word parsing handles the recorded
response, and — because the verdicts are frozen — the recall / false-positive rates
are deterministic, citable numbers.

Messages use varied phrasing on both sides (12 acute, 13 routine), including casual
pain mentions that are NOT acute ("no pain no gain", "legs are sore") to guard
against over-triggering — a classifier tuned only for recall drifts toward useless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.agent.classifier as classifier_module
from app.agent.classifier import classify_acute_injury
from tests.fixtures._replay import FakeAnthropicClient

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "claude_responses"
RF_FIXTURES = sorted(FIXTURES_DIR.glob("rf_*.json"))


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


ALL = [_load(p) for p in RF_FIXTURES]
ACUTE = [f for f in ALL if f["truth_acute"]]
ROUTINE = [f for f in ALL if not f["truth_acute"]]

# Baselines from the recorded run (see recording_metrics.json). Deterministic in CI:
# the recorded verdicts are frozen, so these fractions can't drift under a live model.
MIN_RECALL = 12  # recorded: 12/12 acute caught (100% recall)
MAX_FALSE_POSITIVES = 0  # recorded: 0/13 routine flagged (0% false-positive rate)


@pytest.mark.parametrize("fixture", ALL, ids=[f["id"] for f in ALL])
async def test_classifier_replays_recorded_verdict(
    monkeypatch: pytest.MonkeyPatch, fixture: dict
) -> None:
    """The real classifier maps the recorded verdict to the recorded boolean."""
    fake = FakeAnthropicClient(create_texts=[fixture["recorded_verdict"]])
    monkeypatch.setattr(classifier_module, "get_anthropic_client", lambda: fake)

    result = await classify_acute_injury(fixture["message"])
    assert result is fixture["predicted_acute"]
    # Exactly one classifier call per message.
    assert len(fake.messages.create_calls) == 1


def test_recall() -> None:
    """Recall = acute caught / acute total."""
    assert len(ACUTE) == 12
    caught = sum(1 for f in ACUTE if f["predicted_acute"])
    assert caught >= MIN_RECALL, f"recall {caught}/{len(ACUTE)} below floor"


def test_false_positive_rate() -> None:
    """False-positive rate = routine wrongly flagged / routine total."""
    assert len(ROUTINE) == 13
    flagged = sum(1 for f in ROUTINE if f["predicted_acute"])
    assert flagged <= MAX_FALSE_POSITIVES, f"{flagged}/{len(ROUTINE)} routine msgs flagged"


def test_message_set_shape() -> None:
    """~25 messages, both sides well represented."""
    assert len(ALL) == 25
    assert len(ACUTE) >= 10 and len(ROUTINE) >= 10
