"""Progression-math tests — pure computation, no LLM, no HTTP.

Drives ``progression_service.analyze`` directly against a NullPool DB session,
building multi-session histories with :func:`add_session_with_sets`. Sessions are
inserted oldest-first here for readability, but ``analyze`` orders by session date,
so ``days_ago`` controls the true oldest->newest sequence.
"""

from __future__ import annotations

import uuid

import pytest

from app.services import progression_service
from app.services.errors import NotFoundError
from tests.conftest import test_session_maker as session_maker
from tests.helpers import add_session_with_sets, create_db_user, first_exercise_id


# --------------------------------------------------------------------------- #
# Epley formula (unit-level, no DB)
# --------------------------------------------------------------------------- #
def test_epley_formula_values() -> None:
    # weight * (1 + reps/30)
    assert progression_service.estimated_1rm(100, 0) == 100.0
    assert progression_service.estimated_1rm(100, 30) == pytest.approx(200.0)
    assert progression_service.estimated_1rm(200, 5) == pytest.approx(200 * (1 + 5 / 30))
    assert progression_service.estimated_1rm(185, 5) == pytest.approx(215.833, abs=1e-3)


# --------------------------------------------------------------------------- #
# Unknown / no-data
# --------------------------------------------------------------------------- #
async def test_unknown_exercise_raises_not_found() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        with pytest.raises(NotFoundError):
            await progression_service.analyze(db, user_id, uuid.uuid4())


async def test_known_exercise_no_history_returns_empty_shape() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        result = await progression_service.analyze(db, user_id, ex)
    assert result["sessions_analyzed"] == 0
    assert result["estimated_1rm_series"] == []
    assert result["trend"] == "flat"
    assert result["rir_trend"] == "insufficient_data"
    assert result["plateaued"] is False
    assert result["plateau_session_count"] == 0


# --------------------------------------------------------------------------- #
# 1RM trend direction + the 2% boundary
# --------------------------------------------------------------------------- #
async def test_trend_increasing() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        # 4 sessions, clearly rising 1RM: first-half mean well below second-half.
        for i, w in enumerate([100, 105, 120, 130]):
            await add_session_with_sets(db, user_id, ex, [(w, 5, 2.0)], days_ago=40 - i * 10)
        result = await progression_service.analyze(db, user_id, ex)
    assert result["trend"] == "increasing"
    assert result["sessions_analyzed"] == 4
    assert len(result["estimated_1rm_series"]) == 4
    # Oldest->newest ordering.
    assert result["estimated_1rm_series"] == sorted(result["estimated_1rm_series"])


async def test_trend_decreasing() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        for i, w in enumerate([130, 120, 105, 100]):
            await add_session_with_sets(db, user_id, ex, [(w, 5, 2.0)], days_ago=40 - i * 10)
        result = await progression_service.analyze(db, user_id, ex)
    assert result["trend"] == "decreasing"


async def test_trend_flat_just_under_2pct() -> None:
    # first half mean 100, second half mean ~101.9 (+1.9%) -> under 2% -> flat.
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        weights = [100.0, 100.0, 101.9, 101.9]  # 0 reps -> e1rm == weight
        for i, w in enumerate(weights):
            await add_session_with_sets(db, user_id, ex, [(w, 0, 2.0)], days_ago=40 - i * 10)
        result = await progression_service.analyze(db, user_id, ex)
    assert result["trend"] == "flat"


async def test_trend_increasing_just_over_2pct() -> None:
    # first half 100, second half 102.1 (+2.1%) -> over 2% -> increasing.
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        weights = [100.0, 100.0, 102.1, 102.1]
        for i, w in enumerate(weights):
            await add_session_with_sets(db, user_id, ex, [(w, 0, 2.0)], days_ago=40 - i * 10)
        result = await progression_service.analyze(db, user_id, ex)
    assert result["trend"] == "increasing"


# --------------------------------------------------------------------------- #
# RIR trend — comparable-load filter + insufficient_data
# --------------------------------------------------------------------------- #
async def test_rir_trend_improving_at_constant_load() -> None:
    # Same weight throughout (comparable), RIR falling 3->2->1->0 = improving.
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        for i, rir in enumerate([3.0, 3.0, 1.0, 0.0]):
            await add_session_with_sets(db, user_id, ex, [(100.0, 5, rir)], days_ago=40 - i * 10)
        result = await progression_service.analyze(db, user_id, ex)
    assert result["rir_trend"] == "improving"


async def test_rir_trend_insufficient_data_when_loads_differ() -> None:
    # Every session a different weight -> most-common weight has only 1 comparable
    # session within ±5%; well under the 3-session minimum -> insufficient_data.
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        for i, w in enumerate([100.0, 130.0, 160.0, 190.0]):
            await add_session_with_sets(db, user_id, ex, [(w, 5, 2.0)], days_ago=40 - i * 10)
        result = await progression_service.analyze(db, user_id, ex)
    assert result["rir_trend"] == "insufficient_data"


async def test_rir_comparable_load_filter_excludes_out_of_band_session() -> None:
    # 3 sessions at 100 (comparable) with falling RIR + 1 outlier at 200. The outlier
    # is excluded from the RIR trend; the 3 comparable sessions drive "improving".
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        await add_session_with_sets(db, user_id, ex, [(100.0, 5, 3.0)], days_ago=40)
        await add_session_with_sets(db, user_id, ex, [(100.0, 5, 2.0)], days_ago=30)
        await add_session_with_sets(db, user_id, ex, [(100.0, 5, 1.0)], days_ago=20)
        await add_session_with_sets(db, user_id, ex, [(200.0, 5, 4.0)], days_ago=10)
        result = await progression_service.analyze(db, user_id, ex)
    assert result["rir_trend"] == "improving"


# --------------------------------------------------------------------------- #
# Plateau detection
# --------------------------------------------------------------------------- #
async def test_plateau_detected_last_three_identical() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        await add_session_with_sets(db, user_id, ex, [(100.0, 5, 2.0)], days_ago=40)
        await add_session_with_sets(db, user_id, ex, [(190.0, 5, 1.0)], days_ago=30)
        await add_session_with_sets(db, user_id, ex, [(190.0, 5, 1.0)], days_ago=20)
        await add_session_with_sets(db, user_id, ex, [(190.0, 5, 1.0)], days_ago=10)
        result = await progression_service.analyze(db, user_id, ex)
    assert result["plateaued"] is True
    assert result["plateau_session_count"] == 3


async def test_no_plateau_when_last_two_identical_only() -> None:
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        await add_session_with_sets(db, user_id, ex, [(180.0, 5, 2.0)], days_ago=30)
        await add_session_with_sets(db, user_id, ex, [(190.0, 5, 1.0)], days_ago=20)
        await add_session_with_sets(db, user_id, ex, [(190.0, 5, 1.0)], days_ago=10)
        result = await progression_service.analyze(db, user_id, ex)
    assert result["plateaued"] is False
    assert result["plateau_session_count"] == 2


async def test_spec_worked_example_flat_and_plateaued() -> None:
    """Worked example: series [185.2, 187.5, 190.0, 190.0, 190.0] -> flat + plateaued.

    Built by choosing weights (reps=0 so e1rm == weight) that reproduce the exact
    series, with the last 3 sessions identical in weight*reps*rir. First-half mean
    (185.2, 187.5) = 186.35; second-half (190, 190) = 190 -> +1.96% < 2% -> flat.
    Last 3 sessions identical -> plateaued, count 3. This confirms the worked example
    is internally consistent with the stated rules.
    """
    async with session_maker() as db:
        user_id = await create_db_user(db)
        ex = await first_exercise_id(db)
        series = [185.2, 187.5, 190.0, 190.0, 190.0]
        for i, w in enumerate(series):
            await add_session_with_sets(db, user_id, ex, [(w, 0, 1.0)], days_ago=50 - i * 10)
        result = await progression_service.analyze(db, user_id, ex)
    assert result["estimated_1rm_series"] == series
    assert result["sessions_analyzed"] == 5
    assert result["trend"] == "flat"
    assert result["plateaued"] is True
    assert result["plateau_session_count"] == 3
