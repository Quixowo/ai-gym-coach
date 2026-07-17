"""Workout session / set-entry endpoint tests.

Covers the validation matrix, the session find-or-create + set_number
computation, the at-most-one-open-session 409, finish->start, history filtering
and ordering, and the auth guard. The catalog is seeded by conftest.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.rate_limit import limiter
from tests.helpers import get_exercise_ids, register_user


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    limiter.reset()


def _log_set(client: TestClient, exercise_id: str, weight=100.0, reps=5, rir=2.0):
    return client.post(
        "/workouts/sets",
        json={"exercise_id": exercise_id, "weight": weight, "reps": reps, "rir": rir},
    )


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def test_endpoints_require_auth(client: TestClient) -> None:
    client.cookies.clear()
    assert client.get("/workouts/sessions").status_code == 401
    assert client.post("/workouts/sessions", json={}).status_code == 401
    assert client.get("/workouts/history").status_code == 401
    assert (
        client.post(
            "/workouts/sets",
            json={"exercise_id": "00000000-0000-0000-0000-000000000000", "weight": 1, "reps": 1},
        ).status_code
        == 401
    )


# --------------------------------------------------------------------------- #
# log_set happy path + session resolution
# --------------------------------------------------------------------------- #
def test_log_set_creates_open_session_and_second_attaches(client: TestClient) -> None:
    register_user(client)
    ex1, ex2 = get_exercise_ids(client, 2)

    # No open session yet -> log_set creates one.
    r1 = _log_set(client, ex1, weight=135, reps=5)
    assert r1.status_code == 201, r1.text
    set1 = r1.json()
    assert set1["set_number"] == 1
    session_id = set1["session_id"]

    # A second log_set (different exercise) attaches to the SAME open session.
    r2 = _log_set(client, ex2, weight=50, reps=10)
    assert r2.status_code == 201
    assert r2.json()["session_id"] == session_id

    # Only one session exists, and it's open.
    sessions = client.get("/workouts/sessions").json()
    assert len(sessions) == 1
    assert sessions[0]["status"] == "open"


def test_set_number_increments_per_exercise(client: TestClient) -> None:
    register_user(client)
    ex1, ex2 = get_exercise_ids(client, 2)

    assert _log_set(client, ex1).json()["set_number"] == 1
    assert _log_set(client, ex1).json()["set_number"] == 2
    # Different exercise restarts at 1 within the same session.
    assert _log_set(client, ex2).json()["set_number"] == 1
    assert _log_set(client, ex1).json()["set_number"] == 3


def test_log_set_rir_optional(client: TestClient) -> None:
    register_user(client)
    (ex1,) = get_exercise_ids(client, 1)
    r = client.post("/workouts/sets", json={"exercise_id": ex1, "weight": 100, "reps": 8})
    assert r.status_code == 201
    assert r.json()["rir"] is None


# --------------------------------------------------------------------------- #
# Validation rejections
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "weight,reps,rir",
    [
        (-1, 5, 2),  # negative weight
        (100, 0, 2),  # reps below 1
        (100, 101, 2),  # reps above 100
        (100, 5, 0.25),  # rir not a 0.5 increment
        (100, 5, 11),  # rir above 10
    ],
)
def test_log_set_validation_rejected(client: TestClient, weight, reps, rir) -> None:
    register_user(client)
    (ex1,) = get_exercise_ids(client, 1)
    r = client.post(
        "/workouts/sets",
        json={"exercise_id": ex1, "weight": weight, "reps": reps, "rir": rir},
    )
    assert r.status_code == 422, r.text


def test_log_set_unknown_exercise_is_structured_404_not_500(client: TestClient) -> None:
    register_user(client)
    # A well-formed but non-existent UUID must not raise an FK IntegrityError/500.
    r = client.post(
        "/workouts/sets",
        json={
            "exercise_id": "00000000-0000-0000-0000-000000000000",
            "weight": 100,
            "reps": 5,
        },
    )
    assert r.status_code == 404, r.text
    assert "not found" in r.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# Session lifecycle: 409 when open, finish then start
# --------------------------------------------------------------------------- #
def test_start_session_conflicts_when_open_exists(client: TestClient) -> None:
    register_user(client)
    first = client.post("/workouts/sessions", json={})
    assert first.status_code == 201

    second = client.post("/workouts/sessions", json={})
    assert second.status_code == 409


def test_log_set_then_start_session_conflicts(client: TestClient) -> None:
    """log_set's auto-created session also blocks a manual start (one open invariant)."""
    register_user(client)
    (ex1,) = get_exercise_ids(client, 1)
    _log_set(client, ex1)
    assert client.post("/workouts/sessions", json={}).status_code == 409


def test_finish_then_start_works(client: TestClient) -> None:
    register_user(client)
    first = client.post("/workouts/sessions", json={})
    session_id = first.json()["id"]

    fin = client.post(f"/workouts/sessions/{session_id}/finish")
    assert fin.status_code == 200
    assert fin.json()["status"] == "finished"

    # With the prior session finished, a new one may start.
    again = client.post("/workouts/sessions", json={})
    assert again.status_code == 201
    assert again.json()["id"] != session_id


def test_finish_nonexistent_session_404(client: TestClient) -> None:
    register_user(client)
    r = client.post("/workouts/sessions/00000000-0000-0000-0000-000000000000/finish")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# History filtering + ordering
# --------------------------------------------------------------------------- #
def test_history_filter_by_exercise(client: TestClient) -> None:
    register_user(client)
    ex1, ex2 = get_exercise_ids(client, 2)
    _log_set(client, ex1, weight=100, reps=5)
    _log_set(client, ex2, weight=60, reps=10)

    all_hist = client.get("/workouts/history").json()
    assert len(all_hist) == 2
    assert {h["exercise_name"] for h in all_hist}  # names joined in

    filtered = client.get("/workouts/history", params={"exercise_id": ex1}).json()
    assert len(filtered) == 1
    assert filtered[0]["exercise_id"] == ex1


def test_history_most_recent_first(client: TestClient) -> None:
    register_user(client)
    (ex1,) = get_exercise_ids(client, 1)
    _log_set(client, ex1, weight=100, reps=5)
    _log_set(client, ex1, weight=105, reps=5)
    _log_set(client, ex1, weight=110, reps=5)

    hist = client.get("/workouts/history").json()
    created = [h["created_at"] for h in hist]
    assert created == sorted(created, reverse=True)
    # Most recent write (set_number 3) leads.
    assert hist[0]["set_number"] == 3


def test_history_date_range_filter(client: TestClient) -> None:
    register_user(client)
    (ex1,) = get_exercise_ids(client, 1)
    _log_set(client, ex1, weight=100, reps=5)

    # A future window excludes today's set; a wide window includes it.
    empty = client.get(
        "/workouts/history",
        params={"start_date": "2099-01-01", "end_date": "2099-12-31"},
    ).json()
    assert empty == []

    included = client.get(
        "/workouts/history",
        params={"start_date": "2000-01-01", "end_date": "2099-12-31"},
    ).json()
    assert len(included) == 1
