"""Application-level access-control checklist.

Verifies the cross-user isolation guarantee: user B can never see or modify user
A's sessions, sets, history, or programs. Every service function filters on the
JWT-derived ``user_id``, so a cross-user reference resolves to 404/absent, never
a leak (this is deliberately application-level access control, not Postgres RLS).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.rate_limit import limiter
from tests.helpers import authed_client, get_exercise_ids, register_user, unique_email


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    limiter.reset()


def test_user_b_cannot_see_user_a_sessions_sets_history(client: TestClient) -> None:
    a_email = unique_email()
    register_user(client, email=a_email)
    ex1, ex2 = get_exercise_ids(client, 2)

    # User A logs sets (auto-creating a session) and starts nothing extra.
    client.post("/workouts/sets", json={"exercise_id": ex1, "weight": 100, "reps": 5})
    client.post("/workouts/sets", json={"exercise_id": ex2, "weight": 60, "reps": 8})
    a_sessions = client.get("/workouts/sessions").json()
    assert len(a_sessions) == 1
    a_session_id = a_sessions[0]["id"]
    assert len(client.get("/workouts/history").json()) == 2

    # User B sees none of A's data.
    b_email = unique_email()
    register_user(client, email=b_email)
    assert client.get("/workouts/sessions").json() == []
    assert client.get("/workouts/history").json() == []

    # B cannot finish A's session -> 404 (not 403, ownership not leaked).
    assert client.post(f"/workouts/sessions/{a_session_id}/finish").status_code == 404

    # A's session is still open (B's attempt didn't touch it).
    with authed_client(client, a_email):
        still = client.get("/workouts/sessions").json()
        assert still[0]["status"] == "open"


def test_user_b_cannot_see_or_modify_user_a_programs(client: TestClient) -> None:
    a_email = unique_email()
    register_user(client, email=a_email)
    (ex1,) = get_exercise_ids(client, 1)
    created = client.post(
        "/programs",
        json={"name": "A Secret Plan", "exercises": [{"exercise_id": ex1, "target_weight": 100}]},
    ).json()
    program_id = created["id"]

    b_email = unique_email()
    register_user(client, email=b_email)
    # B's program list is empty; A's program detail 404s for B.
    assert client.get("/programs").json() == []
    assert client.get(f"/programs/{program_id}").status_code == 404
    assert (
        client.put(
            f"/programs/{program_id}",
            json={"exercises": [{"exercise_id": ex1, "target_weight": 105}]},
        ).status_code
        == 404
    )
