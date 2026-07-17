"""Program endpoint tests.

Covers create, detail, update-within-cap, the 10% load-jump cap rejecting the
whole update atomically (with prior/requested in the detail), cross-user 404, and
the auth guard. The catalog is seeded by conftest.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.rate_limit import limiter
from tests.helpers import authed_client, get_exercise_ids, register_user, unique_email


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    limiter.reset()


def _create_program(client: TestClient, exercises: list[dict], name="Push Day"):
    return client.post("/programs", json={"name": name, "exercises": exercises})


def test_programs_require_auth(client: TestClient) -> None:
    client.cookies.clear()
    assert client.get("/programs").status_code == 401
    assert client.post("/programs", json={"name": "x", "exercises": []}).status_code == 401


def test_create_and_get_program(client: TestClient) -> None:
    register_user(client)
    ex1, ex2 = get_exercise_ids(client, 2)
    resp = _create_program(
        client,
        [
            {"exercise_id": ex1, "target_sets": 3, "target_reps": 5, "target_weight": 100.0},
            {"exercise_id": ex2, "target_sets": 3, "target_reps": 8, "target_weight": 40.0},
        ],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "Push Day"
    assert len(body["exercises"]) == 2
    # Ordered by order_index (defaulted to list position).
    assert [e["order_index"] for e in body["exercises"]] == [0, 1]
    assert body["exercises"][0]["exercise_name"]  # names joined in

    # Detail round-trips and appears in the list.
    detail = client.get(f"/programs/{body['id']}")
    assert detail.status_code == 200
    assert detail.json()["id"] == body["id"]
    assert any(p["id"] == body["id"] for p in client.get("/programs").json())


def test_update_within_cap_succeeds(client: TestClient) -> None:
    register_user(client)
    (ex1,) = get_exercise_ids(client, 1)
    created = _create_program(client, [{"exercise_id": ex1, "target_weight": 100.0}]).json()

    # +10% exactly (100 -> 110) is allowed (cap is "> prior * 1.10").
    upd = client.put(
        f"/programs/{created['id']}",
        json={"exercises": [{"exercise_id": ex1, "target_weight": 110.0}]},
    )
    assert upd.status_code == 200, upd.text
    assert upd.json()["exercises"][0]["target_weight"] == 110.0


def test_update_over_cap_rejects_whole_update_atomically(client: TestClient) -> None:
    register_user(client)
    ex1, ex2 = get_exercise_ids(client, 2)
    created = _create_program(
        client,
        [
            {"exercise_id": ex1, "target_weight": 100.0},
            {"exercise_id": ex2, "target_weight": 50.0},
        ],
    ).json()
    program_id = created["id"]

    # ex1 within cap (110), ex2 over cap (50 -> 60 = +20%). Whole update must fail.
    upd = client.put(
        f"/programs/{program_id}",
        json={
            "exercises": [
                {"exercise_id": ex1, "target_weight": 110.0},
                {"exercise_id": ex2, "target_weight": 60.0},
            ]
        },
    )
    assert upd.status_code == 422, upd.text
    detail = upd.json()["detail"]
    assert "50" in detail and "60" in detail  # prior + requested named
    assert ex2 in detail  # offending exercise named

    # Nothing was written: ex1 stays at 100 (not the requested 110).
    after = client.get(f"/programs/{program_id}").json()
    weights = {e["exercise_id"]: e["target_weight"] for e in after["exercises"]}
    assert weights[ex1] == 100.0
    assert weights[ex2] == 50.0


def test_create_has_no_cap(client: TestClient) -> None:
    """A create has no prior, so any weight is accepted."""
    register_user(client)
    (ex1,) = get_exercise_ids(client, 1)
    resp = _create_program(client, [{"exercise_id": ex1, "target_weight": 500.0}])
    assert resp.status_code == 201


def test_update_program_not_owned_404(client: TestClient) -> None:
    # User A owns a program.
    a_email = unique_email()
    register_user(client, email=a_email)
    (ex1,) = get_exercise_ids(client, 1)
    created = _create_program(client, [{"exercise_id": ex1, "target_weight": 100.0}]).json()
    program_id = created["id"]

    # User B cannot update it -> 404 (ownership not leaked).
    b_email = unique_email()
    register_user(client, email=b_email)
    upd = client.put(
        f"/programs/{program_id}",
        json={"exercises": [{"exercise_id": ex1, "target_weight": 105.0}]},
    )
    assert upd.status_code == 404

    # And A's data is untouched.
    with authed_client(client, a_email):
        after = client.get(f"/programs/{program_id}").json()
        assert after["exercises"][0]["target_weight"] == 100.0
