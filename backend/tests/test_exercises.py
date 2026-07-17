"""Exercise catalog endpoint tests.

Covers listing, fuzzy search (hit + below-threshold miss), and the 401-without-cookie
guard. The catalog is seeded once by the ``seeded_exercises`` conftest fixture.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.rate_limit import limiter
from tests.helpers import register_user


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    limiter.reset()


def test_list_exercises_requires_auth(client: TestClient) -> None:
    client.cookies.clear()
    assert client.get("/exercises").status_code == 401


def test_search_requires_auth(client: TestClient) -> None:
    client.cookies.clear()
    assert client.get("/exercises/search", params={"q": "bench"}).status_code == 401


def test_list_exercises_returns_catalog(client: TestClient) -> None:
    register_user(client)
    resp = client.get("/exercises")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) >= 50  # ~54 seeded
    names = {e["name"] for e in body}
    assert "Barbell Bench Press" in names


def test_search_fuzzy_hit(client: TestClient) -> None:
    register_user(client)
    resp = client.get("/exercises/search", params={"q": "bench press"})
    assert resp.status_code == 200
    matches = resp.json()["matches"]
    assert matches, "expected at least one match for 'bench press'"
    # Best match should be a bench-press variant, with a real score.
    assert "Bench Press" in matches[0]["name"]
    assert matches[0]["score"] >= 60
    assert len(matches) <= 5


def test_search_casual_name_resolves(client: TestClient) -> None:
    register_user(client)
    resp = client.get("/exercises/search", params={"q": "back squat"})
    assert resp.status_code == 200
    matches = resp.json()["matches"]
    assert any("Squat" in m["name"] for m in matches)


def test_search_below_threshold_returns_empty(client: TestClient) -> None:
    register_user(client)
    resp = client.get("/exercises/search", params={"q": "zzzqwerty nonsense xyzzy"})
    assert resp.status_code == 200
    assert resp.json()["matches"] == []
