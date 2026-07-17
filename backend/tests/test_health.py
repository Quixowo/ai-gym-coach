from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    """/health runs SELECT 1 through the async engine.

    Uses the shared session-scoped ``client`` fixture (NullPool engine override),
    so it plays nicely alongside the multi-request auth tests. Requires a
    reachable Postgres (CI service / local ``docker compose up -d``).
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
