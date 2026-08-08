"""Integration tests for GET /api/v1/health."""

from __future__ import annotations

from backend.app.api.v1.endpoints import health as health_module


def test_health_check_reports_ok_against_real_fixtures(client):
    """The real repo's committed DB and FAISS index should both be reachable."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["db_connected"] is True


def test_health_check_reports_db_connected_false_when_db_unavailable(client, monkeypatch):
    def _broken_session_scope(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(health_module, "session_scope", _broken_session_scope)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["db_connected"] is False


def test_health_check_reports_index_loaded_false_when_index_missing(client, monkeypatch):
    monkeypatch.setattr(health_module, "_index_loaded", lambda: False)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["index_loaded"] is False
