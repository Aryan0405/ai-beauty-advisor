"""Tests for the correlation-ID middleware and structured logging (spec §16)."""

from __future__ import annotations

import json
import logging
import uuid

import pytest

from backend.app.core.logging import _JSONFormatter, get_request_id, reset_request_id, set_request_id
from backend.app.core.middleware import REQUEST_ID_HEADER


def test_generates_a_uuid4_request_id_when_none_supplied(client):
    response = client.get("/api/v1/health")

    request_id = response.headers.get(REQUEST_ID_HEADER)
    assert request_id is not None
    # Raises ValueError if not a well-formed UUID.
    uuid.UUID(request_id)


def test_echoes_back_a_client_supplied_request_id(client):
    response = client.get(
        "/api/v1/health", headers={REQUEST_ID_HEADER: "client-supplied-id-123"}
    )

    assert response.headers.get(REQUEST_ID_HEADER) == "client-supplied-id-123"


def test_each_request_gets_a_distinct_request_id(client):
    first = client.get("/api/v1/health").headers.get(REQUEST_ID_HEADER)
    second = client.get("/api/v1/health").headers.get(REQUEST_ID_HEADER)

    assert first != second


def test_request_id_header_present_on_structured_error_responses(
    client, monkeypatch, fixture_session_scope
):
    from backend.app.api.v1.endpoints import products as products_module

    monkeypatch.setattr(products_module, "session_scope", fixture_session_scope)

    response = client.get("/api/v1/products/does-not-exist")

    assert response.status_code == 404
    assert response.headers.get(REQUEST_ID_HEADER) is not None


def test_request_id_header_present_on_500_response(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app.api.v1.endpoints import products as products_module
    from backend.app.main import app

    def _explode(db, product_id):
        raise KeyError("boom")

    monkeypatch.setattr(products_module, "get_product_by_id", _explode)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/products/cosmetics-000001")

    assert response.status_code == 500
    assert response.headers.get(REQUEST_ID_HEADER) is not None


class TestJSONFormatter:
    def test_formats_one_json_object_per_line_with_request_id(self):
        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="backend.app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="something_happened",
            args=(),
            exc_info=None,
        )
        record.request_id = "abc-123"
        record.custom_field = 42

        payload = json.loads(formatter.format(record))

        assert payload["message"] == "something_happened"
        assert payload["request_id"] == "abc-123"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "backend.app.test"
        assert payload["custom_field"] == 42
        assert "timestamp" in payload

    def test_defaults_request_id_to_dash_when_unset(self):
        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="backend.app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="no request in flight",
            args=(),
            exc_info=None,
        )

        payload = json.loads(formatter.format(record))

        assert payload["request_id"] == "-"


def test_request_id_contextvar_set_and_reset():
    assert get_request_id() is None

    token = set_request_id("test-id")
    assert get_request_id() == "test-id"

    reset_request_id(token)
    assert get_request_id() is None
