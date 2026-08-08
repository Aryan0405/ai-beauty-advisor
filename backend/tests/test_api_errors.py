"""Cross-cutting checks for the structured error envelope (spec section 15).

Every non-2xx response body must be shaped ``{"error": {"code", "message"}}``
regardless of which endpoint or failure mode produced it. Endpoint-specific
behavior (ranking, filters, explanations) is covered in the per-endpoint
test files; this file is about the error *contract* staying consistent.
"""

from __future__ import annotations

from functools import partial

from fastapi.testclient import TestClient

from backend.app.api.v1.endpoints import products as products_module
from backend.app.api.v1.endpoints import recommendations as recommendations_endpoint_module
from backend.app.services import recommendation as recommendation_module
from backend.app.vectorstore.faiss_index import search_index as real_search_index


def _assert_error_shape(body: dict, code: str, message_contains: str | None = None) -> None:
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    if message_contains is not None:
        assert message_contains in body["error"]["message"]


def test_health_check_returns_200_with_plain_body(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    # Health is the one endpoint that never wraps in the error envelope.
    assert "error" not in response.json()


def test_recommend_success_returns_200_with_no_error_key(
    client, monkeypatch, fixture_session_scope, test_index_paths, mock_gemini
):
    monkeypatch.setattr(
        recommendation_module, "embed_query", lambda query: test_index_paths.embeddings[0]
    )
    monkeypatch.setattr(
        recommendation_module,
        "search_index",
        partial(
            real_search_index,
            index_path=test_index_paths.index_path,
            id_mapping_path=test_index_paths.id_mapping_path,
        ),
    )
    monkeypatch.setattr(recommendation_module, "session_scope", fixture_session_scope)

    response = client.post("/api/v1/recommend", json={"query": "gentle cleanser"})

    assert response.status_code == 200
    assert "error" not in response.json()


def test_pydantic_validation_failure_returns_400_structured(client):
    # top_k=0 violates Field(ge=1) -- a request-validation failure FastAPI
    # would default to 422 for; spec section 15 wants 400 instead.
    response = client.post("/api/v1/recommend", json={"query": "serum", "top_k": 0})
    assert response.status_code == 400
    _assert_error_shape(response.json(), "invalid_request")


def test_missing_required_field_returns_400_structured(client):
    response = client.post("/api/v1/recommend", json={})
    assert response.status_code == 400
    _assert_error_shape(response.json(), "invalid_request", message_contains="query")


def test_service_value_error_returns_400_structured(client):
    # Passes schema validation (non-empty string) but fails the service's
    # own business-rule check once it strips whitespace.
    response = client.post("/api/v1/recommend", json={"query": "   "})
    assert response.status_code == 400
    _assert_error_shape(response.json(), "invalid_request", message_contains="empty")


def test_product_not_found_returns_404_structured(
    client, monkeypatch, fixture_session_scope
):
    monkeypatch.setattr(products_module, "session_scope", fixture_session_scope)

    response = client.get("/api/v1/products/no-such-id")

    assert response.status_code == 404
    _assert_error_shape(response.json(), "not_found", message_contains="not found")


def test_retrieval_unavailable_returns_503_structured(client, monkeypatch):
    monkeypatch.setattr(recommendation_module, "embed_query", lambda query: query)

    def _broken_search_index(*_args, **_kwargs):
        raise RuntimeError("faiss index missing")

    monkeypatch.setattr(recommendation_module, "search_index", _broken_search_index)

    response = client.post("/api/v1/recommend", json={"query": "hydrating serum"})

    assert response.status_code == 503
    _assert_error_shape(response.json(), "service_unavailable")


def test_unhandled_exception_returns_500_structured_without_leaking_details(monkeypatch):
    def _explode(query, filters=None, top_k=5):
        raise KeyError("a database column that should never be missing")

    monkeypatch.setattr(recommendations_endpoint_module, "get_recommendations", _explode)

    # ServerErrorMiddleware sends our custom 500 response and *then*
    # re-raises the exception (so a real ASGI server can log it -- the
    # response bytes are already on the wire by that point). TestClient's
    # default raise_server_exceptions=True surfaces that re-raise as a
    # Python exception instead of the response, so it must be disabled
    # here to actually observe what a real client receives.
    from backend.app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/recommend", json={"query": "hydrating serum"})

    assert response.status_code == 500
    body = response.json()
    _assert_error_shape(body, "internal_error")
    # The generic message must not leak the exception's own text.
    assert "database column" not in body["error"]["message"]
    assert body["error"]["message"] == "An unexpected error occurred."


def test_unhandled_exception_in_products_endpoint_returns_500_structured(monkeypatch):
    def _explode(db, product_id):
        raise KeyError("unexpected repository failure")

    monkeypatch.setattr(products_module, "get_product_by_id", _explode)

    from backend.app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/products/cosmetics-000001")

    assert response.status_code == 500
    _assert_error_shape(response.json(), "internal_error")
