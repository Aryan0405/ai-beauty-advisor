"""Integration tests for GET /api/v1/products/{product_id}."""

from __future__ import annotations

from backend.app.api.v1.endpoints import products as products_module


def _wire_products_db(monkeypatch, fixture_session_scope):
    monkeypatch.setattr(products_module, "session_scope", fixture_session_scope)


def test_get_product_returns_full_metadata(client, monkeypatch, fixture_session_scope):
    _wire_products_db(monkeypatch, fixture_session_scope)

    response = client.get("/api/v1/products/cosmetics-000001")

    assert response.status_code == 200
    body = response.json()
    assert body["product_id"] == "cosmetics-000001"
    assert body["name"] == "Gentle Foaming Cleanser"
    assert body["brand"] == "Aurora Labs"
    assert body["category"] == "Cleanser"
    assert body["price"] == 18.5


def test_get_product_returns_404_for_unknown_id(client, monkeypatch, fixture_session_scope):
    _wire_products_db(monkeypatch, fixture_session_scope)

    response = client.get("/api/v1/products/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"
    assert body["error"]["message"] == "Product not found."


def test_get_product_preserves_null_price(client, monkeypatch, fixture_session_scope):
    _wire_products_db(monkeypatch, fixture_session_scope)

    response = client.get("/api/v1/products/cosmetics-000006")

    assert response.status_code == 200
    assert response.json()["price"] is None
