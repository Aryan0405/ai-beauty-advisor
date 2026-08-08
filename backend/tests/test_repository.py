"""Unit tests for the SQLite repository layer."""

from __future__ import annotations

import pytest

from backend.app.db.repository import (
    get_all_products,
    get_product_by_id,
    get_products_by_category,
    get_products_by_ids,
)
from backend.app.db.session import session_scope


def test_get_product_by_id_returns_matching_product(test_db_path):
    with session_scope(test_db_path) as db:
        product = get_product_by_id(db, "cosmetics-000001")

    assert product is not None
    assert product.name == "Gentle Foaming Cleanser"
    assert product.brand == "Aurora Labs"
    assert product.embedding_id == 0


def test_get_product_by_id_returns_none_for_missing_id(test_db_path):
    with session_scope(test_db_path) as db:
        product = get_product_by_id(db, "does-not-exist")

    assert product is None


def test_get_product_by_id_preserves_nullable_price(test_db_path):
    with session_scope(test_db_path) as db:
        product = get_product_by_id(db, "cosmetics-000006")

    assert product is not None
    assert product.price is None


def test_get_all_products_orders_by_product_id(test_db_path):
    with session_scope(test_db_path) as db:
        products = get_all_products(db, limit=100)

    ids = [product.product_id for product in products]
    assert ids == sorted(ids)
    assert len(products) == 6


def test_get_all_products_respects_limit_and_offset(test_db_path):
    with session_scope(test_db_path) as db:
        page = get_all_products(db, limit=2, offset=2)

    assert [product.product_id for product in page] == [
        "cosmetics-000003",
        "cosmetics-000004",
    ]


def test_get_all_products_rejects_invalid_limit(test_db_path):
    with session_scope(test_db_path) as db:
        with pytest.raises(ValueError):
            get_all_products(db, limit=0)


def test_get_all_products_rejects_negative_offset(test_db_path):
    with session_scope(test_db_path) as db:
        with pytest.raises(ValueError):
            get_all_products(db, limit=10, offset=-1)


def test_get_products_by_category_is_case_insensitive(test_db_path):
    with session_scope(test_db_path) as db:
        products = get_products_by_category(db, "moisturizer")

    ids = {product.product_id for product in products}
    assert ids == {"cosmetics-000002", "cosmetics-000003"}


def test_get_products_by_category_returns_empty_for_unknown_category(test_db_path):
    with session_scope(test_db_path) as db:
        products = get_products_by_category(db, "Lipstick")

    assert products == []


def test_get_products_by_ids_returns_requested_products(test_db_path):
    with session_scope(test_db_path) as db:
        products = get_products_by_ids(
            db, ["cosmetics-000004", "cosmetics-000001", "missing-id"]
        )

    ids = {product.product_id for product in products}
    assert ids == {"cosmetics-000001", "cosmetics-000004"}


def test_get_products_by_ids_returns_empty_list_for_empty_input(test_db_path):
    with session_scope(test_db_path) as db:
        products = get_products_by_ids(db, [])

    assert products == []
