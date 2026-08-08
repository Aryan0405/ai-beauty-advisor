"""Unit tests for the hybrid ranking logic in services/recommendation.py.

FAISS, the embedding model, and the database are all mocked here so the
ranking math and filter logic can be exercised in isolation and fast.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

from backend.app.models import Product
from backend.app.services import recommendation as recommendation_module
from backend.app.services.recommendation import get_recommendations


def _product(**overrides) -> Product:
    defaults = dict(
        product_id="p1",
        name="Test Product",
        brand="TestBrand",
        category="Moisturizer",
        skin_type="Oily",
        ingredients="Water",
        description="A test product.",
        price=10.0,
        source_rank=4.0,
        embedding_id=0,
    )
    defaults.update(overrides)
    return Product(**defaults)


@contextmanager
def _fake_session_scope(*_args, **_kwargs):
    yield None


def _patch_embedding_and_search(monkeypatch, candidates):
    monkeypatch.setattr(
        recommendation_module, "embed_query", lambda query: np.zeros(2, dtype=np.float32)
    )
    monkeypatch.setattr(
        recommendation_module, "search_index", lambda embedding, k: candidates
    )
    monkeypatch.setattr(recommendation_module, "session_scope", _fake_session_scope)


def test_empty_query_raises_value_error(monkeypatch):
    _patch_embedding_and_search(monkeypatch, [])
    with pytest.raises(ValueError):
        get_recommendations("   ")


def test_invalid_top_k_raises_value_error(monkeypatch):
    _patch_embedding_and_search(monkeypatch, [])
    with pytest.raises(ValueError):
        get_recommendations("hydrating serum", top_k=0)


def test_unsupported_filter_raises_value_error(monkeypatch):
    _patch_embedding_and_search(monkeypatch, [])
    with pytest.raises(ValueError):
        get_recommendations("hydrating serum", filters={"color": "red"})


def test_non_string_filter_value_raises_value_error(monkeypatch):
    _patch_embedding_and_search(monkeypatch, [])
    with pytest.raises(ValueError):
        get_recommendations("hydrating serum", filters={"category": 123})


def test_no_candidates_returns_empty_list(monkeypatch):
    _patch_embedding_and_search(monkeypatch, [])
    assert get_recommendations("hydrating serum") == []


def test_products_missing_from_db_are_skipped(monkeypatch):
    _patch_embedding_and_search(monkeypatch, [("p1", 0.9), ("missing", 0.8)])
    monkeypatch.setattr(
        recommendation_module,
        "get_products_by_ids",
        lambda db, ids: [_product(product_id="p1")],
    )

    results = get_recommendations("hydrating serum")

    assert [r.product.product_id for r in results] == ["p1"]


def test_category_filter_excludes_non_matching_products(monkeypatch):
    candidates = [("p1", 0.9), ("p2", 0.8)]
    _patch_embedding_and_search(monkeypatch, candidates)
    products = [
        _product(product_id="p1", category="Moisturizer"),
        _product(product_id="p2", category="Cleanser"),
    ]
    monkeypatch.setattr(
        recommendation_module, "get_products_by_ids", lambda db, ids: products
    )

    results = get_recommendations("anything", filters={"category": "cleanser"})

    assert [r.product.product_id for r in results] == ["p2"]


def test_skin_type_filter_requires_all_listed_types_present(monkeypatch):
    candidates = [("p1", 0.9)]
    _patch_embedding_and_search(monkeypatch, candidates)
    products = [_product(product_id="p1", skin_type="Oily,Sensitive")]
    monkeypatch.setattr(
        recommendation_module, "get_products_by_ids", lambda db, ids: products
    )

    matches = get_recommendations("anything", filters={"skin_type": "Oily"})
    assert len(matches) == 1

    no_matches = get_recommendations("anything", filters={"skin_type": "Oily,Dry"})
    assert no_matches == []


def test_brand_filter_is_case_insensitive_exact_match(monkeypatch):
    candidates = [("p1", 0.9)]
    _patch_embedding_and_search(monkeypatch, candidates)
    products = [_product(product_id="p1", brand="Aurora Labs")]
    monkeypatch.setattr(
        recommendation_module, "get_products_by_ids", lambda db, ids: products
    )

    matches = get_recommendations("anything", filters={"brand": "aurora labs"})
    assert len(matches) == 1

    no_matches = get_recommendations("anything", filters={"brand": "Verdant"})
    assert no_matches == []


def test_match_score_combines_similarity_keyword_and_filter_weights(monkeypatch):
    candidates = [("p1", 0.8)]
    _patch_embedding_and_search(monkeypatch, candidates)
    product = _product(
        product_id="p1",
        name="Oily Skin Moisturizer",
        brand="Aurora Labs",
        category="Moisturizer",
        skin_type="Oily",
        ingredients="Water",
    )
    monkeypatch.setattr(
        recommendation_module, "get_products_by_ids", lambda db, ids: [product]
    )

    results = get_recommendations(
        "oily moisturizer", filters={"category": "Moisturizer"}
    )

    assert len(results) == 1
    result = results[0]
    assert result.similarity_score == pytest.approx(0.8)
    # query terms {"oily", "moisturizer"} both appear in the product text -> keyword_score = 1.0
    expected = 0.9 * 0.8 + 0.075 * 1.0 + 0.025 * 1.0
    assert result.match_score == pytest.approx(expected)


def test_results_are_sorted_by_match_score_and_truncated_to_top_k(monkeypatch):
    candidates = [("low", 0.1), ("high", 0.9), ("mid", 0.5)]
    _patch_embedding_and_search(monkeypatch, candidates)
    products = [
        _product(product_id="low", name="Low"),
        _product(product_id="high", name="High"),
        _product(product_id="mid", name="Mid"),
    ]
    monkeypatch.setattr(
        recommendation_module, "get_products_by_ids", lambda db, ids: products
    )

    results = get_recommendations("anything", top_k=2)

    assert [r.product.product_id for r in results] == ["high", "mid"]
    assert len(results) == 2


def test_keyword_score_is_zero_when_query_has_no_meaningful_terms(monkeypatch):
    candidates = [("p1", 0.5)]
    _patch_embedding_and_search(monkeypatch, candidates)
    product = _product(product_id="p1")
    monkeypatch.setattr(
        recommendation_module, "get_products_by_ids", lambda db, ids: [product]
    )

    # "to" and "of" are <= 2 chars and filtered out of query terms.
    results = get_recommendations("to of")

    assert results[0].match_score == pytest.approx(0.9 * 0.5)
