"""Unit tests for the precision@5 relevance heuristic itself.

Fast and offline: exercises the pure `_is_relevant` scoring function
directly on synthetic products, without touching the real embedding
model, FAISS index, or SQLite DB that run_precision_eval.py needs for an
actual retrieval run (that part is a manual/QA tool, not a pytest test --
see its module docstring).
"""

from __future__ import annotations

from backend.tests.eval.queries import EVAL_QUERIES
from backend.tests.eval.relevance_cases import RELEVANCE_CASES
from backend.tests.eval.run_precision_eval import _is_relevant


def _product(category: str, skin_type: str) -> dict:
    return {"product_id": "p1", "name": "Test Product", "category": category, "skin_type": skin_type}


def test_relevant_when_category_and_skin_type_match():
    product = _product("Moisturizer", "Oily,Sensitive")
    assert _is_relevant(product, {"Moisturizer"}, {"Oily"}) is True


def test_not_relevant_when_category_does_not_match():
    product = _product("Cleanser", "Oily")
    assert _is_relevant(product, {"Moisturizer"}, {"Oily"}) is False


def test_not_relevant_when_skin_type_tags_dont_intersect():
    product = _product("Moisturizer", "Dry,Sensitive")
    assert _is_relevant(product, {"Moisturizer"}, {"Oily"}) is False


def test_relevant_on_category_alone_when_no_skin_type_expected():
    product = _product("Treatment", "")
    assert _is_relevant(product, {"Treatment"}, set()) is True


def test_relevant_when_category_is_any_of_multiple_expected():
    product = _product("Moisturizer", "")
    assert _is_relevant(product, {"Treatment", "Moisturizer"}, set()) is True


def test_relevant_when_any_expected_skin_type_present():
    product = _product("Sun protect", "Combination,Dry,Normal")
    assert _is_relevant(product, {"Sun protect"}, {"Oily", "Normal"}) is True


def test_relevance_cases_cover_every_curated_query_exactly_once():
    case_queries = [case["query"] for case in RELEVANCE_CASES]
    assert sorted(case_queries) == sorted(EVAL_QUERIES)
    assert len(case_queries) == len(set(case_queries)) == len(EVAL_QUERIES)


def test_every_case_has_at_least_one_expected_category():
    for case in RELEVANCE_CASES:
        assert case["expected_categories"], f"no expected_categories for {case['query']!r}"
