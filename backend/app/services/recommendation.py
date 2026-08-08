"""Hybrid recommendation ranking over FAISS candidates and product metadata."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Mapping

from backend.app.db.repository import get_products_by_ids
from backend.app.db.session import session_scope
from backend.app.models import Product
from backend.app.vectorstore.faiss_index import search_index
from backend.ingestion.build_embeddings import embed_query


LOGGER = logging.getLogger(__name__)
CANDIDATE_COUNT = 50
SIMILARITY_WEIGHT = 0.9
KEYWORD_WEIGHT = 0.075
FILTER_WEIGHT = 0.025
SUPPORTED_FILTERS = {"category", "brand", "skin_type"}


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A product and the scores used to rank it for a user query."""

    product: Product
    similarity_score: float
    match_score: float


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _query_terms(query: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", query.casefold()) if len(term) > 2}


def _matches_filters(product: Product, filters: Mapping[str, str]) -> bool:
    """Check exact category/brand and inclusive skin-type filter matches."""
    for name, value in filters.items():
        expected = _normalized(value)
        if name == "category" and _normalized(product.category) != expected:
            return False
        if name == "brand" and _normalized(product.brand) != expected:
            return False
        if name == "skin_type":
            expected_types = {
                _normalized(item) for item in value.split(",") if item.strip()
            }
            product_types = {
                _normalized(item) for item in product.skin_type.split(",") if item.strip()
            }
            if not expected_types or not expected_types.issubset(product_types):
                return False
    return True


def _keyword_score(product: Product, query_terms: set[str]) -> float:
    """Calculate a small lexical score complementary to semantic similarity."""
    if not query_terms:
        return 0.0
    product_text = " ".join(
        (
            product.name,
            product.brand,
            product.category,
            product.skin_type,
            product.ingredients,
        )
    )
    matched_terms = query_terms.intersection(_query_terms(product_text))
    return len(matched_terms) / len(query_terms)


def get_recommendations(
    query: str, filters: Mapping[str, str] | None = None, top_k: int = 5
) -> list[Recommendation]:
    """Return filtered, hybrid-ranked recommendations for a natural-language query."""
    if not query.strip():
        raise ValueError("query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    active_filters = dict(filters or {})
    unsupported_filters = set(active_filters).difference(SUPPORTED_FILTERS)
    if unsupported_filters:
        names = ", ".join(sorted(unsupported_filters))
        raise ValueError(f"Unsupported filters: {names}")
    if any(not isinstance(value, str) for value in active_filters.values()):
        raise ValueError("Filter values must be strings.")

    candidate_count = max(CANDIDATE_COUNT, top_k * 10)
    retrieval_start = time.perf_counter()
    query_embedding = embed_query(query)
    candidates = search_index(query_embedding, candidate_count)
    retrieval_latency_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
    LOGGER.info(
        "retrieval_completed",
        extra={"retrieval_latency_ms": retrieval_latency_ms, "faiss_result_count": len(candidates)},
    )
    if not candidates:
        return []

    candidate_ids = [product_id for product_id, _ in candidates]
    similarity_by_id = dict(candidates)
    with session_scope() as db:
        products_by_id = {
            product.product_id: product
            for product in get_products_by_ids(db, candidate_ids)
        }

    query_terms = _query_terms(query)
    filter_score = 1.0 if active_filters else 0.0
    ranked_results: list[Recommendation] = []
    for product_id in candidate_ids:
        product = products_by_id.get(product_id)
        if product is None or not _matches_filters(product, active_filters):
            continue

        similarity_score = similarity_by_id[product_id]
        match_score = (
            SIMILARITY_WEIGHT * similarity_score
            + KEYWORD_WEIGHT * _keyword_score(product, query_terms)
            + FILTER_WEIGHT * filter_score
        )
        ranked_results.append(
            Recommendation(product, similarity_score, match_score))

    return sorted(
        ranked_results,
        key=lambda recommendation: recommendation.match_score,
        reverse=True,
    )[:top_k]
