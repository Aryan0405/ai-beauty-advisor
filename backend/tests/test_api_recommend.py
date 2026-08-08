"""Integration tests for POST /api/v1/recommend.

Uses a real fixture SQLite DB and a real (tiny, one-hot) FAISS index so the
full retrieval path executes for real; only the embedding model and the
Gemini client are mocked.
"""

from __future__ import annotations

from functools import partial

from backend.app.services import recommendation as recommendation_module
from backend.app.vectorstore.faiss_index import search_index as real_search_index


def _wire_recommendation(monkeypatch, fixture_session_scope, test_index_paths, product_index):
    """Route the recommendation service at the real fixture DB and FAISS index.

    ``product_index`` selects which fixture product the mocked query
    embedding will perfectly match (one-hot embeddings => score 1.0).
    """
    monkeypatch.setattr(
        recommendation_module,
        "embed_query",
        lambda query: test_index_paths.embeddings[product_index],
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


def test_recommend_returns_ranked_results_with_explanation(
    client, monkeypatch, fixture_session_scope, test_index_paths, mock_gemini
):
    _wire_recommendation(monkeypatch, fixture_session_scope, test_index_paths, product_index=1)

    response = client.post(
        "/api/v1/recommend",
        json={"query": "oil free moisturizer for oily skin", "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "oil free moisturizer for oily skin"
    assert body["count"] == 3
    assert len(body["recommendations"]) == 3

    top = body["recommendations"][0]
    assert top["product_id"] == "cosmetics-000002"
    assert top["name"] == "Oil-Free Mattifying Moisturizer"
    assert top["similarity_score"] > 0.99
    assert set(top.keys()) == {
        "product_id",
        "name",
        "brand",
        "category",
        "skin_type",
        "ingredients",
        "description",
        "price",
        "source_rank",
        "similarity_score",
        "match_score",
        "explanation",
    }

    # A single Gemini call produced a distinct, grounded explanation for
    # every returned product -- not one explanation shared across the group.
    assert len(mock_gemini) == 1
    explanations = [item["explanation"] for item in body["recommendations"]]
    assert all(explanation is not None for explanation in explanations)
    assert len(set(explanations)) == len(explanations)
    for item in body["recommendations"]:
        assert item["explanation"] == f"Recommended match for product {item['product_id']}."


def test_recommend_orders_results_by_descending_match_score(
    client, monkeypatch, fixture_session_scope, test_index_paths, mock_gemini
):
    _wire_recommendation(monkeypatch, fixture_session_scope, test_index_paths, product_index=3)

    response = client.post(
        "/api/v1/recommend", json={"query": "vitamin c serum", "top_k": 6}
    )

    assert response.status_code == 200
    scores = [item["match_score"] for item in response.json()["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    assert response.json()["recommendations"][0]["product_id"] == "cosmetics-000004"


def test_recommend_applies_category_filter(
    client, monkeypatch, fixture_session_scope, test_index_paths, mock_gemini
):
    # Query embedding matches the moisturizer (index 1), but the filter
    # restricts results to Serum -- only cosmetics-000004 should come back.
    _wire_recommendation(monkeypatch, fixture_session_scope, test_index_paths, product_index=1)

    response = client.post(
        "/api/v1/recommend",
        json={"query": "anything", "top_k": 6, "filters": {"category": "Serum"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["recommendations"][0]["product_id"] == "cosmetics-000004"


def test_recommend_rejects_empty_query_with_422(client):
    response = client.post("/api/v1/recommend", json={"query": ""})
    assert response.status_code == 422


def test_recommend_rejects_whitespace_only_query_with_400(client):
    response = client.post("/api/v1/recommend", json={"query": "   "})
    assert response.status_code == 400


def test_recommend_rejects_top_k_out_of_bounds(client):
    assert client.post("/api/v1/recommend", json={"query": "serum", "top_k": 0}).status_code == 422
    assert client.post("/api/v1/recommend", json={"query": "serum", "top_k": 21}).status_code == 422


def test_recommend_returns_503_when_retrieval_subsystem_fails(client, monkeypatch):
    monkeypatch.setattr(
        recommendation_module, "embed_query", lambda query: query
    )

    def _broken_search_index(*_args, **_kwargs):
        raise RuntimeError("faiss-cpu is required to build or search the index.")

    monkeypatch.setattr(recommendation_module, "search_index", _broken_search_index)

    response = client.post("/api/v1/recommend", json={"query": "hydrating serum"})

    assert response.status_code == 503


def test_recommend_degrades_gracefully_when_gemini_fails(
    client, monkeypatch, fixture_session_scope, test_index_paths
):
    _wire_recommendation(monkeypatch, fixture_session_scope, test_index_paths, product_index=0)

    import google.genai as genai_module

    class _FailingClient:
        def __init__(self, api_key):
            pass

        @property
        def models(self):
            raise RuntimeError("gemini unreachable")

        def close(self):
            pass

    monkeypatch.setattr(genai_module, "Client", _FailingClient)

    response = client.post("/api/v1/recommend", json={"query": "gentle cleanser"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) > 0
    assert all(item["explanation"] is None for item in body["recommendations"])


def test_recommend_nulls_out_products_missing_from_a_partial_gemini_batch(
    client, monkeypatch, fixture_session_scope, test_index_paths
):
    """Gemini answering for only some products degrades per-item, not for the whole request."""
    _wire_recommendation(monkeypatch, fixture_session_scope, test_index_paths, product_index=1)

    import json

    import google.genai as genai_module

    class _FakeResponse:
        # Only ever explains the first product a batch is asked about.
        def __init__(self, contents):
            products_json = contents.split("<products>")[1].split("</products>")[0]
            first_id = json.loads(products_json)[0]["product_id"]
            self.text = json.dumps(
                {
                    "explanations": [
                        {"product_id": first_id, "explanation": "Only this one got explained."}
                    ]
                }
            )
            self.parsed = None

    class _FakeModels:
        def generate_content(self, *, model, contents, config):
            return _FakeResponse(contents)

    class _FakeClient:
        def __init__(self, api_key):
            self.models = _FakeModels()

        def close(self):
            pass

    monkeypatch.setattr(genai_module, "Client", _FakeClient)

    response = client.post(
        "/api/v1/recommend", json={"query": "oily moisturizer", "top_k": 6}
    )

    assert response.status_code == 200
    recommendations = response.json()["recommendations"]
    explained = [item for item in recommendations if item["explanation"] is not None]
    unexplained = [item for item in recommendations if item["explanation"] is None]
    assert len(explained) == 1
    assert len(unexplained) == len(recommendations) - 1
    assert explained[0]["explanation"] == "Only this one got explained."
