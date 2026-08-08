"""Unit tests for the batched, structured-output Gemini explanation service.

No live network call is ever made: google.genai.Client is monkeypatched in
every test that would otherwise reach it.
"""

from __future__ import annotations

import json

from backend.app.services import explanation as explanation_module
from backend.app.services.explanation import ExplanationBatch, generate_explanations


SAMPLE_PRODUCTS = [
    {
        "product_id": "p1",
        "name": "Oil-Free Moisturizer",
        "brand": "Aurora Labs",
        "category": "Moisturizer",
        "skin_type": "Oily",
        "ingredients": "Water, Niacinamide",
        "description": "Lightweight oil-free moisturizer.",
        "price": 24.99,
        "match_score": 0.87,
    },
    {
        "product_id": "p2",
        "name": "Gentle Cleanser",
        "brand": "Verdant",
        "category": "Cleanser",
        "skin_type": "Sensitive",
        "ingredients": "Water, Glycerin",
        "description": "A gentle daily cleanser.",
        "price": 15.0,
        "match_score": 0.71,
    },
]


def _install_fake_client(monkeypatch, generate_content):
    class _FakeModels:
        def generate_content(self, *, model, contents, config):
            return generate_content(model=model, contents=contents, config=config)

    class _FakeClient:
        def __init__(self, api_key):
            self.models = _FakeModels()

        def close(self):
            pass

    import google.genai as genai_module

    monkeypatch.setattr(genai_module, "Client", _FakeClient)


def _json_response(explanations: list[dict]) -> object:
    class _FakeResponse:
        text = json.dumps({"explanations": explanations})
        parsed = None

    return _FakeResponse()


def test_empty_query_returns_empty_mapping():
    assert generate_explanations("   ", SAMPLE_PRODUCTS) == {}


def test_empty_products_returns_empty_mapping():
    assert generate_explanations("oily moisturizer", []) == {}


def test_missing_api_key_returns_empty_mapping(monkeypatch):
    monkeypatch.setattr(
        explanation_module, "_load_gemini_settings", lambda: ("", "gemini-2.5-flash")
    )
    assert generate_explanations("oily moisturizer", SAMPLE_PRODUCTS) == {}


def test_successful_generation_returns_one_explanation_per_product(monkeypatch):
    captured = {}

    def _generate_content(*, model, contents, config):
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        return _json_response(
            [
                {"product_id": "p1", "explanation": "Great for oily skin."},
                {"product_id": "p2", "explanation": "Gentle enough for daily use."},
            ]
        )

    _install_fake_client(monkeypatch, _generate_content)

    result = generate_explanations("oily moisturizer", SAMPLE_PRODUCTS)

    assert result == {
        "p1": "Great for oily skin.",
        "p2": "Gentle enough for daily use.",
    }
    # Distinct text per product -- not one shared explanation reused everywhere.
    assert result["p1"] != result["p2"]


def test_single_gemini_call_covers_the_whole_batch(monkeypatch):
    call_count = {"n": 0}

    def _generate_content(*, model, contents, config):
        call_count["n"] += 1
        return _json_response(
            [
                {"product_id": "p1", "explanation": "Great for oily skin."},
                {"product_id": "p2", "explanation": "Gentle enough for daily use."},
            ]
        )

    _install_fake_client(monkeypatch, _generate_content)

    generate_explanations("oily moisturizer", SAMPLE_PRODUCTS)

    assert call_count["n"] == 1


def test_uses_structured_output_response_schema(monkeypatch):
    captured = {}

    def _generate_content(*, model, contents, config):
        captured["config"] = config
        return _json_response([{"product_id": "p1", "explanation": "ok"}])

    _install_fake_client(monkeypatch, _generate_content)

    generate_explanations("oily moisturizer", SAMPLE_PRODUCTS[:1])

    assert captured["config"].response_mime_type == "application/json"
    assert captured["config"].response_schema is ExplanationBatch


def test_partial_batch_only_maps_products_present(monkeypatch):
    def _generate_content(*, model, contents, config):
        # Gemini only explained one of the two products.
        return _json_response([{"product_id": "p1", "explanation": "Great for oily skin."}])

    _install_fake_client(monkeypatch, _generate_content)

    result = generate_explanations("oily moisturizer", SAMPLE_PRODUCTS)

    assert result == {"p1": "Great for oily skin."}
    assert "p2" not in result


def test_hallucinated_product_id_is_filtered_out(monkeypatch):
    def _generate_content(*, model, contents, config):
        return _json_response(
            [
                {"product_id": "p1", "explanation": "Great for oily skin."},
                {"product_id": "not-a-real-product", "explanation": "Invented."},
            ]
        )

    _install_fake_client(monkeypatch, _generate_content)

    result = generate_explanations("oily moisturizer", SAMPLE_PRODUCTS)

    assert result == {"p1": "Great for oily skin."}


def test_blank_explanation_text_is_excluded(monkeypatch):
    def _generate_content(*, model, contents, config):
        return _json_response(
            [
                {"product_id": "p1", "explanation": "   "},
                {"product_id": "p2", "explanation": "Gentle enough for daily use."},
            ]
        )

    _install_fake_client(monkeypatch, _generate_content)

    result = generate_explanations("oily moisturizer", SAMPLE_PRODUCTS)

    assert result == {"p2": "Gentle enough for daily use."}


def test_malformed_json_response_returns_empty_mapping(monkeypatch):
    class _FakeResponse:
        text = "not valid json"
        parsed = None

    def _generate_content(*, model, contents, config):
        return _FakeResponse()

    _install_fake_client(monkeypatch, _generate_content)

    assert generate_explanations("oily moisturizer", SAMPLE_PRODUCTS) == {}


def test_uses_response_parsed_when_sdk_provides_it(monkeypatch):
    class _FakeResponse:
        text = "{}"  # would fail to parse if the .parsed shortcut were skipped
        parsed = ExplanationBatch(
            explanations=[
                {"product_id": "p1", "explanation": "Great for oily skin."},
            ]
        )

    def _generate_content(*, model, contents, config):
        return _FakeResponse()

    _install_fake_client(monkeypatch, _generate_content)

    result = generate_explanations("oily moisturizer", SAMPLE_PRODUCTS)

    assert result == {"p1": "Great for oily skin."}


def test_exception_during_generation_returns_empty_mapping(monkeypatch):
    def _generate_content(*, model, contents, config):
        raise TimeoutError("gemini timed out")

    _install_fake_client(monkeypatch, _generate_content)

    assert generate_explanations("oily moisturizer", SAMPLE_PRODUCTS) == {}


def test_client_construction_failure_returns_empty_mapping(monkeypatch):
    import google.genai as genai_module

    def _raise(api_key):
        raise RuntimeError("client init failed")

    monkeypatch.setattr(genai_module, "Client", _raise)

    assert generate_explanations("oily moisturizer", SAMPLE_PRODUCTS) == {}


def test_prompt_includes_product_id_for_correlation(monkeypatch):
    captured = {}

    def _generate_content(*, model, contents, config):
        captured["contents"] = contents
        return _json_response([{"product_id": "p1", "explanation": "ok"}])

    _install_fake_client(monkeypatch, _generate_content)

    generate_explanations("oily moisturizer", SAMPLE_PRODUCTS[:1])

    products_json = captured["contents"].split("<products>")[1].split("</products>")[0]
    parsed = json.loads(products_json)
    assert parsed[0]["product_id"] == "p1"


def test_prompt_omits_internal_fields_not_in_allowlist(monkeypatch):
    captured = {}

    def _generate_content(*, model, contents, config):
        captured["contents"] = contents
        return _json_response([{"product_id": "p1", "explanation": "ok"}])

    _install_fake_client(monkeypatch, _generate_content)

    products = [{**SAMPLE_PRODUCTS[0], "embedding_id": 42, "source_rank": 4.5}]
    generate_explanations("oily moisturizer", products)

    products_json = captured["contents"].split("<products>")[1].split("</products>")[0]
    parsed = json.loads(products_json)
    assert "embedding_id" not in parsed[0]
    assert "source_rank" not in parsed[0]
    assert parsed[0]["name"] == "Oil-Free Moisturizer"
