"""Shared fixtures for the backend test suite."""

from __future__ import annotations

import json
import os
import sqlite3
from types import SimpleNamespace

# A fake key so importing the app never requires a real secret or .env file.
# Individual tests still mock the Gemini client and never make live calls.
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.db.ingest import _create_schema
from backend.app.db.session import session_scope as real_session_scope
from backend.app.vectorstore import faiss_index as faiss_index_module


SAMPLE_PRODUCTS = [
    {
        "product_id": "cosmetics-000001",
        "name": "Gentle Foaming Cleanser",
        "brand": "Aurora Labs",
        "category": "Cleanser",
        "skin_type": "Oily,Sensitive",
        "ingredients": "Water, Glycerin, Salicylic Acid",
        "description": "Gentle foaming cleanser for oily, sensitive skin.",
        "price": 18.5,
        "source_rank": 4.2,
        "embedding_id": 0,
    },
    {
        "product_id": "cosmetics-000002",
        "name": "Oil-Free Mattifying Moisturizer",
        "brand": "Aurora Labs",
        "category": "Moisturizer",
        "skin_type": "Oily",
        "ingredients": "Water, Niacinamide, Dimethicone",
        "description": "Lightweight oil-free moisturizer for oily skin.",
        "price": 24.99,
        "source_rank": 4.5,
        "embedding_id": 1,
    },
    {
        "product_id": "cosmetics-000003",
        "name": "Rich Repair Night Cream",
        "brand": "Verdant",
        "category": "Moisturizer",
        "skin_type": "Dry,Sensitive",
        "ingredients": "Shea Butter, Ceramides, Squalane",
        "description": "Rich night cream for dry, sensitive skin.",
        "price": 32.0,
        "source_rank": 4.1,
        "embedding_id": 2,
    },
    {
        "product_id": "cosmetics-000004",
        "name": "Brightening Vitamin C Serum",
        "brand": "Lumina",
        "category": "Serum",
        "skin_type": "Normal,Dry",
        "ingredients": "Water, Ascorbic Acid, Ferulic Acid",
        "description": "Brightening vitamin C serum for normal, dry skin.",
        "price": 45.0,
        "source_rank": 4.7,
        "embedding_id": 3,
    },
    {
        "product_id": "cosmetics-000005",
        "name": "Mineral Sunscreen SPF 50",
        "brand": "Lumina",
        "category": "Sun protect",
        "skin_type": "Combination,Normal",
        "ingredients": "Zinc Oxide, Titanium Dioxide",
        "description": "Mineral sunscreen SPF 50 for combination, normal skin.",
        "price": 21.0,
        "source_rank": 4.3,
        "embedding_id": 4,
    },
    {
        "product_id": "cosmetics-000006",
        "name": "Purifying Clay Mask",
        "brand": "Verdant",
        "category": "Mask",
        "skin_type": "Oily",
        "ingredients": "Kaolin Clay, Water",
        "description": "Purifying clay mask for oily skin.",
        "price": None,
        "source_rank": 3.9,
        "embedding_id": 5,
    },
]


@pytest.fixture
def sample_products() -> list[dict]:
    """Return a fresh copy of the canonical fixture product rows."""
    return [dict(product) for product in SAMPLE_PRODUCTS]


@pytest.fixture
def test_db_path(tmp_path, sample_products):
    """Create a temporary SQLite DB seeded with fixture products."""
    db_path = tmp_path / "test.db"
    connection = sqlite3.connect(db_path)
    try:
        _create_schema(connection)
        connection.executemany(
            """
            INSERT INTO products (
                product_id, name, brand, category, skin_type, ingredients,
                description, price, source_rank, embedding_id
            ) VALUES (
                :product_id, :name, :brand, :category, :skin_type, :ingredients,
                :description, :price, :source_rank, :embedding_id
            )
            """,
            sample_products,
        )
        connection.commit()
    finally:
        connection.close()
    return db_path


@pytest.fixture
def fixture_session_scope(test_db_path):
    """A drop-in replacement for ``session_scope`` bound to the fixture DB."""

    def _session_scope(*_args, **_kwargs):
        return real_session_scope(test_db_path)

    return _session_scope


@pytest.fixture
def test_index_paths(tmp_path, sample_products):
    """Build a small, real FAISS index whose vectors are one-hot per product.

    A query embedding equal to ``embeddings[i]`` is a perfect match for
    product ``i`` and orthogonal to every other product, making search
    results fully deterministic for tests.
    """
    count = len(sample_products)
    embeddings = np.eye(count, dtype=np.float32)
    product_ids = np.array([p["product_id"] for p in sample_products], dtype=str)

    embeddings_path = tmp_path / "product_embeddings.npy"
    ids_path = tmp_path / "product_ids.npy"
    np.save(embeddings_path, embeddings)
    np.save(ids_path, product_ids)

    index_path = tmp_path / "index" / "products.faiss"
    mapping_path = tmp_path / "index" / "product_ids.json"
    faiss_index_module.build_index(
        embeddings_path=embeddings_path,
        product_ids_path=ids_path,
        index_path=index_path,
        id_mapping_path=mapping_path,
    )

    return SimpleNamespace(
        index_path=index_path,
        id_mapping_path=mapping_path,
        embeddings=embeddings,
        product_ids=list(product_ids),
    )


def _product_ids_from_prompt(contents: str) -> list[str]:
    """Pull the product_id values out of the ``<products>`` JSON block."""
    products_json = contents.split("<products>")[1].split("</products>")[0]
    return [item["product_id"] for item in json.loads(products_json)]


@pytest.fixture
def mock_gemini(monkeypatch):
    """Patch google.genai so it returns one canned explanation per product.

    No live network call is ever made. The fake response mirrors the real
    structured-output contract: a JSON object with an ``explanations`` list
    of ``{product_id, explanation}`` pairs, one per product in the prompt.
    """
    calls: list[dict] = []

    class _FakeResponse:
        def __init__(self, contents: str):
            product_ids = _product_ids_from_prompt(contents)
            self.text = json.dumps(
                {
                    "explanations": [
                        {
                            "product_id": product_id,
                            "explanation": f"Recommended match for product {product_id}.",
                        }
                        for product_id in product_ids
                    ]
                }
            )
            self.parsed = None

    class _FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            return _FakeResponse(contents)

    class _FakeClient:
        def __init__(self, api_key):
            self.models = _FakeModels()

        def close(self):
            pass

    import google.genai as genai_module

    monkeypatch.setattr(genai_module, "Client", _FakeClient)
    return calls


@pytest.fixture
def client():
    """A FastAPI TestClient for the real app."""
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client
