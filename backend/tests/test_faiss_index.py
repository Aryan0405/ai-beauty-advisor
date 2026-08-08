"""Unit tests for FAISS index build, persistence, and search."""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.app.vectorstore.faiss_index import build_index, search_index


def _write_embeddings(tmp_path, embeddings, product_ids):
    embeddings_path = tmp_path / "embeddings.npy"
    ids_path = tmp_path / "ids.npy"
    np.save(embeddings_path, np.asarray(embeddings, dtype=np.float32))
    np.save(ids_path, np.asarray(product_ids, dtype=str))
    return embeddings_path, ids_path


def test_build_index_persists_index_and_mapping(tmp_path):
    embeddings, ids_path_input = (
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        ["a", "b", "c"],
    )
    embeddings_path, ids_path = _write_embeddings(tmp_path, embeddings, ids_path_input)
    index_path = tmp_path / "index" / "products.faiss"
    mapping_path = tmp_path / "index" / "product_ids.json"

    count = build_index(embeddings_path, ids_path, index_path, mapping_path)

    assert count == 3
    assert index_path.exists()
    assert mapping_path.exists()

    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert payload["product_ids"] == ["a", "b", "c"]
    assert payload["index_type"] == "IndexFlatIP"
    assert payload["dimension"] == 2


def test_build_index_rejects_mismatched_counts(tmp_path):
    embeddings_path, ids_path = _write_embeddings(
        tmp_path, [[1.0, 0.0], [0.0, 1.0]], ["only-one"]
    )
    with pytest.raises(ValueError):
        build_index(embeddings_path, ids_path)


def test_build_index_rejects_empty_embeddings(tmp_path):
    embeddings_path, ids_path = _write_embeddings(tmp_path, np.empty((0, 4)), [])
    with pytest.raises(ValueError):
        build_index(embeddings_path, ids_path)


def test_build_index_rejects_duplicate_product_ids(tmp_path):
    embeddings_path, ids_path = _write_embeddings(
        tmp_path, [[1.0, 0.0], [0.0, 1.0]], ["dup", "dup"]
    )
    with pytest.raises(ValueError):
        build_index(embeddings_path, ids_path)


def test_build_index_rejects_zero_vectors(tmp_path):
    embeddings_path, ids_path = _write_embeddings(
        tmp_path, [[0.0, 0.0], [1.0, 0.0]], ["zero", "unit"]
    )
    with pytest.raises(ValueError):
        build_index(embeddings_path, ids_path)


def test_search_index_returns_nearest_neighbor_first(tmp_path):
    embeddings_path, ids_path = _write_embeddings(
        tmp_path,
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
        ["orthogonal-x", "orthogonal-y", "near-x"],
    )
    index_path = tmp_path / "index" / "products.faiss"
    mapping_path = tmp_path / "index" / "product_ids.json"
    build_index(embeddings_path, ids_path, index_path, mapping_path)

    results = search_index(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=3,
        index_path=index_path,
        id_mapping_path=mapping_path,
    )

    assert [product_id for product_id, _ in results][0] == "orthogonal-x"
    assert len(results) == 3
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_search_index_caps_results_at_index_size(tmp_path):
    embeddings_path, ids_path = _write_embeddings(
        tmp_path, [[1.0, 0.0], [0.0, 1.0]], ["a", "b"]
    )
    index_path = tmp_path / "index" / "products.faiss"
    mapping_path = tmp_path / "index" / "product_ids.json"
    build_index(embeddings_path, ids_path, index_path, mapping_path)

    results = search_index(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=50,
        index_path=index_path,
        id_mapping_path=mapping_path,
    )

    assert len(results) == 2


def test_search_index_rejects_invalid_top_k(tmp_path):
    embeddings_path, ids_path = _write_embeddings(
        tmp_path, [[1.0, 0.0]], ["a"]
    )
    index_path = tmp_path / "index" / "products.faiss"
    mapping_path = tmp_path / "index" / "product_ids.json"
    build_index(embeddings_path, ids_path, index_path, mapping_path)

    with pytest.raises(ValueError):
        search_index(
            np.array([1.0, 0.0], dtype=np.float32),
            top_k=0,
            index_path=index_path,
            id_mapping_path=mapping_path,
        )


def test_search_index_rejects_wrong_dimension_query(tmp_path):
    embeddings_path, ids_path = _write_embeddings(
        tmp_path, [[1.0, 0.0], [0.0, 1.0]], ["a", "b"]
    )
    index_path = tmp_path / "index" / "products.faiss"
    mapping_path = tmp_path / "index" / "product_ids.json"
    build_index(embeddings_path, ids_path, index_path, mapping_path)

    # A dimension mismatch means the deployment is misconfigured (embedding
    # model doesn't match the index), not that the caller sent bad input --
    # RuntimeError, not ValueError, so the API layer maps it to 503.
    with pytest.raises(RuntimeError):
        search_index(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            top_k=1,
            index_path=index_path,
            id_mapping_path=mapping_path,
        )


def test_search_index_detects_out_of_sync_mapping(tmp_path):
    embeddings_path, ids_path = _write_embeddings(
        tmp_path, [[1.0, 0.0], [0.0, 1.0]], ["a", "b"]
    )
    index_path = tmp_path / "index" / "products.faiss"
    mapping_path = tmp_path / "index" / "product_ids.json"
    build_index(embeddings_path, ids_path, index_path, mapping_path)

    # Corrupt the mapping so it no longer matches the persisted index.
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    payload["product_ids"] = ["a"]
    mapping_path.write_text(json.dumps(payload), encoding="utf-8")

    # An out-of-sync index/mapping is a broken deployment, not a bad
    # request -- RuntimeError, not ValueError, so the API layer maps it to
    # 503 (spec section 15: "corrupt index" -> 503).
    with pytest.raises(RuntimeError):
        search_index(
            np.array([1.0, 0.0], dtype=np.float32),
            top_k=1,
            index_path=index_path,
            id_mapping_path=mapping_path,
        )
