"""Persistent FAISS index creation, loading, and similarity search helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "products.faiss"
DEFAULT_ID_MAPPING_PATH = PROJECT_ROOT / "data" / "index" / "product_ids.json"


def _faiss() -> Any:
    """Import FAISS only for operations that require the optional dependency."""
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError("faiss-cpu is required to build or search the index.") from error
    return faiss


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Return contiguous L2-normalized float32 vectors for cosine similarity."""
    normalized = np.ascontiguousarray(vectors, dtype=np.float32).copy()
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embeddings must not contain zero vectors.")
    normalized /= norms
    return normalized


def _load_product_ids(mapping_path: str | Path) -> list[str]:
    """Load the ordered mapping from FAISS positions to product IDs."""
    with Path(mapping_path).open(encoding="utf-8") as mapping_file:
        payload = json.load(mapping_file)
    product_ids = payload.get("product_ids")
    if not isinstance(product_ids, list) or not all(
        isinstance(product_id, str) for product_id in product_ids
    ):
        raise ValueError("Invalid FAISS product-ID mapping file.")
    return product_ids


def build_index(
    embeddings_path: str | Path,
    product_ids_path: str | Path,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    id_mapping_path: str | Path = DEFAULT_ID_MAPPING_PATH,
) -> int:
    """Build an ``IndexFlatIP`` index and persist its ordered product-ID mapping."""
    embeddings = np.load(embeddings_path)
    product_ids = np.load(product_ids_path).astype(str).tolist()
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty two-dimensional array.")
    if embeddings.shape[0] != len(product_ids):
        raise ValueError("Embedding and product-ID counts do not match.")
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("Product IDs must be unique.")

    vectors = _normalize(embeddings)
    faiss = _faiss()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    resolved_index_path = Path(index_path)
    resolved_mapping_path = Path(id_mapping_path)
    resolved_index_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(resolved_index_path))
    resolved_mapping_path.write_text(
        json.dumps(
            {
                "index_type": "IndexFlatIP",
                "dimension": vectors.shape[1],
                "product_ids": product_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return len(product_ids)


def search_index(
    query_embedding: np.ndarray,
    top_k: int,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    id_mapping_path: str | Path = DEFAULT_ID_MAPPING_PATH,
) -> list[tuple[str, float]]:
    """Return top matching product IDs and cosine-similarity scores."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    faiss = _faiss()
    index = faiss.read_index(str(index_path))
    product_ids = _load_product_ids(id_mapping_path)
    if index.ntotal != len(product_ids):
        raise ValueError("FAISS index and product-ID mapping are out of sync.")

    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape != (1, index.d):
        raise ValueError(f"Expected one query embedding with dimension {index.d}.")

    scores, vector_ids = index.search(_normalize(query), min(top_k, index.ntotal))
    return [
        (product_ids[vector_id], float(score))
        for score, vector_id in zip(scores[0], vector_ids[0], strict=True)
        if vector_id != -1
    ]
