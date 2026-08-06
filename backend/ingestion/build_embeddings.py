"""Generate and persist product embeddings for downstream FAISS indexing.

Run from the repository root with:
    python -m backend.ingestion.build_embeddings
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from backend.app.db.session import DEFAULT_DATABASE_PATH, session_scope


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel(Protocol):
    """Minimal interface used from a Sentence Transformers model."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        """Encode text strings into a two-dimensional NumPy array."""


def build_product_text(row: sqlite3.Row) -> str:
    """Create the rich, product-only text representation used for embedding."""
    return "\n".join(
        (
            f"Product: {row['name']}",
            f"Brand: {row['brand']}",
            f"Category: {row['category']}",
            f"Skin types: {row['skin_type']}",
            f"Description: {row['description']}",
            f"Ingredients: {row['ingredients']}",
        )
    )


def _load_model(model_name: str) -> EmbeddingModel:
    """Load the configured Sentence Transformers model only when needed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is required to generate embeddings."
        ) from error

    return SentenceTransformer(model_name)


@lru_cache
def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> EmbeddingModel:
    """Return a cached Sentence Transformers model for batch or query encoding."""
    return _load_model(model_name)


def embed_query(
    query: str,
    model_name: str = DEFAULT_MODEL_NAME,
    model: EmbeddingModel | None = None,
) -> np.ndarray:
    """Encode one user query with the same normalized embedding model."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query cannot be empty")

    encoder = model or get_embedding_model(model_name)
    embedding = np.asarray(
        encoder.encode(
            [normalized_query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError("The embedding model returned an invalid query embedding.")
    return embedding[0]


def _fetch_embedding_inputs(database_path: str | Path) -> tuple[list[str], list[str]]:
    """Read stable product IDs and their corresponding embedding text."""
    with session_scope(database_path) as connection:
        rows = connection.execute(
            """
            SELECT product_id, name, brand, category, skin_type, description, ingredients
            FROM products
            ORDER BY product_id
            """
        ).fetchall()

    product_ids = [row["product_id"] for row in rows]
    product_texts = [build_product_text(row) for row in rows]
    return product_ids, product_texts


def generate_embeddings(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    model: EmbeddingModel | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Embed all products and save aligned embedding and product-ID arrays."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    product_ids, product_texts = _fetch_embedding_inputs(database_path)
    encoder = model or get_embedding_model(model_name)
    embeddings = np.asarray(
        encoder.encode(
            product_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(product_ids):
        raise ValueError("The embedding model returned an invalid embedding matrix.")

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "product_embeddings.npy", embeddings)
    np.save(output_path / "product_ids.npy", np.asarray(product_ids, dtype=str))
    return embeddings, product_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate product embeddings.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Execute the embedding generation command-line interface."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    embeddings, product_ids = generate_embeddings(
        database_path=args.db_path,
        output_directory=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    LOGGER.info(
        "Saved %s embeddings with dimension %s to %s",
        len(product_ids),
        embeddings.shape[1],
        args.output_dir,
    )


if __name__ == "__main__":
    main()
