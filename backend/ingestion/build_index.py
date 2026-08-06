"""Build the persisted FAISS index from generated product embeddings.

Run from the repository root with:
    python -m backend.ingestion.build_index
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from backend.app.db.session import DEFAULT_DATABASE_PATH, session_scope
from backend.app.vectorstore.faiss_index import (
    DEFAULT_ID_MAPPING_PATH,
    DEFAULT_INDEX_PATH,
    build_index,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "product_embeddings.npy"
DEFAULT_PRODUCT_IDS_PATH = PROJECT_ROOT / "data" / "processed" / "product_ids.npy"


def sync_embedding_ids(database_path: str | Path, product_ids_path: str | Path) -> None:
    """Synchronize SQLite ``embedding_id`` values with FAISS vector positions."""
    product_ids = np.load(product_ids_path).astype(str).tolist()
    with session_scope(database_path) as connection:
        connection.execute("UPDATE products SET embedding_id = NULL")
        connection.executemany(
            "UPDATE products SET embedding_id = ? WHERE product_id = ?",
            enumerate(product_ids),
        )
        updated_count = connection.execute(
            "SELECT COUNT(*) FROM products WHERE embedding_id IS NOT NULL"
        ).fetchone()[0]
    if updated_count != len(product_ids):
        raise ValueError("SQLite products and the FAISS ID mapping are out of sync.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the persistent FAISS index.")
    parser.add_argument("--embeddings-path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--product-ids-path", type=Path, default=DEFAULT_PRODUCT_IDS_PATH)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--id-mapping-path", type=Path, default=DEFAULT_ID_MAPPING_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DATABASE_PATH)
    return parser.parse_args()


def main() -> None:
    """Execute the index-build command-line interface."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    indexed_count = build_index(
        embeddings_path=args.embeddings_path,
        product_ids_path=args.product_ids_path,
        index_path=args.index_path,
        id_mapping_path=args.id_mapping_path,
    )
    sync_embedding_ids(args.db_path, args.product_ids_path)
    LOGGER.info("Built FAISS index with %s product embeddings", indexed_count)


if __name__ == "__main__":
    main()
