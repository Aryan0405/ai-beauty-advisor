"""Load the cosmetics CSV into the local SQLite products table.

Run from the repository root with:
    python -m backend.app.db.ingest
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "cosmetics.csv"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "beauty_advisor.db"
REQUIRED_COLUMNS = {
    "Label",
    "Brand",
    "Name",
    "Price",
    "Rank",
    "Ingredients",
    "Combination",
    "Dry",
    "Normal",
    "Oily",
    "Sensitive",
}
SKIN_TYPE_COLUMNS = ("Combination", "Dry", "Normal", "Oily", "Sensitive")


def _clean_text(value: object) -> str:
    """Strip and normalize whitespace in a source text field."""
    return " ".join(str(value).strip().split())


def _create_schema(connection: sqlite3.Connection) -> None:
    """Create a fresh products table and its filter indexes."""
    connection.executescript(
        """
        DROP TABLE IF EXISTS products;

        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            brand TEXT NOT NULL,
            category TEXT NOT NULL,
            skin_type TEXT NOT NULL,
            ingredients TEXT NOT NULL,
            description TEXT NOT NULL,
            price REAL,
            source_rank REAL NOT NULL,
            embedding_id INTEGER
        );

        CREATE INDEX idx_products_category ON products(category);
        CREATE INDEX idx_products_skin_type ON products(skin_type);
        """
    )


def ingest_data(csv_path: Path, database_path: Path) -> int:
    """Clean the source CSV, rebuild ``products``, and return inserted rows."""
    dataframe = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing}")

    dataframe = dataframe.dropna(subset=["Label", "Brand", "Name", "Ingredients"])
    records: list[tuple[object, ...]] = []

    for source_index, row in dataframe.iterrows():
        category = _clean_text(row["Label"])
        brand = _clean_text(row["Brand"])
        name = _clean_text(row["Name"])
        ingredients = _clean_text(row["Ingredients"])
        if not all((category, brand, name, ingredients)):
            continue

        skin_types = [
            skin_type
            for skin_type in SKIN_TYPE_COLUMNS
            if int(row[skin_type]) == 1
        ]
        skin_type = ",".join(skin_types)
        description = (
            f"Name: {name}. Brand: {brand}. Category: {category}. "
            f"Skin types: {skin_type}. Ingredients: {ingredients}"
        )
        records.append(
            (
                f"cosmetics-{source_index + 1:06d}",
                name,
                brand,
                category,
                skin_type,
                ingredients,
                description,
                float(row["Price"]),
                float(row["Rank"]),
                None,
            )
        )

    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        _create_schema(connection)
        connection.executemany(
            """
            INSERT INTO products (
                product_id, name, brand, category, skin_type, ingredients,
                description, price, source_rank, embedding_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

    return len(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest cosmetics CSV into SQLite.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DATABASE_PATH)
    return parser.parse_args()


def main() -> None:
    """Execute the ingestion command-line interface."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    inserted_rows = ingest_data(args.csv_path, args.db_path)
    LOGGER.info("Inserted %s products into %s", inserted_rows, args.db_path)


if __name__ == "__main__":
    main()
