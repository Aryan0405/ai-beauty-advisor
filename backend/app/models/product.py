"""Data structure representing a stored beauty product."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Product:
    """A row from the SQLite ``products`` table."""

    product_id: str
    name: str
    brand: str
    category: str
    skin_type: str
    ingredients: str
    description: str
    price: float | None
    source_rank: float
    embedding_id: int | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Product":
        """Build a product from a connection configured with ``sqlite3.Row``."""
        return cls(
            product_id=row["product_id"],
            name=row["name"],
            brand=row["brand"],
            category=row["category"],
            skin_type=row["skin_type"],
            ingredients=row["ingredients"],
            description=row["description"],
            price=row["price"],
            source_rank=row["source_rank"],
            embedding_id=row["embedding_id"],
        )
