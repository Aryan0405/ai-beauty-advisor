"""Reusable, parameterized data-access queries for products."""

from __future__ import annotations

import sqlite3

from ..models import Product


def get_product_by_id(
    db: sqlite3.Connection, product_id: str
) -> Product | None:
    """Return one product by its stable ID, if it exists."""
    row = db.execute(
        "SELECT * FROM products WHERE product_id = ?", (product_id,)
    ).fetchone()
    return Product.from_row(row) if row is not None else None


def get_all_products(
    db: sqlite3.Connection, limit: int = 100, offset: int = 0
) -> list[Product]:
    """Return a page of products in stable ID order."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if offset < 0:
        raise ValueError("offset cannot be negative")

    rows = db.execute(
        "SELECT * FROM products ORDER BY product_id LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [Product.from_row(row) for row in rows]


def get_products_by_category(
    db: sqlite3.Connection, category: str
) -> list[Product]:
    """Return products in a category using a case-insensitive exact match."""
    rows = db.execute(
        """
        SELECT * FROM products
        WHERE category = ? COLLATE NOCASE
        ORDER BY product_id
        """,
        (category.strip(),),
    ).fetchall()
    return [Product.from_row(row) for row in rows]
