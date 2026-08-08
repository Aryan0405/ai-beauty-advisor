"""Business logic for single-product lookups."""

from __future__ import annotations

from backend.app.db.repository import get_product_by_id
from backend.app.db.session import session_scope
from backend.app.models import Product


def get_product(product_id: str) -> Product | None:
    """Return one product by its stable source-derived ID, or None."""
    with session_scope() as db:
        return get_product_by_id(db, product_id)
