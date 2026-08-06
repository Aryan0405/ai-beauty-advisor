"""Product lookup endpoint."""

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, status

from backend.app.db.repository import get_product_by_id
from backend.app.db.session import session_scope
from backend.app.schemas import ProductResponse


router = APIRouter()


@router.get("/products/{product_id}", response_model=ProductResponse)
def get_product(product_id: str) -> ProductResponse:
    """Fetch a single product by its stable source-derived ID."""
    with session_scope() as db:
        product = get_product_by_id(db, product_id)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found.",
        )
    return ProductResponse(**asdict(product))
