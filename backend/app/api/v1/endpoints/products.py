"""Product lookup endpoint."""

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas import ProductResponse
from backend.app.services.product_service import get_product


router = APIRouter()


@router.get("/products/{product_id}", response_model=ProductResponse)
def read_product(product_id: str) -> ProductResponse:
    """Fetch a single product by its stable source-derived ID."""
    product = get_product(product_id)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found.",
        )
    return ProductResponse(**asdict(product))
