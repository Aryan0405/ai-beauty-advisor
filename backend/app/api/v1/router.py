"""Versioned API router assembly."""

from fastapi import APIRouter

from .endpoints import health, products, recommendations


router = APIRouter(prefix="/api/v1")
router.include_router(health.router)
router.include_router(products.router)
router.include_router(recommendations.router)
