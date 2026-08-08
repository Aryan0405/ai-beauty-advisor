"""Health-check endpoint."""

from fastapi import APIRouter

from backend.app.schemas import HealthResponse
from backend.app.services import health_service


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report process availability plus index and database readiness."""
    return HealthResponse(
        status="ok",
        index_loaded=health_service.index_loaded(),
        db_connected=health_service.db_connected(),
    )
