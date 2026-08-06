"""Health-check endpoint."""

from fastapi import APIRouter

from backend.app.schemas import HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Confirm that the API process is available."""
    return HealthResponse(status="ok")
