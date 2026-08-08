"""Health-check endpoint."""

from fastapi import APIRouter

from backend.app.core.config import get_settings
from backend.app.db.session import session_scope
from backend.app.schemas import HealthResponse


router = APIRouter()


def _index_loaded() -> bool:
    """Check that the configured FAISS index file exists on disk."""
    return get_settings().faiss_index_path.exists()


def _db_connected() -> bool:
    """Check that the configured SQLite database is reachable."""
    try:
        with session_scope() as db:
            db.execute("SELECT 1")
        return True
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report process availability plus index and database readiness."""
    return HealthResponse(
        status="ok",
        index_loaded=_index_loaded(),
        db_connected=_db_connected(),
    )
