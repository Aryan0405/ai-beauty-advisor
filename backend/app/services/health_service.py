"""Business logic backing the health-check endpoint.

Deliberately thin: a health check's whole purpose is reporting infra
connectivity, so this mostly delegates straight to db/vectorstore. It still
lives here rather than in the API layer so the endpoint itself stays a
thin router with no direct db/vectorstore imports, consistent with the
rest of the codebase's api -> services -> (db | vectorstore | llm) layering.
"""

from __future__ import annotations

from backend.app.core.config import get_settings
from backend.app.db.session import session_scope


def index_loaded() -> bool:
    """Check that the configured FAISS index file exists on disk."""
    return get_settings().faiss_index_path.exists()


def db_connected() -> bool:
    """Check that the configured SQLite database is reachable."""
    try:
        with session_scope() as db:
            db.execute("SELECT 1")
        return True
    except Exception:
        return False
