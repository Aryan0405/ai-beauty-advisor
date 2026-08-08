"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.router import router as api_v1_router
from backend.app.core.config import get_settings
from backend.app.core.error_handlers import register_exception_handlers


settings = get_settings()

app = FastAPI(title="AI Beauty Advisor")
register_exception_handlers(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_v1_router)
