"""FastAPI application entry point."""

import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from backend.app.api.v1.router import router as api_v1_router
from backend.app.core.config import get_settings
from backend.app.core.error_handlers import register_exception_handlers
from backend.app.core.logging import configure_logging
from backend.app.core.middleware import RequestIDMiddleware


try:
    settings = get_settings()
except ValidationError as error:
    # Fail fast with one clear, actionable line instead of a raw pydantic
    # traceback -- this is what an operator actually sees in container logs
    # when a required variable (e.g. GEMINI_API_KEY) is missing.
    missing_vars = ", ".join(
        str(err["loc"][0]).upper() for err in error.errors() if err["type"] == "missing"
    )
    if missing_vars:
        sys.stderr.write(
            f"FATAL: missing required environment variable(s): {missing_vars}. "
            "Copy .env.example to .env and fill in real values, or set them "
            "in your deployment environment before starting the service.\n"
        )
    else:
        sys.stderr.write(f"FATAL: invalid configuration: {error}\n")
    raise SystemExit(1) from error

configure_logging(settings.log_level)

app = FastAPI(title="AI Beauty Advisor")
register_exception_handlers(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Registered after CORSMiddleware so it runs closer to the router (Starlette
# applies user_middleware in reverse-of-registration order around the
# request) -- request_started/finished should bracket everything downstream.
app.add_middleware(RequestIDMiddleware)
app.include_router(api_v1_router)
