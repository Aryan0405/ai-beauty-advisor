"""Global exception handlers producing the spec's structured error envelope.

Every error response body has the shape ``{"error": {"code", "message"}}``
(spec section 15), regardless of whether it originated as a raised
HTTPException, a Pydantic request-validation failure, or an unhandled
exception.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


LOGGER = logging.getLogger(__name__)

_CODES_BY_STATUS = {
    status.HTTP_400_BAD_REQUEST: "invalid_request",
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_503_SERVICE_UNAVAILABLE: "service_unavailable",
}


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


async def _handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    code = _CODES_BY_STATUS.get(exc.status_code, "http_error")
    return _error_response(exc.status_code, code, str(exc.detail))


async def _handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Spec section 15 wants validation errors (empty query, invalid top_k)
    # as 400 with field-level detail, not FastAPI's default 422.
    errors = exc.errors()
    first = errors[0] if errors else {}
    field = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    detail = first.get("msg", "Invalid request.")
    message = f"{field}: {detail}" if field else detail
    return _error_response(status.HTTP_400_BAD_REQUEST, "invalid_request", message)


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception(
        "Unhandled exception while processing %s %s", request.method, request.url.path
    )
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "An unexpected error occurred.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the structured-error handlers to ``app`` (spec section 15)."""
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(Exception, _handle_unexpected_error)
