"""Correlation-ID middleware (spec section 16).

Assigns a UUID4 request ID to every incoming request (or reuses an
upstream-supplied ``X-Request-ID``, so a request can be traced across
services), makes it available to every log line emitted while that
request is being handled (via ``core.logging``'s contextvar), echoes it
back in the response so a caller can correlate their own logs against
ours, and logs the request's start/end with total latency.
"""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from backend.app.core.logging import set_request_id


LOGGER = logging.getLogger("backend.app.request")
REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign/propagate a correlation ID and log each request's lifecycle."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        # Deliberately not reset at the end of this request: ServerErrorMiddleware
        # sits *outside* this middleware, so on an unhandled exception it still
        # needs to see this value after this method's frame unwinds in order to
        # attach the header to the 500 it builds (core/error_handlers.py). This
        # is safe without an explicit reset -- every request sets its own value
        # here before doing anything else, so there is nothing for a later
        # request to observe from an earlier one.
        set_request_id(request_id)
        start = time.perf_counter()

        LOGGER.info(
            "request_started",
            extra={"http_method": request.method, "http_path": request.url.path},
        )
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            LOGGER.exception(
                "request_failed",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise
        else:
            duration_ms = (time.perf_counter() - start) * 1000
            LOGGER.info(
                "request_finished",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
