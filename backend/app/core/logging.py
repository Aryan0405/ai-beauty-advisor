"""Structured JSON logging with per-request correlation IDs (spec section 16).

Every log line is one JSON object. The current request's correlation ID
(assigned by ``RequestIDMiddleware``) is attached to every log line emitted
while that request is being handled, via a ``contextvars.ContextVar`` --
this works correctly across FastAPI's sync-endpoint threadpool because
``anyio.to_thread.run_sync`` (which FastAPI uses to run sync endpoints)
explicitly copies the current context into the worker thread.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any


_request_id_ctx_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# LLM prompts/responses are sensitive/verbose and only meant for local
# debugging -- always DEBUG, disabled by default (spec section 16).
PROMPT_LOG_LEVEL = logging.DEBUG


def set_request_id(request_id: str | None) -> Token:
    """Bind the correlation ID for the current request context.

    Returns a token; pass it to ``reset_request_id`` when the request ends.
    """
    return _request_id_ctx_var.set(request_id)


def reset_request_id(token: Token) -> None:
    """Undo a prior ``set_request_id`` call."""
    _request_id_ctx_var.reset(token)


def get_request_id() -> str | None:
    """Return the correlation ID bound to the current request, if any."""
    return _request_id_ctx_var.get()


class _RequestIDFilter(logging.Filter):
    """Attach the current request's correlation ID to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


class _JSONFormatter(logging.Formatter):
    """Render one log record as a single JSON line."""

    _RESERVED = frozenset(
        logging.LogRecord(
            name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
        ).__dict__
    ) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        # Any extra=... fields passed to the log call ride along verbatim.
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter and request-ID filter on the root logger.

    Idempotent: safe to call more than once (e.g. once from application
    startup, once from a test) without duplicating handlers.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if any(isinstance(handler, logging.StreamHandler) and
           isinstance(getattr(handler, "formatter", None), _JSONFormatter)
           for handler in root_logger.handlers):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    handler.addFilter(_RequestIDFilter())
    root_logger.handlers = [handler]
