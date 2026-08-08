"""
Structured logging configuration for clinical-rag-eval-engine.

Provides JSON-formatted logging that is both human-readable during development
and machine-parseable for production observability. All modules should use:

    from clinical_rag.ops.logging import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Format log records as single-line JSON objects.

    Each log line is a self-contained JSON object with standardized fields,
    making logs easy to parse, filter, and aggregate.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include any extra fields passed via logger.info("msg", extra={...})
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__ and key not in (
                "message",
                "msg",
            ):
                try:
                    json.dumps(value)  # only include JSON-serializable extras
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        return json.dumps(log_entry)


class SimpleFormatter(logging.Formatter):
    """Human-readable formatter for development use."""

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT = "%H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)


def setup_logging(
    level: str = "INFO",
    log_format: str = "simple",
) -> None:
    """
    Configure the root logger for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Either 'json' for structured production logging or
                    'simple' for human-readable development logging.
    """
    root_logger = logging.getLogger("clinical_rag")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicate output on re-init
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(SimpleFormatter())

    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ["httpx", "httpcore", "chromadb", "openai"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Patched:
    When a module is run directly (`python -m clinical_rag.ingest.indexer`),
    Python sets its __name__ to "__main__". A logger by that name sits outside
    the "clinical_rag" tree and therefore never inherits the handler installed
    by setup_logging(), so its records are silently dropped. We remap it so
    entrypoint modules log identically whether imported or run as scripts.

    Usage:
        from clinical_rag.ops.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Loaded %d chunks", chunk_count)
    """
    if name == "__main__":
        name = "clinical_rag.__main__"
    return logging.getLogger(name)
