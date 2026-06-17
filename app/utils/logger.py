"""
Structured logger — wraps structlog for consistent, readable log output.

Usage:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Extraction complete", pages=12, chars=4500)
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def _configure_logging() -> None:
    """One-time logging setup. Safe to call multiple times."""
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level     = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        format  = "%(message)s",
        stream  = sys.stdout,
        level   = log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class    = structlog.make_filtering_bound_logger(log_level),
        context_class    = dict,
        logger_factory   = structlog.PrintLoggerFactory(),
        cache_logger_on_first_use = True,
    )


_configure_logging()


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a bound structlog logger for the given module name.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        A structlog BoundLogger instance.
    """
    return structlog.get_logger(name)
