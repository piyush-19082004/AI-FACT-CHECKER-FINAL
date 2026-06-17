"""
Rate Limiter & Timeout Guards — wraps API calls with:
  • Exponential backoff retry on transient failures
  • Per-call timeout enforcement (prevents hanging on slow APIs)
  • Token-bucket style RPM throttling

Usage:
    from app.utils.rate_limiter import retry_with_backoff, with_timeout

    @retry_with_backoff
    @with_timeout(seconds=30)
    def call_gemini():
        return llm.invoke(prompt)
"""

from __future__ import annotations

import random
import signal
import threading
import time
from functools import wraps
from typing import Callable, TypeVar, Any, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    before_sleep_log,
)

from app.utils.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ── Retryable exception types ─────────────────────────────────────────────────

_RETRYABLE_EXCEPTIONS: tuple = (
    ConnectionError,
    TimeoutError,
    OSError,
)

try:
    import google.api_core.exceptions as _gexc
    _RETRYABLE_EXCEPTIONS = _RETRYABLE_EXCEPTIONS + (
        _gexc.ResourceExhausted,    # 429 — quota exceeded
        _gexc.ServiceUnavailable,   # 503 — Gemini down
        _gexc.InternalServerError,  # 500 — transient
        _gexc.DeadlineExceeded,     # Timeout at API level
    )
except ImportError:
    pass


# ── Retry with exponential backoff ────────────────────────────────────────────

def retry_with_backoff(func: F) -> F:
    """
    Decorator: retry up to 3 times with exponential backoff + jitter.

    Wait schedule:  2s → 4s → 8s  (+ random 0–1s jitter)
    Max wait:       30s per attempt
    Retries on:     quota (429), network errors, timeouts
    Gives up after: 3 retries (4 total attempts)

    Note: The original exception is re-raised after all retries are exhausted.
    """
    @retry(
        retry        = retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        wait         = wait_exponential(multiplier=2, min=2, max=30) + wait_random(0, 1),
        stop         = stop_after_attempt(4),
        reraise      = True,
        before_sleep = before_sleep_log(logger, log_level=20),
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


# ── Timeout decorator ─────────────────────────────────────────────────────────

class CallTimeoutError(TimeoutError):
    """Raised when a decorated function exceeds its timeout."""


def with_timeout(seconds: int = 30):
    """
    Decorator factory: abort a function call if it exceeds `seconds`.

    Uses a background thread + threading.Event for cross-platform compatibility
    (signal.alarm only works on Unix; this approach works on Windows too).

    Args:
        seconds: Maximum time allowed for the function call.

    Usage:
        @with_timeout(seconds=30)
        def slow_api_call():
            ...

    Note: The decorated function runs in the main thread. The timer runs in a
    daemon thread. On timeout, CallTimeoutError is raised in the caller's scope
    via a threading mechanism. Works on Windows.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result    = [None]
            exc_store = [None]
            done      = threading.Event()

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exc_store[0] = e
                finally:
                    done.set()

            t = threading.Thread(target=target, daemon=True)
            t.start()
            finished = done.wait(timeout=seconds)

            if not finished:
                logger.error(
                    "API call timed out",
                    function=func.__name__,
                    timeout_secs=seconds,
                )
                raise CallTimeoutError(
                    f"'{func.__name__}' timed out after {seconds}s. "
                    "The API may be slow or unreachable. Please try again."
                )

            if exc_store[0] is not None:
                raise exc_store[0]

            return result[0]

        return wrapper  # type: ignore[return-value]

    return decorator


# ── RPM rate limiter ──────────────────────────────────────────────────────────

def with_rate_limit(calls_per_minute: int = 14):
    """
    Decorator factory: token-bucket RPM throttle.

    Adds a minimum interval between calls to stay within the specified RPM.
    Gemini free tier = 15 RPM → use 14 for safety margin.

    Usage:
        @with_rate_limit(calls_per_minute=14)
        def call_api():
            ...
    """
    min_interval = 60.0 / calls_per_minute

    def decorator(func: F) -> F:
        last_called: list[float] = [0.0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.monotonic() - last_called[0]
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed + random.uniform(0, 0.3)
                logger.debug(
                    "Rate limiting — sleeping before API call",
                    sleep_secs = round(sleep_time, 2),
                    function   = func.__name__,
                )
                time.sleep(sleep_time)
            last_called[0] = time.monotonic()
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ── Convenience: combined retry + timeout ─────────────────────────────────────

def retry_with_timeout(timeout_secs: int = 30, retries: int = 3):
    """
    Decorator factory combining retry_with_backoff + with_timeout.

    This is the recommended decorator for LLM API calls:

        @retry_with_timeout(timeout_secs=30, retries=3)
        def call_gemini():
            ...

    Order: timeout wraps the function first, then retry wraps the timeout.
    A timeout → CallTimeoutError (which IS retryable) → retry fires.
    """
    def decorator(func: F) -> F:
        # Apply timeout first (inner), then retry (outer)
        timed   = with_timeout(seconds=timeout_secs)(func)

        @retry(
            retry        = retry_if_exception_type(
                _RETRYABLE_EXCEPTIONS + (CallTimeoutError,)
            ),
            wait         = wait_exponential(multiplier=2, min=2, max=30) + wait_random(0, 1),
            stop         = stop_after_attempt(retries + 1),
            reraise      = True,
            before_sleep = before_sleep_log(logger, log_level=20),
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return timed(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
