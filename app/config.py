"""
Centralised application configuration using pydantic-settings.

All settings resolve in priority order:
    1. Explicit constructor args (for testing)
    2. Environment variables (set in .env or Streamlit Cloud Secrets)
    3. Default values defined here

Usage:
        from app.config import get_settings
        cfg = get_settings()
        print(cfg.google_api_key)
        print(cfg.max_claims)
"""

from __future__ import annotations

import functools
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings, loaded from environment / .env file.
    All fields have safe defaults so the app starts without any env vars set.
    """

    model_config = SettingsConfigDict(
        env_file         = ".env",
        env_file_encoding= "utf-8",
        case_sensitive   = False,
        extra            = "ignore",
    )

    # ── API Keys (never have defaults — must come from env/user) ─────────────
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key — get one at https://aistudio.google.com/app/apikey",
    )
    tavily_api_key: Optional[str] = Field(
        default=None,
        description="Tavily Search API key — get one at https://app.tavily.com",
    )

    # ── PDF Processing ────────────────────────────────────────────────────────
    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum PDF upload size in megabytes",
    )
    max_pdf_pages: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of PDF pages to process",
    )

    # ── Claim Extraction ──────────────────────────────────────────────────────
    max_claims: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum number of claims to extract per document",
    )
    gemini_extract_model: str = Field(
        default="gemini-2.0-flash-lite",
        description="Gemini model used for claim extraction (cheaper/faster)",
    )
    gemini_verify_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model used for fact verification (stronger reasoning)",
    )

    # ── Search ────────────────────────────────────────────────────────────────
    search_results_per_claim: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of web search results to retrieve per claim",
    )

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    inter_claim_delay_secs: float = Field(
        default=4.5,
        ge=0.0,
        le=60.0,
        description="Seconds to wait between claim verifications (Gemini RPM guard)",
    )
    llm_timeout_secs: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout in seconds for each LLM API call",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on transient API failures",
    )

    # ── Caching ───────────────────────────────────────────────────────────────
    cache_ttl_secs: int = Field(
        default=3600,
        ge=0,
        description="Streamlit cache TTL in seconds (0 = no expiry)",
    )

    # ── Security ──────────────────────────────────────────────────────────────
    max_text_chars_per_chunk: int = Field(
        default=3000,
        ge=500,
        le=10000,
        description="Maximum characters per LLM prompt chunk",
    )
    prompt_injection_guard: bool = Field(
        default=True,
        description="Enable prompt injection sanitization on extracted PDF text",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Log level: DEBUG | INFO | WARNING | ERROR",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("google_api_key", "tavily_api_key", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        """Strip accidental whitespace from API keys. Empty string → None."""
        if v is not None and isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return upper

    @field_validator("gemini_extract_model", "gemini_verify_model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        known_prefixes = ("gemini-", "models/gemini-")
        if not any(v.startswith(p) for p in known_prefixes):
            raise ValueError(
                f"Model name {v!r} doesn't look like a Gemini model. "
                f"Expected it to start with 'gemini-'."
            )
        return v

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def has_google_key(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_tavily_key(self) -> bool:
        return bool(self.tavily_api_key)

    def safe_dict(self) -> dict:
        """Return settings dict with API keys redacted — safe for logging."""
        d = self.model_dump()
        for key in ("google_api_key", "tavily_api_key"):
            if d.get(key):
                d[key] = d[key][:8] + "…[REDACTED]"
        return d


# ── Module-level cached singleton ─────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the global Settings singleton (loaded once, cached forever).

    In tests, call `get_settings.cache_clear()` before patching env vars.
    """
    return Settings()
