"""
Verdict model — represents the outcome of verifying a single claim.

Verdict values follow a clear semantic:
  VERIFIED      → evidence confirms the claim
  INACCURATE    → claim is partially wrong / misleading
  FALSE         → evidence directly contradicts the claim
  OUTDATED      → claim was once true but is no longer current
  UNVERIFIABLE  → insufficient or no reliable evidence found
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


# ── Verdict Enum ─────────────────────────────────────────────────────────────

class VerdictLabel(str, Enum):
    VERIFIED     = "Verified"
    INACCURATE   = "Inaccurate"
    FALSE        = "False"
    OUTDATED     = "Outdated"
    UNVERIFIABLE = "Unverifiable"


# ── Emoji + Colour mappings (used by UI layer) ────────────────────────────────

VERDICT_EMOJI: dict[VerdictLabel, str] = {
    VerdictLabel.VERIFIED:     "✅",
    VerdictLabel.INACCURATE:   "⚠️",
    VerdictLabel.FALSE:        "❌",
    VerdictLabel.OUTDATED:     "🕐",
    VerdictLabel.UNVERIFIABLE: "❓",
}

VERDICT_COLOR: dict[VerdictLabel, str] = {
    VerdictLabel.VERIFIED:     "#22c55e",   # green
    VerdictLabel.INACCURATE:   "#f59e0b",   # amber
    VerdictLabel.FALSE:        "#ef4444",   # red
    VerdictLabel.OUTDATED:     "#a78bfa",   # purple
    VerdictLabel.UNVERIFIABLE: "#6b7280",   # grey
}


# ── Evidence Source ───────────────────────────────────────────────────────────

class EvidenceSource(BaseModel):
    """A single web source retrieved during fact-checking."""

    url:     str        = Field(..., description="URL of the source page")
    title:   str        = Field(..., description="Page or article title")
    snippet: str        = Field(..., description="Relevant excerpt from the source")
    date:    Optional[str] = Field(
        default=None,
        description="Publication or last-updated date (ISO-8601 if available)"
    )

    @field_validator("url")
    @classmethod
    def url_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("URL must not be empty")
        return v.strip()

    @field_validator("snippet")
    @classmethod
    def truncate_snippet(cls, v: str) -> str:
        """Guard against excessively long snippets filling the UI."""
        max_chars = 500
        return v[:max_chars] + "…" if len(v) > max_chars else v


# ── Verdict ───────────────────────────────────────────────────────────────────

class Verdict(BaseModel):
    """The complete verdict for one claim after web search + LLM reasoning."""

    label:       VerdictLabel        = Field(..., description="Verdict classification")
    confidence:  float               = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in verdict, 0.0 – 1.0"
    )
    explanation: str                 = Field(
        ...,
        description="Plain-English reasoning for the verdict"
    )
    sources:     list[EvidenceSource] = Field(
        default_factory=list,
        description="Web sources used to reach the verdict"
    )

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def emoji(self) -> str:
        return VERDICT_EMOJI[self.label]

    @property
    def color(self) -> str:
        return VERDICT_COLOR[self.label]

    @property
    def confidence_pct(self) -> str:
        return f"{self.confidence * 100:.0f}%"

    @property
    def display_label(self) -> str:
        return f"{self.emoji} {self.label.value}"

    def to_row_dict(self) -> dict:
        """Flat dict suitable for a Pandas DataFrame row."""
        return {
            "verdict":     self.label.value,
            "confidence":  self.confidence_pct,
            "explanation": self.explanation,
            "sources":     " | ".join(s.url for s in self.sources),
        }
