"""
Claim model — represents a single verifiable factual claim extracted from a PDF.

A claim is the atomic unit of fact-checking. Every downstream step
(search, verify, display) operates on a list of Claim objects.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.models.verdict import Verdict


# ── Claim Category ────────────────────────────────────────────────────────────

class ClaimCategory(str, Enum):
    """Semantic type of a claim — helps tailor search queries."""
    STATISTICAL  = "Statistical"    # Numbers, percentages, counts
    HISTORICAL   = "Historical"     # Dates, events, past facts
    SCIENTIFIC   = "Scientific"     # Research findings, studies
    ECONOMIC     = "Economic"       # GDP, inflation, financial data
    POLITICAL    = "Political"      # Policies, elections, legislation
    GEOGRAPHICAL = "Geographical"   # Locations, populations, areas
    OTHER        = "Other"          # Anything else verifiable


# ── Claim ─────────────────────────────────────────────────────────────────────

class Claim(BaseModel):
    """A single verifiable factual claim extracted from a PDF document."""

    # ── Extraction fields ──────────────────────────────────────────────────
    id:          int             = Field(..., description="1-based index within document")
    text:        str             = Field(..., description="Verbatim claim text as extracted")
    category:    ClaimCategory   = Field(..., description="Semantic type of claim")
    page_number: int             = Field(..., ge=1, description="Source page in PDF (1-based)")
    context:     Optional[str]   = Field(
        default=None,
        description="Surrounding sentence(s) providing context for the claim"
    )

    # ── Verification fields (populated after web search) ──────────────────
    verdict:     Optional[Verdict] = Field(
        default=None,
        description="Populated after fact-checking; None means not yet checked"
    )
    search_query: Optional[str] = Field(
        default=None,
        description="The query sent to the search engine for this claim"
    )

    # ── Validators ────────────────────────────────────────────────────────
    @field_validator("text")
    @classmethod
    def text_must_be_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Claim text must not be empty")
        return v

    @field_validator("text")
    @classmethod
    def text_minimum_length(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("Claim text is too short to be verifiable")
        return v

    # ── Convenience properties ─────────────────────────────────────────────
    @property
    def is_verified(self) -> bool:
        return self.verdict is not None

    @property
    def short_text(self) -> str:
        """Truncated claim for display in narrow columns."""
        max_len = 120
        return self.text[:max_len] + "…" if len(self.text) > max_len else self.text

    def to_row_dict(self) -> dict:
        """
        Flat dict for a Pandas DataFrame row.
        Merges claim fields with verdict fields (if available).
        """
        base = {
            "id":          self.id,
            "claim":       self.short_text,
            "category":    self.category.value,
            "page":        self.page_number,
        }
        if self.verdict:
            base.update(self.verdict.to_row_dict())
        else:
            base.update({
                "verdict":     "Pending",
                "confidence":  "—",
                "explanation": "—",
                "sources":     "—",
            })
        return base

    def __repr__(self) -> str:
        status = self.verdict.label.value if self.verdict else "Unchecked"
        return f"<Claim #{self.id} [{self.category.value}] [{status}]: {self.short_text!r}>"


# ── Batch Result ──────────────────────────────────────────────────────────────

class FactCheckResult(BaseModel):
    """Top-level result object for an entire PDF fact-check session."""

    filename:     str          = Field(..., description="Original PDF filename")
    total_pages:  int          = Field(..., ge=1, description="Total pages in PDF")
    total_claims: int          = Field(..., ge=0, description="Number of claims extracted")
    claims:       list[Claim]  = Field(default_factory=list)

    @property
    def checked_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.is_verified]

    @property
    def verdict_summary(self) -> dict[str, int]:
        """Count of each verdict label across all checked claims."""
        from collections import Counter
        counts = Counter(
            c.verdict.label.value
            for c in self.checked_claims
            if c.verdict
        )
        return dict(counts)

    def to_dataframe(self):
        """Convert all claims to a Pandas DataFrame."""
        import pandas as pd
        if not self.claims:
            return pd.DataFrame()
        return pd.DataFrame([c.to_row_dict() for c in self.claims])
