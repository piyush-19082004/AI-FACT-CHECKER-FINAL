"""Models package — Pydantic data models for claims, verdicts, and evidence."""

from app.models.verdict import (
    VerdictLabel,
    Verdict,
    EvidenceSource,
    VERDICT_EMOJI,
    VERDICT_COLOR,
)
from app.models.claim import (
    ClaimCategory,
    Claim,
    FactCheckResult,
)

__all__ = [
    "VerdictLabel",
    "Verdict",
    "EvidenceSource",
    "VERDICT_EMOJI",
    "VERDICT_COLOR",
    "ClaimCategory",
    "Claim",
    "FactCheckResult",
]
