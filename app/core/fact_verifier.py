"""
Fact Verifier — Stage 4 of the fact-checking pipeline.

Responsibilities:
  • Accept a Claim + its list of EvidenceSource objects
  • Build a structured reasoning prompt for Gemini 1.5 Pro
  • Parse the LLM's verdict into a typed Verdict object
  • Assign a confidence score based on evidence quality + agreement
  • Handle edge cases: no evidence, conflicting sources, outdated info

Design decisions:
  • Uses gemini-1.5-pro (not flash) — reasoning quality matters here
  • Temperature = 0.0 — verdicts must be deterministic and reproducible
  • JSON schema enforced via PydanticOutputParser + OutputFixingParser
  • 5-class verdict system with strict definitions in the prompt
  • Evidence formatted as numbered list for clear LLM citation
"""

from __future__ import annotations

import os
import re
import textwrap
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from app.models.claim import Claim
from app.models.verdict import Verdict, VerdictLabel, EvidenceSource
from app.utils.logger import get_logger
from app.utils.rate_limiter import retry_with_backoff

logger = get_logger(__name__)

# Allow tests to patch `ChatGoogleGenerativeAI` on this module even if the
# optional `langchain_google_genai` package is not present locally.
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:
    ChatGoogleGenerativeAI = None


# ── Intermediate schema for LLM output ───────────────────────────────────────

class RawVerdict(BaseModel):
    """Shape the LLM must output for a single verdict."""

    label: str = Field(
        ...,
        description=(
            "Verdict label. MUST be exactly one of: "
            "Verified, Inaccurate, False, Outdated, Unverifiable"
        )
    )
    confidence: float = Field(
        ...,
        description="Confidence in verdict from 0.0 (very uncertain) to 1.0 (very certain)",
        ge=0.0,
        le=1.0,
    )
    explanation: str = Field(
        ...,
        description=(
            "Plain-English explanation of the verdict in 2-4 sentences. "
            "Quote specific figures or source details where possible."
        )
    )
    cited_source_indices: list[int] = Field(
        default_factory=list,
        description="0-based indices of the evidence sources that informed this verdict",
    )


# ── Verification Prompt ───────────────────────────────────────────────────────

_VERIFY_PROMPT = PromptTemplate(
    input_variables=["claim", "category", "evidence_block", "format_instructions"],
    template=textwrap.dedent("""
        You are a professional fact-checker with expertise in verifying factual claims
        against web evidence. You are rigorous, impartial, and evidence-driven.

        CLAIM TO VERIFY:
        "{claim}"

        CLAIM CATEGORY: {category}

        WEB EVIDENCE RETRIEVED:
        {evidence_block}

        YOUR TASK:
        Analyse the claim against the evidence above and assign a verdict.

        VERDICT DEFINITIONS (choose EXACTLY one):
        ─────────────────────────────────────────
        • Verified      — The evidence CLEARLY and DIRECTLY supports the claim.
                          Multiple reliable sources agree. Numbers match.
        • Inaccurate    — The claim is partially correct but contains a significant
                          error (wrong number, wrong date, misleading framing).
                          Evidence shows what the correct version is.
        • False         — The evidence DIRECTLY CONTRADICTS the claim.
                          The claim is factually wrong.
        • Outdated      — The claim WAS true at some point but is no longer current.
                          Evidence shows it has since changed.
        • Unverifiable  — Insufficient, conflicting, or no reliable evidence found.
                          Do NOT guess — use this when genuinely unsure.

        CONFIDENCE SCORE GUIDE:
        ─────────────────────────────────────────
        0.9–1.0  → Multiple high-quality sources strongly agree
        0.7–0.9  → One strong source, or multiple weaker ones agree
        0.5–0.7  → Evidence is suggestive but incomplete or indirect
        0.3–0.5  → Evidence is weak, ambiguous, or partially conflicting
        0.1–0.3  → Almost no useful evidence; verdict is a best-guess
        0.0–0.1  → No relevant evidence at all

        IMPORTANT RULES:
        - Prioritise sources with publication dates — prefer the most recent data
        - If numbers differ slightly (rounding), lean toward Inaccurate not False
        - If no evidence mentions the claim at all, verdict = Unverifiable
        - Be specific in your explanation — cite numbers and source titles
        - Do NOT hallucinate facts not present in the evidence

        {format_instructions}

        Return ONLY the JSON. No explanation outside the JSON object.
    """).strip(),
)


# ── Fact Verifier ─────────────────────────────────────────────────────────────

class FactVerifier:
    """
    Verifies a single claim against web evidence using Gemini 1.5 Pro.

    Usage:
        verifier = FactVerifier()
        verdict  = verifier.verify(claim, sources)
        claim.verdict = verdict
    """

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model_name:  str           = "llama-3.3-70b-versatile",
        temperature: float         = 0.0,
        provider:    str           = "groq",   # "groq" (free) or "google"
    ):
        self._llm    = self._build_llm(api_key, model_name, temperature, provider)
        self._parser = self._build_parser()
        logger.info("FactVerifier initialised", provider=provider, model=model_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def verify(self, claim: Claim, sources: list[EvidenceSource]) -> Verdict:
        """
        Verify a claim against its evidence sources.

        Args:
            claim:   The Claim to verify.
            sources: Evidence sources retrieved from web search.

        Returns:
            A Verdict object. Never raises — returns Unverifiable on all errors.
        """
        logger.info("Verifying claim", claim_id=claim.id, sources=len(sources))

        # No evidence → always Unverifiable, skip LLM call
        if not sources:
            logger.info("No evidence — skipping LLM, returning Unverifiable", claim_id=claim.id)
            return Verdict(
                label       = VerdictLabel.UNVERIFIABLE,
                confidence  = 0.0,
                explanation = (
                    "No web sources were found for this claim. "
                    "It could not be verified or refuted."
                ),
                sources     = [],
            )

        try:
            raw = self._call_llm(claim, sources)
            return self._raw_to_verdict(raw, sources)
        except Exception as exc:
            logger.error(
                "Verification failed — returning Unverifiable",
                claim_id = claim.id,
                error    = str(exc),
            )
            return Verdict(
                label       = VerdictLabel.UNVERIFIABLE,
                confidence  = 0.0,
                explanation = (
                    f"Verification could not be completed due to a technical error. "
                    f"Please try again. (Detail: {str(exc)[:100]})"
                ),
                sources     = sources,
            )

    # ── LLM + Parser Setup ────────────────────────────────────────────────────

    def _build_llm(
        self,
        api_key:     Optional[str],
        model_name:  str,
        temperature: float,
        provider:    str = "groq",
    ):
        from app.utils.llm_factory import build_llm
        if not api_key:
            env_map = {"groq": "GROQ_API_KEY", "google": "GOOGLE_API_KEY"}
            api_key = os.getenv(env_map.get(provider, "GROQ_API_KEY"), "")
        return build_llm(provider=provider, api_key=api_key, model_name=model_name, temperature=temperature)

    def _build_parser(self) -> OutputFixingParser:
        base    = PydanticOutputParser(pydantic_object=RawVerdict)
        fixer   = OutputFixingParser.from_llm(parser=base, llm=self._llm)
        return fixer

    # ── LLM Call ─────────────────────────────────────────────────────────────

    @retry_with_backoff
    def _call_llm(self, claim: Claim, sources: list[EvidenceSource]) -> RawVerdict:
        """Build prompt, call Gemini, parse and return RawVerdict."""
        evidence_block = self._format_evidence(sources)
        format_instr   = PydanticOutputParser(
            pydantic_object=RawVerdict
        ).get_format_instructions()

        prompt_text = _VERIFY_PROMPT.format(
            claim               = claim.text,
            category            = claim.category.value,
            evidence_block      = evidence_block,
            format_instructions = format_instr,
        )

        logger.debug(
            "Calling Gemini for verification",
            claim_id     = claim.id,
            prompt_chars = len(prompt_text),
        )

        response = self._llm.invoke(prompt_text)
        content  = response.content if hasattr(response, "content") else str(response)
        content  = _strip_markdown_fences(content)

        logger.debug("Gemini verification response", claim_id=claim.id, chars=len(content))
        return self._parser.parse(content)

    # ── Formatting Helpers ────────────────────────────────────────────────────

    def _format_evidence(self, sources: list[EvidenceSource]) -> str:
        """Format evidence sources as a numbered block for the prompt."""
        if not sources:
            return "No evidence sources available."

        # Respect a maximum number of evidence items to include in the prompt
        max_evidence = int(os.getenv("FACTCHECKER_MAX_EVIDENCE", "3"))
        lines = []
        for i, src in enumerate(sources[:max_evidence]):
            date_str = f" (Published: {src.date})" if src.date else ""
            lines.append(
                f"[{i}] Title: {src.title}{date_str}\n"
                f"    URL: {src.url}\n"
                f"    Excerpt: {src.snippet}"
            )
        if len(sources) > max_evidence:
            lines.append(f"... and {len(sources) - max_evidence} more sources omitted for brevity")
        return "\n\n".join(lines)

    # ── Conversion ────────────────────────────────────────────────────────────

    def _raw_to_verdict(
        self,
        raw:     RawVerdict,
        sources: list[EvidenceSource],
    ) -> Verdict:
        """Convert RawVerdict (LLM output) to a typed Verdict."""
        # Parse verdict label — case-insensitive with fuzzy matching
        label = _parse_verdict_label(raw.label)

        # Attach only the sources the LLM cited
        cited_sources = self._resolve_cited_sources(raw.cited_source_indices, sources)

        return Verdict(
            label       = label,
            confidence  = max(0.0, min(1.0, raw.confidence)),   # Clamp just in case
            explanation = raw.explanation.strip(),
            sources     = cited_sources if cited_sources else sources,  # Fallback to all
        )

    def _resolve_cited_sources(
        self,
        indices: list[int],
        sources: list[EvidenceSource],
    ) -> list[EvidenceSource]:
        """Return sources at the given indices; skip out-of-range indices."""
        result = []
        for i in indices:
            if 0 <= i < len(sources):
                result.append(sources[i])
        return result


# ── Module Helpers ────────────────────────────────────────────────────────────

_VERDICT_MAP = {
    "verified":     VerdictLabel.VERIFIED,
    "inaccurate":   VerdictLabel.INACCURATE,
    "false":        VerdictLabel.FALSE,
    "outdated":     VerdictLabel.OUTDATED,
    "unverifiable": VerdictLabel.UNVERIFIABLE,
    # Common LLM variations
    "partially correct": VerdictLabel.INACCURATE,
    "misleading":        VerdictLabel.INACCURATE,
    "incorrect":         VerdictLabel.FALSE,
    "wrong":             VerdictLabel.FALSE,
    "stale":             VerdictLabel.OUTDATED,
    "unknown":           VerdictLabel.UNVERIFIABLE,
    "uncertain":         VerdictLabel.UNVERIFIABLE,
    "insufficient":      VerdictLabel.UNVERIFIABLE,
}


def _parse_verdict_label(raw: str) -> VerdictLabel:
    """Map LLM-generated label string → VerdictLabel enum (fuzzy, case-insensitive)."""
    normalised = raw.strip().lower()
    if normalised in _VERDICT_MAP:
        return _VERDICT_MAP[normalised]
    # Substring match fallback
    for key, label in _VERDICT_MAP.items():
        if key in normalised:
            return label
    logger.warning("Unknown verdict label from LLM — defaulting to Unverifiable", raw=raw)
    return VerdictLabel.UNVERIFIABLE


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` fences that Gemini sometimes adds."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()
