"""
Claim Extractor — Stage 2 of the fact-checking pipeline.

Responsibilities:
  • Accept full document text (from PDFExtractor)
  • Chunk large documents into overlapping windows
  • Send each chunk to Gemini 1.5 Pro via LangChain
  • Parse structured JSON output into typed Claim objects
  • Deduplicate semantically similar claims
  • Return a ranked, de-duplicated List[Claim]

Design decisions:
  • Uses LangChain's PydanticOutputParser for typed, validated output
  • Overlapping chunks (10% overlap) to avoid splitting claims at boundaries
  • OutputFixingParser as fallback — re-prompts Gemini to fix malformed JSON
  • Prompt engineered to focus on NUMERICAL/STATISTICAL claims (most verifiable)
  • Category detection baked into the prompt (not post-processed)
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

from app.models.claim import Claim, ClaimCategory
from app.core.pdf_extractor import ExtractionResult
from app.utils.logger import get_logger
from app.utils.rate_limiter import retry_with_backoff

logger = get_logger(__name__)


# Optional dependency shim: allow tests to patch `ChatGoogleGenerativeAI` even
# when `langchain_google_genai` is not installed in the environment.
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:
    ChatGoogleGenerativeAI = None


# ── Intermediate Pydantic schema for LLM output ───────────────────────────────
# Separate from app.models.claim.Claim — this is the raw shape the LLM returns.
# We then convert RawClaim → Claim with full validation.

class RawClaim(BaseModel):
    """Shape the LLM must output for each extracted claim."""
    text:        str = Field(..., description="Exact verifiable factual claim, verbatim or near-verbatim from the text")
    category:    str = Field(..., description="One of: Statistical, Historical, Scientific, Economic, Political, Geographical, Other")
    page_hint:   int = Field(default=1, description="Approximate page number where this claim appears")
    context:     str = Field(default="", description="1-2 surrounding sentences providing context")


class RawClaimList(BaseModel):
    """Wrapper so the LLM returns a list of claims as a single JSON object."""
    claims: list[RawClaim] = Field(default_factory=list)


# ── Prompt Template ───────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text", "format_instructions"],
    template=textwrap.dedent("""
        You are an expert fact-checker. Your task is to extract ALL verifiable
        factual claims from the text below.

        FOCUS ON (in priority order):
        1. STATISTICAL claims — specific numbers, percentages, counts, rates
           Example: "The unemployment rate fell to 3.7% in November 2023."
        2. SCIENTIFIC claims — research findings, study results, measurements
           Example: "CO2 concentration reached 421 ppm in May 2023."
        3. ECONOMIC claims — GDP figures, inflation rates, market data
           Example: "India's GDP grew by 7.2% in fiscal year 2022–23."
        4. HISTORICAL claims — dated events, milestones, records
           Example: "The Eiffel Tower was completed in 1889."
        5. GEOGRAPHICAL claims — areas, populations, distances
           Example: "The Amazon River is approximately 6,400 km long."
        6. POLITICAL claims — specific policy outcomes, election results
           Example: "The bill passed with 67 votes in the Senate."

        RULES:
        - Extract each claim as a COMPLETE, STANDALONE statement
        - Include the specific number/date/statistic in the claim text
        - Do NOT paraphrase — stay close to the original wording
        - Do NOT extract vague opinions or predictions
        - Do NOT extract the same claim twice
        - Limit to a maximum of 15 claims per chunk
        - Assign the most specific category that applies

        TEXT TO ANALYSE:
        ----------------
        {text}
        ----------------

        {format_instructions}

        Return ONLY the JSON. No explanation, no markdown fences.
    """).strip(),
)


# ── Claim Extractor ───────────────────────────────────────────────────────────

class ClaimExtractor:
    """
    Extracts verifiable factual claims from document text using Gemini + LangChain.

    Usage:
        extractor = ClaimExtractor()
        claims = extractor.extract(extraction_result)
    """

    # Chunk settings — tuned to fit within Gemini's context while staying fast
    # Allow runtime override for low-memory environments via env vars:
    # - FACTCHECKER_LOW_MEMORY=1 will halve the default chunk size
    # - FACTCHECKER_CHUNK_SIZE can set an explicit chunk size
    _default_chunk = int(os.getenv("FACTCHECKER_CHUNK_SIZE", "3000"))
    if os.getenv("FACTCHECKER_LOW_MEMORY", "0") in ("1", "true", "True"):
        _default_chunk = max(512, _default_chunk // 2)
    CHUNK_SIZE    = _default_chunk   # characters per chunk
    CHUNK_OVERLAP = 300              # overlap to avoid boundary splits
    MAX_CLAIMS    = 20               # hard cap on total claims per document

    def __init__(
        self,
        api_key:    Optional[str] = None,
        model_name: str           = "llama-3.3-70b-versatile",
        temperature: float        = 0.0,
        max_claims: int           = MAX_CLAIMS,
        provider:   str           = "groq",   # "groq" (free) or "google"
    ):
        self.max_claims = max_claims
        self._llm      = self._build_llm(api_key, model_name, temperature, provider)
        self._parser   = self._build_parser()
        logger.info("ClaimExtractor initialised", provider=provider, model=model_name, max_claims=max_claims)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, extraction_result: ExtractionResult) -> list[Claim]:
        """
        Main entry point. Accepts an ExtractionResult from PDFExtractor.

        Args:
            extraction_result: Output of PDFExtractor.extract()

        Returns:
            List of Claim objects, deduplicated and ranked by verifiability.
        """
        full_text = extraction_result.full_text
        filename  = extraction_result.filename

        logger.info(
            "Starting claim extraction",
            filename=filename,
            total_chars=len(full_text),
        )

        if not full_text.strip():
            logger.warning("Empty text — no claims to extract", filename=filename)
            return []

        chunks     = self._chunk_text(full_text)
        raw_claims = self._extract_from_chunks(chunks, extraction_result)
        claims     = self._deduplicate(raw_claims)
        claims     = claims[: self.max_claims]

        # Re-assign sequential IDs after dedup
        for i, claim in enumerate(claims, start=1):
            claim.id = i

        logger.info(
            "Claim extraction complete",
            filename=filename,
            total_claims=len(claims),
        )
        return claims

    def extract_from_text(self, text: str, filename: str = "document") -> list[Claim]:
        """
        Convenience overload — accepts raw text string directly.
        Useful for testing without a full ExtractionResult.
        """
        from app.core.pdf_extractor import ExtractionResult, PageText
        mock_page   = PageText(page_number=1, text=text)
        mock_result = ExtractionResult(
            filename      = filename,
            total_pages   = 1,
            pages         = [mock_page],
            skipped_pages = [],
        )
        return self.extract(mock_result)

    # ── LLM + Parser Setup ────────────────────────────────────────────────────

    def _build_llm(
        self,
        api_key:     Optional[str],
        model_name:  str,
        temperature: float,
        provider:    str = "groq",
    ):
        """Instantiate an LLM via the provider factory (Groq or Google)."""
        from app.utils.llm_factory import build_llm
        # Fall back to env vars if no key passed explicitly
        if not api_key:
            env_map = {"groq": "GROQ_API_KEY", "google": "GOOGLE_API_KEY"}
            api_key = os.getenv(env_map.get(provider, "GROQ_API_KEY"), "")
        return build_llm(provider=provider, api_key=api_key, model_name=model_name, temperature=temperature)

    def _build_parser(self) -> OutputFixingParser:
        """
        Build a two-layer parser:
          1. PydanticOutputParser — primary: parse JSON into RawClaimList
          2. OutputFixingParser   — fallback: re-prompts Gemini to fix bad JSON
        """
        base_parser = PydanticOutputParser(pydantic_object=RawClaimList)
        fixing_parser = OutputFixingParser.from_llm(
            parser = base_parser,
            llm    = self._llm,
        )
        return fixing_parser

    # ── Text Chunking ─────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.
        Tries to split at paragraph boundaries to preserve semantic coherence.
        """
        if len(text) <= self.CHUNK_SIZE:
            return [text]

        chunks    : list[str] = []
        start     : int       = 0

        while start < len(text):
            end = start + self.CHUNK_SIZE

            if end < len(text):
                # Try to split at a paragraph boundary near the end of the chunk
                split_pos = text.rfind("\n\n", start, end)
                if split_pos == -1 or split_pos <= start:
                    # Fall back to sentence boundary
                    split_pos = text.rfind(". ", start, end)
                if split_pos == -1 or split_pos <= start:
                    split_pos = end   # Hard split as last resort
                else:
                    split_pos += 2    # Include the delimiter
                end = split_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Advance with overlap
            start = end - self.CHUNK_OVERLAP
            if start >= len(text):
                break

        logger.debug("Text chunked", total_chunks=len(chunks))
        return chunks

    # ── Extraction Loop ───────────────────────────────────────────────────────

    def _extract_from_chunks(
        self,
        chunks:           list[str],
        extraction_result: ExtractionResult,
    ) -> list[Claim]:
        """Process each chunk through the LLM and collect raw claims."""
        all_claims: list[Claim] = []
        claim_id_counter        = 1

        for chunk_idx, chunk in enumerate(chunks, start=1):
            logger.info(
                "Processing chunk",
                chunk=chunk_idx,
                of=len(chunks),
                chars=len(chunk),
            )
            try:
                raw_list = self._call_llm(chunk)
                for raw in raw_list.claims:
                    claim = self._raw_to_claim(raw, claim_id_counter, extraction_result)
                    if claim:
                        all_claims.append(claim)
                        claim_id_counter += 1

            except Exception as exc:
                logger.error(
                    "Chunk extraction failed — skipping",
                    chunk=chunk_idx,
                    error=str(exc),
                )
                continue   # Never crash the whole pipeline for one bad chunk

        return all_claims

    @retry_with_backoff
    def _call_llm(self, text: str) -> RawClaimList:
        """
        Send one chunk to Gemini and parse the response.
        Decorated with retry logic for transient API failures.
        """
        format_instructions = PydanticOutputParser(
            pydantic_object=RawClaimList
        ).get_format_instructions()

        prompt_text = _EXTRACTION_PROMPT.format(
            text                = text,
            format_instructions = format_instructions,
        )

        logger.debug("Calling Gemini for claim extraction", prompt_chars=len(prompt_text))
        response = self._llm.invoke(prompt_text)

        # response is an AIMessage — get string content
        content = response.content if hasattr(response, "content") else str(response)

        # Strip any accidental markdown fences before parsing
        content = _strip_markdown_fences(content)

        logger.debug("Gemini responded", response_chars=len(content))
        return self._parser.parse(content)

    # ── Conversion & Validation ───────────────────────────────────────────────

    def _raw_to_claim(
        self,
        raw:               RawClaim,
        claim_id:          int,
        extraction_result: ExtractionResult,
    ) -> Optional[Claim]:
        """Convert a RawClaim (LLM output) to a validated Claim object."""
        try:
            # Map LLM category string → ClaimCategory enum (case-insensitive, fuzzy)
            category = _parse_category(raw.category)

            # Resolve page number — LLM gives a hint; clamp to valid range
            page_num = max(1, min(raw.page_hint, extraction_result.total_pages))

            claim = Claim(
                id          = claim_id,
                text        = raw.text.strip(),
                category    = category,
                page_number = page_num,
                context     = raw.context.strip() or None,
            )
            return claim

        except Exception as exc:
            logger.warning(
                "Skipping invalid claim from LLM",
                raw_text=raw.text[:80],
                error=str(exc),
            )
            return None

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _deduplicate(self, claims: list[Claim]) -> list[Claim]:
        """
        Remove near-duplicate claims using token overlap (Jaccard similarity).
        Threshold: if two claims share >60% of their word tokens, keep only the longer one.

        This is a lightweight approach — no embeddings needed.
        """
        if len(claims) <= 1:
            return claims

        def tokenize(text: str) -> set[str]:
            return set(re.findall(r"\b\w+\b", text.lower()))

        def jaccard(a: set, b: set) -> float:
            if not a and not b:
                return 1.0
            return len(a & b) / len(a | b)

        kept: list[Claim] = []
        for candidate in claims:
            candidate_tokens = tokenize(candidate.text)
            is_duplicate = False
            for existing in kept:
                existing_tokens = tokenize(existing.text)
                similarity = jaccard(candidate_tokens, existing_tokens)
                if similarity >= 0.60:
                    # Keep the longer, more specific version
                    if len(candidate.text) > len(existing.text):
                        kept.remove(existing)
                        kept.append(candidate)
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(candidate)

        removed = len(claims) - len(kept)
        if removed:
            logger.info("Deduplication complete", removed=removed, kept=len(kept))

        return kept


# ── Module Helpers ────────────────────────────────────────────────────────────

def _parse_category(raw: str) -> ClaimCategory:
    """
    Map an LLM-generated category string to a ClaimCategory enum value.
    Case-insensitive, fuzzy — handles common variations.
    """
    mapping = {
        "statistical":  ClaimCategory.STATISTICAL,
        "statistic":    ClaimCategory.STATISTICAL,
        "numeric":      ClaimCategory.STATISTICAL,
        "numerical":    ClaimCategory.STATISTICAL,
        "historical":   ClaimCategory.HISTORICAL,
        "history":      ClaimCategory.HISTORICAL,
        "scientific":   ClaimCategory.SCIENTIFIC,
        "science":      ClaimCategory.SCIENTIFIC,
        "research":     ClaimCategory.SCIENTIFIC,
        "economic":     ClaimCategory.ECONOMIC,
        "economy":      ClaimCategory.ECONOMIC,
        "financial":    ClaimCategory.ECONOMIC,
        "political":    ClaimCategory.POLITICAL,
        "politics":     ClaimCategory.POLITICAL,
        "geographical": ClaimCategory.GEOGRAPHICAL,
        "geographic":   ClaimCategory.GEOGRAPHICAL,
        "geography":    ClaimCategory.GEOGRAPHICAL,
        "other":        ClaimCategory.OTHER,
    }
    return mapping.get(raw.strip().lower(), ClaimCategory.OTHER)


def _strip_markdown_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` fences that Gemini sometimes adds
    despite being told not to.
    """
    text = text.strip()
    # Remove leading fence
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    # Remove trailing fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


# ── Module-level convenience ──────────────────────────────────────────────────

def extract_claims(
    extraction_result: ExtractionResult,
    api_key: Optional[str] = None,
    max_claims: int = 20,
) -> list[Claim]:
    """
    Convenience wrapper — create a ClaimExtractor and run extraction.

    Example:
        from app.core.claim_extractor import extract_claims
        claims = extract_claims(pdf_result, max_claims=15)
    """
    extractor = ClaimExtractor(api_key=api_key, max_claims=max_claims)
    return extractor.extract(extraction_result)
