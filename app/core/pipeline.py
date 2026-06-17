"""
Pipeline Orchestrator — ties all stages of the fact-checking pipeline together.

Stage flow:
    1. Validate input   → PDFFileValidator    → reject bad files early
    2. PDF Extraction   → PDFExtractor        → ExtractionResult
    3. Sanitise text    → ContentSanitiser    → cleaned text (prompt injection guard)
    4. Claim Extraction → ClaimExtractor      → List[Claim]
    5. Web Search       → WebSearcher         → List[EvidenceSource] per claim
    6. Fact Verification→ FactVerifier        → Verdict per claim
    7. Result Assembly  → FactCheckResult     → for Streamlit display

Caching:
    Each stage is wrapped with @st.cache_data so repeat runs with the same
    PDF/settings hit the cache instead of burning API quota.

Design decisions:
  • Claims processed sequentially — keeps rate limits predictable
  • Progress callback supports Streamlit's st.progress() widget
  • Every stage wrapped in try/except — partial results always returned
  • Status object accumulates warnings/errors for post-run display
"""

from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.core.pdf_extractor import (
    PDFExtractor, ExtractionResult, PageText,
    PDFExtractionError, ScannedPDFError, EmptyPDFError,
)
from app.core.claim_extractor import ClaimExtractor
from app.core.web_searcher import WebSearcher
from app.core.fact_verifier import FactVerifier
from app.models.claim import Claim, ClaimCategory, FactCheckResult
from app.models.verdict import EvidenceSource, Verdict, VerdictLabel
from app.utils.logger import get_logger
from app.utils.validators import PDFFileValidator, ContentSanitiser
from app.utils.cache import (
    make_pdf_cache_key, make_claims_cache_key,
    make_search_cache_key, make_verify_cache_key,
    cached_extract_pdf, cached_extract_claims,
    cached_search, cached_verify,
)

logger = get_logger(__name__)

ProgressCallback = Callable[[str, float], None]


# ── Pipeline Configuration ────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """All tunable settings for a pipeline run."""
    # ── LLM provider: "groq" (free, recommended) or "google" (needs billing) ──
    llm_provider:         str           = "groq"
    groq_api_key:         Optional[str] = None
    google_api_key:       Optional[str] = None
    tavily_api_key:       Optional[str] = None
    max_pdf_pages:        int           = 20
    max_claims:           int           = 15
    max_results:          int           = 3
    # Groq defaults (free)
    groq_extract_model:   str           = "llama-3.3-70b-versatile"
    groq_verify_model:    str           = "llama-3.3-70b-versatile"
    # Google defaults (billing required)
    gemini_extract_model: str           = "gemini-2.0-flash-lite"
    gemini_verify_model:  str           = "gemini-2.0-flash"
    inter_claim_delay:    float         = 1.0   # Groq is faster; smaller delay needed
    llm_timeout_secs:     int           = 30
    max_file_size_mb:     int           = 10
    enable_cache:         bool          = True
    enable_sanitiser:     bool          = True

    @property
    def active_api_key(self) -> str:
        """Return the API key for the active provider."""
        if self.llm_provider == "groq":
            return self.groq_api_key or ""
        return self.google_api_key or ""

    @property
    def extract_model(self) -> str:
        if self.llm_provider == "groq":
            return self.groq_extract_model
        return self.gemini_extract_model

    @property
    def verify_model(self) -> str:
        if self.llm_provider == "groq":
            return self.groq_verify_model
        return self.gemini_verify_model


# ── Pipeline Status ───────────────────────────────────────────────────────────

@dataclass
class PipelineStatus:
    stage:         str       = "idle"
    current_claim: int       = 0
    total_claims:  int       = 0
    errors:        list[str] = field(default_factory=list)
    warnings:      list[str] = field(default_factory=list)
    elapsed_secs:  float     = 0.0
    is_complete:   bool      = False

    @property
    def progress_fraction(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return self.current_claim / self.total_claims


# ── Main Pipeline ─────────────────────────────────────────────────────────────

class FactCheckPipeline:
    """
    End-to-end fact-checking pipeline with caching, validation, and sanitisation.

    Usage:
        pipeline = FactCheckPipeline(config)
        result   = pipeline.run(pdf_bytes, filename="report.pdf")
    """

    def __init__(self, config: PipelineConfig):
        self.config    = config
        self.status    = PipelineStatus()
        self._sanitiser = ContentSanitiser() if config.enable_sanitiser else None
        self._start_time: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        pdf_bytes:   bytes,
        filename:    str                       = "document.pdf",
        progress_cb: Optional[ProgressCallback] = None,
    ) -> FactCheckResult:
        """
        Run the full pipeline on a PDF document.

        Raises:
            PDFExtractionError / ScannedPDFError / EmptyPDFError  — propagated to UI
            EnvironmentError  — missing API key
        """
        self._start_time = time.monotonic()
        self._notify(progress_cb, "🔒 Validating file…", 0.0)

        # ── Stage 0: File validation ──────────────────────────────────────────
        self._validate_file(pdf_bytes, filename)

        self._notify(progress_cb, "📖 Extracting PDF text…", 0.05)
        logger.info("Pipeline started", filename=filename)

        # ── Stage 1: PDF extraction (cached) ──────────────────────────────────
        extraction_dict = self._run_extraction_cached(pdf_bytes, filename)
        self.status.warnings.extend(extraction_dict.get("warnings", []))

        full_text = extraction_dict["full_text"]

        # ── Stage 2: Content sanitisation ────────────────────────────────────
        if self._sanitiser and full_text:
            full_text = self._sanitiser.sanitise(full_text, filename)

        self._notify(progress_cb, "🧠 Extracting factual claims…", 0.12)

        # ── Stage 3: Claim extraction (cached) ────────────────────────────────
        claims = self._run_claim_extraction_cached(full_text, filename)

        if not claims:
            logger.warning("No claims extracted", filename=filename)
            self._finalise()
            return FactCheckResult(
                filename     = filename,
                total_pages  = extraction_dict["total_pages"],
                total_claims = 0,
                claims       = [],
            )

        self.status.total_claims = len(claims)
        self._notify(progress_cb, f"🌐 Searching & verifying {len(claims)} claims…", 0.25)

        # ── Stages 4 + 5: Search & Verify per claim (each step cached) ────────
        verified_claims = self._run_search_and_verify(claims, progress_cb)

        # ── Stage 6: Assemble ─────────────────────────────────────────────────
        self._finalise()
        result = FactCheckResult(
            filename     = filename,
            total_pages  = extraction_dict["total_pages"],
            total_claims = len(verified_claims),
            claims       = verified_claims,
        )

        self._notify(progress_cb, "✅ Fact-checking complete!", 1.0)
        logger.info(
            "Pipeline complete",
            filename        = filename,
            total_claims    = len(verified_claims),
            elapsed_secs    = round(self.status.elapsed_secs, 1),
            verdict_summary = result.verdict_summary,
        )
        return result

    # ── Stage Runners ─────────────────────────────────────────────────────────

    def _validate_file(self, pdf_bytes: bytes, filename: str) -> None:
        """Stage 0 — validate file size and magic bytes. Propagates exceptions."""
        validator = PDFFileValidator(
            max_size_bytes=self.config.max_file_size_mb * 1024 * 1024
        )
        validator.validate(pdf_bytes, filename)   # Raises on failure

    def _run_extraction_cached(self, pdf_bytes: bytes, filename: str) -> dict:
        """Stage 1 — PDF extraction, cached by file content hash."""
        pdf_hash = make_pdf_cache_key(pdf_bytes)

        if self.config.enable_cache:
            return cached_extract_pdf(
                _pdf_hash = pdf_hash,
                pdf_bytes = pdf_bytes,
                filename  = filename,
                max_pages = self.config.max_pdf_pages,
            )
        # Cache disabled (e.g. tests)
        from app.core.pdf_extractor import PDFExtractor
        extractor = PDFExtractor(max_pages=self.config.max_pdf_pages)
        result    = extractor.extract(pdf_bytes, filename)
        return {
            "filename":       result.filename,
            "total_pages":    result.total_pages,
            "full_text":      result.full_text,
            "text_page_count":result.text_page_count,
            "skipped_pages":  result.skipped_pages,
            "warnings":       result.warnings,
            "pages": [
                {"page_number": p.page_number, "text": p.text}
                for p in result.pages
            ],
        }

    def _run_claim_extraction_cached(
        self, full_text: str, filename: str
    ) -> list[Claim]:
        """Stage 3 — claim extraction, cached by text content + settings."""
        if not full_text.strip():
            return []

        api_key = self.config.active_api_key
        if not api_key:
            raise EnvironmentError(
                f"No API key found for provider '{self.config.llm_provider}'. "
                "Add a Groq key (free at groq.com) or Google key to the sidebar."
            )

        # If the document is very large, avoid creating a single huge prompt
        # and instead stream pages in small batches into the ClaimExtractor.
        # This reduces peak memory usage on low-RAM machines.
        stream_threshold = int(os.getenv("FACTCHECKER_FULLTEXT_STREAM_THRESHOLD", "200000"))

        if len(full_text) > stream_threshold:
            logger.info("Large document detected — using streaming claim extraction", filename=filename, chars=len(full_text))
            # pages are provided by the extraction_dict earlier; reconstruct from cached pages if available
            # Try to read pages from cached extraction result stored in the PDF cache
            try:
                # cached_extract_pdf returns a dict with a `pages` list when available
                # We'll re-run a non-cached extraction to get PageText objects if needed
                from app.core.pdf_extractor import PageText, ExtractionResult
                # Attempt to split the full_text into logical page-like batches using double-newline separators
                page_texts = [p.strip() for p in full_text.split("\n\n") if p.strip()]
                # Batch pages to keep each batch reasonably small
                batch_chars = 0
                batch: list[PageText] = []
                extracted_claims: list[Claim] = []
                extractor = ClaimExtractor(
                    api_key    = api_key,
                    model_name = self.config.extract_model,
                    max_claims = self.config.max_claims,
                    provider   = self.config.llm_provider,
                )

                def flush_batch():
                    nonlocal extracted_claims, batch
                    if not batch:
                        return
                    mock_result = ExtractionResult(
                        filename=filename,
                        total_pages=len(batch),
                        pages=batch,
                        skipped_pages=[],
                    )
                    try:
                        c = extractor.extract(mock_result)
                        # append while enforcing max_claims
                        for claim in c:
                            if len(extracted_claims) >= self.config.max_claims:
                                break
                            extracted_claims.append(claim)
                    except Exception as exc:
                        logger.warning("Chunk extraction failed, continuing", error=str(exc))
                    batch = []

                for i, pt in enumerate(page_texts, start=1):
                    page_obj = PageText(page_number=i, text=pt)
                    batch.append(page_obj)
                    batch_chars += len(pt)
                    # flush when batch large enough or reached max claims
                    if batch_chars >= (extractor.CHUNK_SIZE * 2) or len(extracted_claims) >= self.config.max_claims:
                        flush_batch()
                        batch_chars = 0
                    if len(extracted_claims) >= self.config.max_claims:
                        break

                # final flush
                flush_batch()
                # reassign sequential ids
                for idx, claim in enumerate(extracted_claims, start=1):
                    claim.id = idx
                return extracted_claims
            except Exception as exc:
                logger.warning("Streaming extraction fallback failed — falling back to cached path", error=str(exc))

        # Default path (cached/non-cached as before)
        text_hash = make_claims_cache_key(
            full_text,
            self.config.extract_model,
            self.config.max_claims,
        )

        try:
            if self.config.enable_cache:
                raw_dicts = cached_extract_claims(
                    _text_hash = text_hash,
                    full_text  = full_text,
                    filename   = filename,
                    api_key    = api_key,
                    model_name = self.config.extract_model,
                    max_claims = self.config.max_claims,
                    provider   = self.config.llm_provider,
                )
            else:
                from app.core.pdf_extractor import ExtractionResult, PageText
                mock_page   = PageText(page_number=1, text=full_text)
                mock_result = ExtractionResult(
                    filename=filename, total_pages=1,
                    pages=[mock_page], skipped_pages=[],
                )
                extractor = ClaimExtractor(
                    api_key    = api_key,
                    model_name = self.config.extract_model,
                    max_claims = self.config.max_claims,
                    provider   = self.config.llm_provider,
                )
                claims = extractor.extract(mock_result)
                return claims

            # Reconstruct Claim objects from serialised dicts
            claims = [
                Claim(
                    id          = d["id"],
                    text        = d["text"],
                    category    = ClaimCategory(d["category"]),
                    page_number = d["page_number"],
                    context     = d.get("context"),
                )
                for d in raw_dicts
            ]
            return claims

        except EnvironmentError:
            raise
        except Exception as exc:
            msg = f"Claim extraction failed: {exc}"
            logger.error(msg)
            self.status.errors.append(msg)
            return []

    def _run_search_and_verify(
        self,
        claims:      list[Claim],
        progress_cb: Optional[ProgressCallback],
    ) -> list[Claim]:
        """Stages 4+5 — search + verify each claim, both steps cached."""
        api_key    = self.config.active_api_key
        total      = len(claims)

        for idx, claim in enumerate(claims, start=1):
            self.status.current_claim = idx
            fraction = 0.25 + (idx / total) * 0.75

            self._notify(
                progress_cb,
                f"🔍 Verifying claim {idx}/{total}: {claim.short_text[:55]}…",
                fraction,
            )

            # ── Search (cached per query) ─────────────────────────────────────
            sources_dicts: list[dict] = []
            try:
                from app.core.web_searcher import SearchQueryBuilder
                query = SearchQueryBuilder().build(claim)
                claim.search_query = query

                q_hash = make_search_cache_key(query)
                if self.config.enable_cache:
                    sources_dicts = cached_search(
                        _query_hash    = q_hash,
                        query          = query,
                        tavily_api_key = self.config.tavily_api_key,
                        max_results    = self.config.max_results,
                    )
                else:
                    searcher      = WebSearcher(tavily_api_key=self.config.tavily_api_key)
                    raw_sources   = searcher.search_for_claim(claim)
                    sources_dicts = [
                        {"url": s.url, "title": s.title,
                         "snippet": s.snippet, "date": s.date}
                        for s in raw_sources
                    ]
            except Exception as exc:
                logger.warning("Search failed for claim", claim_id=claim.id, error=str(exc))

                # Truncate sources to configured max to reduce memory and prompt size
                max_sources = max(1, int(os.getenv("FACTCHECKER_MAX_EVIDENCE", str(self.config.max_results))))
                sources_dicts = sources_dicts[:max_sources]
                sources = [
                    EvidenceSource(
                        url=s["url"], title=s["title"],
                        snippet=s["snippet"], date=s.get("date"),
                    )
                    for s in sources_dicts
                ]

            # ── Verify (cached per claim + evidence) ──────────────────────────
            try:
                sources_json = json.dumps(sources_dicts, sort_keys=True)
                v_hash       = make_verify_cache_key(claim.text, sources_json)

                if self.config.enable_cache:
                    verdict_dict = cached_verify(
                        _verify_hash    = v_hash,
                        claim_text      = claim.text,
                        claim_category  = claim.category.value,
                        sources_json    = sources_json,
                        api_key         = api_key,
                        model_name      = self.config.verify_model,
                        provider        = self.config.llm_provider,
                    )
                    claim.verdict = Verdict(
                        label       = VerdictLabel(verdict_dict["label"]),
                        confidence  = verdict_dict["confidence"],
                        explanation = verdict_dict["explanation"],
                        sources     = [
                            EvidenceSource(
                                url=s["url"], title=s["title"],
                                snippet=s["snippet"], date=s.get("date"),
                            )
                            for s in verdict_dict["sources"]
                        ],
                    )
                else:
                    verifier      = FactVerifier(
                        api_key    = api_key,
                        model_name = self.config.verify_model,
                        provider   = self.config.llm_provider,
                    )
                    claim.verdict = verifier.verify(claim, sources)

            except Exception as exc:
                logger.error("Verification failed", claim_id=claim.id, error=str(exc))
                claim.verdict = Verdict(
                    label       = VerdictLabel.UNVERIFIABLE,
                    confidence  = 0.0,
                    explanation = "Verification could not be completed due to a technical error.",
                    sources     = sources,
                )

            # Respect Gemini RPM limits
            if idx < total:
                time.sleep(self.config.inter_claim_delay)

        return claims

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _finalise(self) -> None:
        self.status.elapsed_secs = time.monotonic() - self._start_time
        self.status.is_complete  = True

    def _notify(
        self,
        callback: Optional[ProgressCallback],
        label:    str,
        fraction: float,
    ) -> None:
        self.status.stage = label
        if callback:
            try:
                callback(label, fraction)
            except Exception:
                pass


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_pipeline(
    pdf_bytes:      bytes,
    filename:       str,
    google_api_key: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    max_claims:     int            = 15,
    progress_cb:    Optional[ProgressCallback] = None,
) -> FactCheckResult:
    """One-line pipeline runner for Streamlit."""
    config   = PipelineConfig(
        google_api_key = google_api_key,
        tavily_api_key = tavily_api_key,
        max_claims     = max_claims,
    )
    pipeline = FactCheckPipeline(config)
    return pipeline.run(pdf_bytes, filename, progress_cb=progress_cb)
