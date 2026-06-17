"""
Streamlit cache wrappers for all expensive pipeline operations.

Strategy:
  • PDF extraction  — cached by SHA-256 hash of raw bytes (session TTL)
  • Claim extraction— cached by text hash + model + max_claims (1-hour TTL)
  • Search results  — cached by claim text + query (1-hour TTL)
  • Verification    — cached by claim text + evidence fingerprint (1-hour TTL)

Why pass the hash as the first argument?
  Streamlit hashes ALL arguments to compute a cache key. For large objects
  (full text, PDF bytes) this is slow. By passing a pre-computed hash as
  the first arg and the data as the second, we let Streamlit use the cheap
  hash as the primary key while still making the data available to the function.

Important: these functions use @st.cache_data which serialises return values.
  Pydantic models serialise cleanly; LangChain objects would not, so we
  return plain dicts/lists and reconstruct models in the pipeline.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import streamlit as st

from app.utils.logger import get_logger

logger = get_logger(__name__)

CACHE_TTL = 3600   # 1 hour — exported for use elsewhere


# ── Hashing helpers ───────────────────────────────────────────────────────────

def hash_bytes(data: bytes) -> str:
    """SHA-256 of raw bytes — stable cache key for uploaded PDFs."""
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    """SHA-256 of UTF-8 encoded string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_claims(claims_dicts: list[dict]) -> str:
    """Deterministic hash of a list of claim dicts."""
    serialised = json.dumps(claims_dicts, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()


# ── Cached PDF Extraction ─────────────────────────────────────────────────────

@st.cache_data(ttl=None, show_spinner=False)   # Session-scoped: no TTL expiry
def cached_extract_pdf(
    _pdf_hash:  str,     # Used as cache key; underscore prefix = Streamlit skips hashing this arg
    pdf_bytes:  bytes,
    filename:   str,
    max_pages:  int,
) -> Optional[dict]:
    """
    Extract text from a PDF and return a serialisable dict.
    Cached indefinitely within the Streamlit session.

    Returns None on failure (caller must handle).
    """
    from app.core.pdf_extractor import PDFExtractor, PDFExtractionError

    logger.info("Cache MISS — extracting PDF", filename=filename, hash=_pdf_hash[:12])
    try:
        extractor = PDFExtractor(max_pages=max_pages)
        result    = extractor.extract(pdf_bytes, filename)

        # Serialise to plain dict for Streamlit's pickle-based cache
        return {
            "filename":      result.filename,
            "total_pages":   result.total_pages,
            "full_text":     result.full_text,
            "text_page_count": result.text_page_count,
            "skipped_pages": result.skipped_pages,
            "warnings":      result.warnings,
            "pages": [
                {"page_number": p.page_number, "text": p.text}
                for p in result.pages
            ],
        }
    except PDFExtractionError:
        raise   # Let the UI layer handle these with user-friendly messages
    except Exception as exc:
        logger.error("Unexpected PDF extraction error", error=str(exc))
        raise


# ── Cached Claim Extraction ───────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_extract_claims(
    _text_hash:  str,         # Cache key: hash of full_text + settings
    full_text:   str,
    filename:    str,
    api_key:     str,
    model_name:  str,
    max_claims:  int,
    provider:    str = "groq",
) -> list[dict]:
    """
    Extract claims from document text and return a list of serialisable dicts.
    Cached for 1 hour — same document + settings always returns the same claims.

    Returns empty list on failure (partial results are acceptable).
    """
    from app.core.claim_extractor import ClaimExtractor
    from app.core.pdf_extractor import ExtractionResult, PageText

    logger.info("Cache MISS — extracting claims", filename=filename, hash=_text_hash[:12])

    # Reconstruct a minimal ExtractionResult from cached text
    mock_page   = PageText(page_number=1, text=full_text)
    mock_result = ExtractionResult(
        filename      = filename,
        total_pages   = 1,
        pages         = [mock_page],
        skipped_pages = [],
    )

    try:
        extractor = ClaimExtractor(
            api_key    = api_key,
            model_name = model_name,
            max_claims = max_claims,
            provider   = provider,
        )
        claims = extractor.extract(mock_result)
        # Serialise Claims → plain dicts
        return [
            {
                "id":          c.id,
                "text":        c.text,
                "category":    c.category.value,
                "page_number": c.page_number,
                "context":     c.context,
            }
            for c in claims
        ]
    except Exception as exc:
        logger.error("Claim extraction error (cached wrapper)", error=str(exc))
        raise


# ── Cached Search ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_search(
    _query_hash:    str,       # Cache key
    query:          str,
    tavily_api_key: Optional[str],
    max_results:    int,
) -> list[dict]:
    """
    Search the web for a query and return serialisable source dicts.
    Cached for 1 hour — same query always hits the same evidence.

    Returns empty list when no results found (never raises).
    """
    from app.core.web_searcher import TavilySearcher, DuckDuckGoSearcher

    logger.info("Cache MISS — searching web", query=query[:60], hash=_query_hash[:12])
    sources = []

    # Try Tavily first
    if tavily_api_key:
        try:
            searcher = TavilySearcher(api_key=tavily_api_key, max_results=max_results)
            sources  = searcher.search(query)
        except Exception as exc:
            logger.warning("Tavily search failed in cache wrapper", error=str(exc))

    # Fallback to DDG
    if not sources:
        try:
            ddg     = DuckDuckGoSearcher(max_results=max_results)
            sources = ddg.search(query)
        except Exception as exc:
            logger.warning("DDG search failed in cache wrapper", error=str(exc))

    return [
        {
            "url":     s.url,
            "title":   s.title,
            "snippet": s.snippet,
            "date":    s.date,
        }
        for s in sources
    ]


# ── Cached Verification ───────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_verify(
    _verify_hash: str,       # Cache key: hash of claim text + evidence
    claim_text:   str,
    claim_category: str,
    sources_json: str,       # JSON-serialised list of source dicts
    api_key:      str,
    model_name:   str,
    provider:     str = "groq",
) -> dict:
    """
    Verify a claim against evidence and return a serialisable verdict dict.
    Cached for 1 hour — same claim + evidence always produces same reasoning.

    Always returns a dict (never raises — returns Unverifiable on errors).
    """
    import json as _json
    from app.core.fact_verifier import FactVerifier
    from app.models.claim import Claim, ClaimCategory
    from app.models.verdict import EvidenceSource, VerdictLabel

    logger.info("Cache MISS — verifying claim", claim=claim_text[:60], hash=_verify_hash[:12])

    # Reconstruct objects from serialised data
    sources_raw = _json.loads(sources_json)
    sources = [
        EvidenceSource(
            url     = s["url"],
            title   = s["title"],
            snippet = s["snippet"],
            date    = s.get("date"),
        )
        for s in sources_raw
    ]

    claim = Claim(
        id          = 1,
        text        = claim_text,
        category    = ClaimCategory(claim_category),
        page_number = 1,
    )

    try:
        verifier = FactVerifier(api_key=api_key, model_name=model_name, provider=provider)
        verdict  = verifier.verify(claim, sources)
    except Exception as exc:
        logger.error("Verification error in cache wrapper", error=str(exc))
        from app.models.verdict import Verdict
        verdict = Verdict(
            label       = VerdictLabel.UNVERIFIABLE,
            confidence  = 0.0,
            explanation = f"Verification failed: {str(exc)[:120]}",
            sources     = sources,
        )

    return {
        "label":       verdict.label.value,
        "confidence":  verdict.confidence,
        "explanation": verdict.explanation,
        "sources": [
            {"url": s.url, "title": s.title, "snippet": s.snippet, "date": s.date}
            for s in verdict.sources
        ],
    }


# ── Cache Key Builders ────────────────────────────────────────────────────────

def make_pdf_cache_key(pdf_bytes: bytes) -> str:
    """Build the cache key for a PDF file upload."""
    return hash_bytes(pdf_bytes)


def make_claims_cache_key(full_text: str, model: str, max_claims: int) -> str:
    """Build the cache key for claim extraction."""
    return hash_text(f"{full_text[:5000]}|{model}|{max_claims}")


def make_search_cache_key(query: str) -> str:
    """Build the cache key for a web search query."""
    return hash_text(query.lower().strip())


def make_verify_cache_key(claim_text: str, sources_json: str) -> str:
    """Build the cache key for claim verification."""
    return hash_text(f"{claim_text}|{sources_json[:500]}")
