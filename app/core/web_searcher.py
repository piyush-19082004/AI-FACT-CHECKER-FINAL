"""
Web Searcher — Stage 3 of the fact-checking pipeline.

Responsibilities:
  • Accept a Claim and build an optimal search query
  • Search the live web using Tavily API (primary)
  • Fall back to DuckDuckGo if Tavily fails or quota is exhausted
  • Return a list of EvidenceSource objects per claim
  • Rate-limit all outbound calls to stay within free-tier limits

Design decisions:
  • Query builder crafts specific queries from claim text + category
  • Tavily returns structured JSON with relevance scores — ideal for AI agents
  • DuckDuckGo fallback uses unofficial search (no API key needed)
  • All errors are caught per-claim — never crash the pipeline
"""

from __future__ import annotations

import os
import re
from typing import Optional

from app.models.claim import Claim, ClaimCategory
from app.models.verdict import EvidenceSource
from app.utils.logger import get_logger
from app.utils.rate_limiter import retry_with_backoff

logger = get_logger(__name__)

# Max results to fetch per claim from each search engine
DEFAULT_MAX_RESULTS = 3


# ── Query Builder ─────────────────────────────────────────────────────────────

class SearchQueryBuilder:
    """
    Constructs optimised search queries from claim text and category.

    A raw claim like:
      "India's GDP grew by 7.2% in fiscal year 2022-23."
    becomes:
      "India GDP growth rate 7.2% 2022-23 fact check"
    """

    # Category-specific suffix hints to steer results toward authoritative sources
    _CATEGORY_HINTS: dict[ClaimCategory, str] = {
        ClaimCategory.STATISTICAL:  "statistics data source",
        ClaimCategory.SCIENTIFIC:   "scientific study research evidence",
        ClaimCategory.ECONOMIC:     "economic data GDP official statistics",
        ClaimCategory.HISTORICAL:   "historical fact verified",
        ClaimCategory.POLITICAL:    "official government policy verified",
        ClaimCategory.GEOGRAPHICAL: "geographic data official source",
        ClaimCategory.OTHER:        "fact check verify",
    }

    def build(self, claim: Claim) -> str:
        """Return a search-engine-friendly query string for the given claim."""
        # Strip filler words that waste query space
        text = self._compress_claim(claim.text)
        hint = self._CATEGORY_HINTS.get(claim.category, "fact check")
        query = f"{text} {hint}"

        # Hard cap — most search engines perform poorly on very long queries
        if len(query) > 200:
            query = query[:197] + "..."

        logger.debug("Search query built", claim_id=claim.id, query=query)
        return query

    def _compress_claim(self, text: str) -> str:
        """Remove filler words and shorten claim for query use."""
        fillers = r"\b(the|a|an|is|are|was|were|has|have|had|been|according to|states that|claims that)\b"
        compressed = re.sub(fillers, " ", text, flags=re.IGNORECASE)
        compressed = re.sub(r"\s{2,}", " ", compressed).strip()
        return compressed


# ── Tavily Searcher ───────────────────────────────────────────────────────────

class TavilySearcher:
    """
    Searches the web using the Tavily API.
    Tavily is purpose-built for AI agents: returns structured JSON with
    relevance scores, publication dates, and clean snippets.
    """

    def __init__(self, api_key: Optional[str] = None, max_results: int = DEFAULT_MAX_RESULTS):
        self.max_results = max_results
        key = api_key or os.getenv("TAVILY_API_KEY")
        if not key:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. "
                "Get a free key at https://app.tavily.com and add it to your .env."
            )
        try:
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=key)
        except ImportError as exc:
            raise ImportError(
                "tavily-python is not installed. Run: pip install tavily-python"
            ) from exc

    @retry_with_backoff
    def search(self, query: str) -> list[EvidenceSource]:
        """
        Execute a Tavily search and return evidence sources.

        Args:
            query: Search query string

        Returns:
            List of EvidenceSource objects (may be empty if no results)
        """
        logger.debug("Tavily search", query=query[:80])
        try:
            response = self._client.search(
                query              = query,
                max_results        = self.max_results,
                search_depth       = "basic",   # "advanced" costs 2 credits vs 1
                include_answer     = False,      # We do our own LLM reasoning
                include_raw_content= False,      # Snippets are enough
            )
            results = response.get("results", [])
            sources = [self._to_evidence(r) for r in results if r.get("url")]
            logger.info("Tavily search complete", results=len(sources), query=query[:60])
            return sources

        except Exception as exc:
            # Re-raise so retry_with_backoff can handle quota / network errors
            logger.warning("Tavily search error", error=str(exc), query=query[:60])
            raise

    def _to_evidence(self, result: dict) -> EvidenceSource:
        return EvidenceSource(
            url     = result.get("url", ""),
            title   = result.get("title", "Untitled"),
            snippet = result.get("content", result.get("snippet", "No excerpt available.")),
            date    = result.get("published_date"),
        )


# ── DuckDuckGo Fallback Searcher ──────────────────────────────────────────────

class DuckDuckGoSearcher:
    """
    Fallback searcher using DuckDuckGo (no API key required).
    Used when Tavily is unavailable or quota is exceeded.

    Note: DDG does not provide publication dates or relevance scores.
    """

    def __init__(self, max_results: int = DEFAULT_MAX_RESULTS):
        self.max_results = max_results
        try:
            from duckduckgo_search import DDGS
            self._DDGS = DDGS
        except ImportError as exc:
            raise ImportError(
                "duckduckgo-search is not installed. Run: pip install duckduckgo-search"
            ) from exc

    @retry_with_backoff
    def search(self, query: str) -> list[EvidenceSource]:
        """Execute a DuckDuckGo search and return evidence sources."""
        logger.debug("DuckDuckGo fallback search", query=query[:80])
        try:
            with self._DDGS() as ddgs:
                raw_results = list(ddgs.text(
                    query,
                    max_results = self.max_results,
                    safesearch  = "moderate",
                ))
            sources = [self._to_evidence(r) for r in raw_results if r.get("href")]
            logger.info("DDG search complete", results=len(sources), query=query[:60])
            return sources

        except Exception as exc:
            logger.warning("DDG search error", error=str(exc), query=query[:60])
            raise

    def _to_evidence(self, result: dict) -> EvidenceSource:
        return EvidenceSource(
            url     = result.get("href", ""),
            title   = result.get("title", "Untitled"),
            snippet = result.get("body", "No excerpt available."),
            date    = None,   # DDG doesn't reliably provide dates
        )


# ── Web Searcher (Facade) ─────────────────────────────────────────────────────

class WebSearcher:
    """
    Unified search interface with automatic Tavily → DuckDuckGo fallback.

    Usage:
        searcher = WebSearcher()
        sources  = searcher.search_for_claim(claim)
    """

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        max_results:    int            = DEFAULT_MAX_RESULTS,
    ):
        self.query_builder = SearchQueryBuilder()
        self._max_results  = max_results

        # Try to initialise Tavily; gracefully degrade to DDG-only if key missing
        self._tavily: Optional[TavilySearcher] = None
        self._ddg:    Optional[DuckDuckGoSearcher] = None

        tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if tavily_key:
            try:
                self._tavily = TavilySearcher(api_key=tavily_key, max_results=max_results)
                logger.info("Tavily search enabled")
            except Exception as exc:
                logger.warning("Tavily unavailable — will use DDG only", error=str(exc))

        try:
            self._ddg = DuckDuckGoSearcher(max_results=max_results)
            logger.info("DuckDuckGo fallback enabled")
        except ImportError:
            logger.warning("DuckDuckGo not available — install duckduckgo-search")

        if not self._tavily and not self._ddg:
            raise RuntimeError(
                "No search engine is available. "
                "Set TAVILY_API_KEY or install duckduckgo-search."
            )

    def search_for_claim(self, claim: Claim) -> list[EvidenceSource]:
        """
        Search the web for evidence related to a claim.
        Tries Tavily first; falls back to DuckDuckGo on any failure.

        Args:
            claim: The Claim to find evidence for.

        Returns:
            List of EvidenceSource objects (empty list if all searches fail).
        """
        query = self.query_builder.build(claim)
        claim.search_query = query  # Store for display in UI

        # ── Primary: Tavily ───────────────────────────────────────────────────
        if self._tavily:
            try:
                sources = self._tavily.search(query)
                if sources:
                    return sources
                logger.info("Tavily returned 0 results — trying DDG", claim_id=claim.id)
            except Exception as exc:
                logger.warning(
                    "Tavily failed — falling back to DDG",
                    claim_id = claim.id,
                    error    = str(exc),
                )

        # ── Fallback: DuckDuckGo ──────────────────────────────────────────────
        if self._ddg:
            try:
                return self._ddg.search(query)
            except Exception as exc:
                logger.error(
                    "DDG fallback also failed",
                    claim_id = claim.id,
                    error    = str(exc),
                )

        logger.warning("All searches failed — returning empty evidence", claim_id=claim.id)
        return []
