"""
PDF Extractor — Stage 1 of the fact-checking pipeline.

Responsibilities:
  • Accept a PDF as raw bytes (from Streamlit file upload)
  • Validate the file is a real, readable PDF
  • Extract text page-by-page using PyMuPDF
  • Normalize whitespace and remove boilerplate noise
  • Detect image-only / scanned PDFs and surface a clear error
  • Return structured PageText objects ready for claim extraction

Design decisions:
  • Uses PyMuPDF (fitz) — fastest Python PDF parser, no Java dependency
  • Processes pages lazily to limit memory on large files
  • Strips running headers/footers heuristically (repeated lines)
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class PageText:
    """Text content extracted from a single PDF page."""
    page_number: int          # 1-based
    text:        str          # Cleaned text content
    char_count:  int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    @property
    def is_empty(self) -> bool:
        return self.char_count < 20  # Fewer than 20 chars = effectively blank

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"<PageText page={self.page_number} chars={self.char_count} preview={preview!r}>"


@dataclass
class ExtractionResult:
    """Result of extracting text from an entire PDF document."""
    filename:       str
    total_pages:    int
    pages:          list[PageText]
    skipped_pages:  list[int]           # Pages with no extractable text
    is_scanned:     bool = False        # True if majority of pages are image-only
    warnings:       list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenated text from all non-empty pages."""
        return "\n\n".join(p.text for p in self.pages if not p.is_empty)

    @property
    def meaningful_pages(self) -> list[PageText]:
        return [p for p in self.pages if not p.is_empty]

    @property
    def text_page_count(self) -> int:
        return len(self.meaningful_pages)

    def __repr__(self) -> str:
        return (
            f"<ExtractionResult '{self.filename}' "
            f"pages={self.total_pages} text_pages={self.text_page_count} "
            f"scanned={self.is_scanned}>"
        )


# ── Exceptions ────────────────────────────────────────────────────────────────

class PDFExtractionError(Exception):
    """Raised when a PDF cannot be opened or is not a valid PDF."""

class ScannedPDFError(PDFExtractionError):
    """Raised when a PDF is image-only and contains no machine-readable text."""

class EmptyPDFError(PDFExtractionError):
    """Raised when a PDF produces no usable text after extraction."""


# ── Extractor ─────────────────────────────────────────────────────────────────

class PDFExtractor:
    """
    Extracts and cleans text from PDF files.

    Usage:
        extractor = PDFExtractor(max_pages=50)
        result = extractor.extract(pdf_bytes, filename="report.pdf")
    """

    # Regex patterns for noise removal
    _WHITESPACE_RE     = re.compile(r"\s{3,}")          # 3+ consecutive whitespace
    _PAGE_NUMBER_RE    = re.compile(r"^\s*\d+\s*$", re.MULTILINE)  # Lone page numbers
    _URL_RE            = re.compile(r"https?://\S+")    # URLs (kept but normalised)
    _BULLET_CLEAN_RE   = re.compile(r"^[•·‣▪▸\-–—]\s+", re.MULTILINE)

    def __init__(
        self,
        max_pages: int = 50,
        min_text_ratio: float = 0.10,   # Min ratio of text pages before flagging as scanned
    ):
        self.max_pages = max_pages
        self.min_text_ratio = min_text_ratio

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, pdf_bytes: bytes, filename: str = "document.pdf") -> ExtractionResult:
        """
        Main entry point. Accepts raw PDF bytes, returns ExtractionResult.

        Args:
            pdf_bytes: Raw bytes from Streamlit's UploadedFile.read()
            filename:  Original filename for display purposes

        Raises:
            PDFExtractionError:  If the bytes are not a valid PDF
            ScannedPDFError:     If the PDF is image-only
            EmptyPDFError:       If no usable text can be extracted
        """
        logger.info("Starting PDF extraction", filename=filename, size_kb=len(pdf_bytes) // 1024)

        doc = self._open_pdf(pdf_bytes, filename)

        try:
            total_pages = doc.page_count
            logger.info("PDF opened successfully", total_pages=total_pages)

            pages_to_process = min(total_pages, self.max_pages)
            warnings: list[str] = []

            if total_pages > self.max_pages:
                warnings.append(
                    f"PDF has {total_pages} pages — only the first {self.max_pages} "
                    f"will be processed."
                )
                logger.warning("PDF truncated", original=total_pages, processing=pages_to_process)

            raw_pages, skipped = self._extract_pages(doc, pages_to_process)

        finally:
            doc.close()

        # ── Scanned PDF detection ──────────────────────────────────────────
        is_scanned = self._detect_scanned(raw_pages, pages_to_process)
        if is_scanned:
            logger.error("Scanned PDF detected — no machine-readable text", filename=filename)
            raise ScannedPDFError(
                f"'{filename}' appears to be a scanned / image-based PDF. "
                "Please upload a text-based PDF or run OCR first."
            )

        # ── Empty document check ───────────────────────────────────────────
        meaningful = [p for p in raw_pages if not p.is_empty]
        if not meaningful:
            raise EmptyPDFError(
                f"'{filename}' contains no extractable text. "
                "The PDF may be encrypted or corrupted."
            )

        # ── Header / footer deduplication ─────────────────────────────────
        cleaned_pages = self._remove_repeated_lines(raw_pages)

        result = ExtractionResult(
            filename=filename,
            total_pages=total_pages,
            pages=cleaned_pages,
            skipped_pages=skipped,
            is_scanned=is_scanned,
            warnings=warnings,
        )

        logger.info(
            "Extraction complete",
            filename=filename,
            meaningful_pages=result.text_page_count,
            total_chars=len(result.full_text),
            skipped_pages=len(skipped),
        )
        return result

    # ── Internal Methods ──────────────────────────────────────────────────────

    def _open_pdf(self, pdf_bytes: bytes, filename: str) -> fitz.Document:
        """Open PDF from bytes; raise PDFExtractionError on failure."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if doc.is_encrypted:
                # Try opening with empty password (some PDFs set encryption but no password)
                if not doc.authenticate(""):
                    doc.close()
                    raise PDFExtractionError(
                        f"'{filename}' is password-protected. "
                        "Please provide an unprotected PDF."
                    )
            return doc
        except fitz.FileDataError as exc:
            raise PDFExtractionError(
                f"'{filename}' is not a valid PDF file. "
                f"Please upload a proper .pdf document. (Detail: {exc})"
            ) from exc

    def _extract_pages(
        self,
        doc: fitz.Document,
        pages_to_process: int,
    ) -> tuple[list[PageText], list[int]]:
        """Extract and clean text from each page."""
        pages: list[PageText] = []
        skipped: list[int] = []

        for page_idx in range(pages_to_process):
            page_num = page_idx + 1  # 1-based
            try:
                page = doc[page_idx]
                raw_text = page.get_text("text")   # Plain text extraction
                cleaned  = self._clean_text(raw_text)

                page_obj = PageText(page_number=page_num, text=cleaned)
                pages.append(page_obj)

                if page_obj.is_empty:
                    skipped.append(page_num)
                    logger.debug("Page skipped — no text", page=page_num)

            except Exception as exc:
                logger.warning("Failed to extract page", page=page_num, error=str(exc))
                skipped.append(page_num)

        return pages, skipped

    def _clean_text(self, raw: str) -> str:
        """
        Normalize extracted text:
          1. Strip lone page numbers
          2. Normalize excessive whitespace
          3. Clean up bullet characters
          4. Collapse multiple blank lines to one
        """
        if not raw:
            return ""

        text = raw

        # Remove lone page numbers (lines containing only digits)
        text = self._PAGE_NUMBER_RE.sub("", text)

        # Normalize bullet characters → dash for consistency
        text = self._BULLET_CLEAN_RE.sub("- ", text)

        # Collapse 3+ spaces/tabs to a single space
        text = self._WHITESPACE_RE.sub(" ", text)

        # Collapse multiple newlines to max two (preserve paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _detect_scanned(self, pages: list[PageText], total_processed: int) -> bool:
        """
        Heuristic: if fewer than min_text_ratio of pages have extractable text,
        the document is likely a scanned PDF.
        """
        if total_processed == 0:
            return False
        meaningful = sum(1 for p in pages if not p.is_empty)
        ratio = meaningful / total_processed
        return ratio < self.min_text_ratio

    def _remove_repeated_lines(self, pages: list[PageText]) -> list[PageText]:
        """
        Remove lines that appear on many pages (running headers/footers).
        A line appearing on > 50% of text pages is treated as boilerplate.
        """
        from collections import Counter

        meaningful = [p for p in pages if not p.is_empty]
        if len(meaningful) < 3:
            return pages  # Not enough pages to detect patterns

        # Collect all lines across pages
        page_line_sets: list[set[str]] = []
        for page in meaningful:
            lines = {
                line.strip()
                for line in page.text.splitlines()
                if len(line.strip()) > 5  # Ignore very short lines
            }
            page_line_sets.append(lines)

        # Count how many pages each line appears on
        all_lines = [line for lines in page_line_sets for line in lines]
        line_counts = Counter(all_lines)
        threshold = max(3, len(meaningful) * 0.5)  # At least 3 pages or 50%

        boilerplate: set[str] = {
            line for line, count in line_counts.items()
            if count >= threshold
        }

        if boilerplate:
            logger.debug("Removing boilerplate lines", count=len(boilerplate))

        # Rebuild pages without boilerplate
        cleaned: list[PageText] = []
        for page in pages:
            if page.is_empty:
                cleaned.append(page)
                continue
            filtered_lines = [
                line for line in page.text.splitlines()
                if line.strip() not in boilerplate
            ]
            new_text = "\n".join(filtered_lines).strip()
            cleaned.append(PageText(page_number=page.page_number, text=new_text))

        return cleaned


# ── Module-level convenience function ─────────────────────────────────────────

def extract_pdf(
    pdf_bytes: bytes,
    filename: str = "document.pdf",
    max_pages: int = 50,
) -> ExtractionResult:
    """
    Convenience wrapper — create a PDFExtractor and run extraction.

    Example:
        with open("report.pdf", "rb") as f:
            result = extract_pdf(f.read(), "report.pdf")
        print(result.full_text[:500])
    """
    extractor = PDFExtractor(max_pages=max_pages)
    return extractor.extract(pdf_bytes, filename)
