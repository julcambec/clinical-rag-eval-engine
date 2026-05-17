"""
PDF document loader for clinical practice guidelines.

Extracts text from PDF files using pymupdf (fitz), preserving page-level
metadata (source document name, page number) for downstream citation tracking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf  # fitz

from clinical_rag.ops.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentPage:
    """
    A single page of extracted text with its metadata.

    Attributes:
        text: Extracted text content from the page.
        metadata: Dictionary containing at minimum 'source' (filename)
                  and 'page' (1-indexed page number).
    """

    text: str
    metadata: dict[str, str | int] = field(default_factory=dict)


def clean_page_text(text: str) -> str:
    """
    Remove common PDF artifacts: headers, footers, page numbers, copyright notices.

    Clinical guideline PDFs typically repeat the document title, copyright line,
    and page number on every page. These add noise to chunks and degrade retrieval
    quality. This function strips them using patterns common across WHO, NICE,
    and CANMAT guideline formats.
    """
    ### TODO: So far I've added NICE patterns, but still need to handle WHO and CANMAT
    ###       artifacts

    lines = text.split("\n")
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines (will be normalized later)
        if not stripped:
            cleaned_lines.append("")
            continue

        # Skip lines that are just page numbers: "Page 66 of 111", "66", "- 42 -"
        if re.match(r"^[-–—]?\s*\d{1,4}\s*[-–—]?$", stripped):
            continue
        if re.match(r"^[Pp]age\s+\d+\s*(of\s+\d+)?\.?$", stripped):
            continue

        # Skip copyright / rights notice lines
        if "©" in stripped or "All rights reserved" in stripped.lower():
            continue
        if "notice of rights" in stripped.lower():
            continue
        if "terms-and-conditions" in stripped.lower():
            continue
        if "subject to notice of rights" in stripped.lower():
            continue

        # Skip repeated document title lines (common NICE pattern)
        # These are short lines that match the document title exactly
        if stripped.startswith("Depression in adults") and len(stripped) < 80:
            continue

        # Skip URL-only lines (footers often have standalone URLs)
        if re.match(r"^https?://\S+$", stripped):
            continue

        cleaned_lines.append(line)

    # Collapse runs of 3+ blank lines into 2 (preserves paragraph structure)
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def load_pdf(pdf_path: Path) -> list[DocumentPage]:
    """
    Extract text from a single PDF file, returning one DocumentPage per page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of DocumentPage objects, one per page with extractable text.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If pymupdf cannot open the file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Loading PDF: %s", pdf_path.name)

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF {pdf_path.name}: {e}") from e

    pages: list[DocumentPage] = []
    empty_page_count = 0

    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        stripped = text.strip()

        if len(stripped) < 50:
            # Skip near-empty pages (cover pages, blank pages, full-page images)
            empty_page_count += 1
            continue

        cleaned = clean_page_text(stripped)
        if len(cleaned) < 50:
            empty_page_count += 1
            continue

        pages.append(
            DocumentPage(
                text=cleaned,
                metadata={
                    "source": pdf_path.name,
                    "page": page_num + 1,  # 1-indexed for human readability
                },
            )
        )

    doc.close()

    logger.info(
        "Loaded %d pages from %s (skipped %d empty pages)",
        len(pages),
        pdf_path.name,
        empty_page_count,
    )

    return pages


def load_guidelines(guidelines_dir: Path) -> list[DocumentPage]:
    """
    Load all PDF files from the guidelines directory.

    Args:
        guidelines_dir: Path to the directory containing guideline PDFs.

    Returns:
        Combined list of DocumentPage objects from all PDFs.

    Raises:
        FileNotFoundError: If the directory does not exist or contains no PDFs.
    """
    guidelines_dir = Path(guidelines_dir)
    if not guidelines_dir.exists():
        raise FileNotFoundError(f"Guidelines directory not found: {guidelines_dir}")

    pdf_files = sorted(guidelines_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {guidelines_dir}")

    logger.info("Found %d PDF file(s) in %s", len(pdf_files), guidelines_dir)

    all_pages: list[DocumentPage] = []
    for pdf_path in pdf_files:
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)

    logger.info(
        "Total: %d pages loaded from %d document(s)",
        len(all_pages),
        len(pdf_files),
    )

    return all_pages
