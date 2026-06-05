"""Tests for the PDF loader module."""

from pathlib import Path

import pytest

from clinical_rag.ingest.loader import DocumentPage, load_guidelines, load_pdf

# It tests against real guideline PDFs if available, otherwise skip.
# This keeps tests meaningful (testing real extraction) without breaking
# CI where PDFs may not be present.


GUIDELINES_DIR = Path("data/guidelines")


def _has_guidelines() -> bool:
    """Check if guideline PDFs are present."""
    return GUIDELINES_DIR.exists() and any(GUIDELINES_DIR.glob("*.pdf"))


@pytest.mark.skipif(not _has_guidelines(), reason="No guideline PDFs in data/guidelines/")
class TestLoadGuidelines:
    """Tests that require actual PDF files."""

    def test_load_guidelines_returns_pages(self):
        """Should return a non-empty list of DocumentPage objects."""
        pages = load_guidelines(GUIDELINES_DIR)
        assert len(pages) > 0
        assert all(isinstance(p, DocumentPage) for p in pages)

    def test_pages_have_required_metadata(self):
        """Each page should have 'source' and 'page' in metadata."""
        pages = load_guidelines(GUIDELINES_DIR)
        for page in pages[:5]:  # check first 5
            assert "source" in page.metadata
            assert "page" in page.metadata
            assert page.metadata["source"].endswith(".pdf")
            assert isinstance(page.metadata["page"], int)
            assert page.metadata["page"] >= 1

    def test_pages_have_meaningful_text(self):
        """Extracted text should have substantial content (not just headers/numbers)."""
        pages = load_guidelines(GUIDELINES_DIR)
        for page in pages[:5]:
            assert len(page.text) >= 50

    def test_load_single_pdf(self):
        """Should load pages from a single PDF file."""
        pdf_files = list(GUIDELINES_DIR.glob("*.pdf"))
        pages = load_pdf(pdf_files[0])
        assert len(pages) > 0
        assert all(p.metadata["source"] == pdf_files[0].name for p in pages)


class TestLoaderEdgeCases:
    """Tests for error handling — no PDF files needed."""

    def test_load_pdf_file_not_found(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_pdf(Path("nonexistent/fake.pdf"))

    def test_load_guidelines_dir_not_found(self):
        """Should raise FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            load_guidelines(Path("nonexistent/directory"))
