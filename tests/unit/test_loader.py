"""Tests for the PDF loader module."""

from pathlib import Path

import pytest

from clinical_rag.ingest.loader import DocumentPage, load_pdf, load_guidelines


# It tests against real guideline PDFs if available, otherwise skip.
# This keeps tests meaningful (testing real extraction) without breaking
# CI where PDFs may not be present.

## TODO
