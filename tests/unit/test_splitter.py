"""Tests for the text splitting module."""

from clinical_rag.ingest.loader import DocumentPage
from clinical_rag.ingest.splitter import TextChunk, create_splitter, split_pages


class TestCreateSplitter:
    """Verify splitter creation with config and overrides."""

    def test_creates_splitter_with_defaults(self):
        """Splitter should be created using config defaults."""
        splitter = create_splitter()
        assert splitter._chunk_size == 1000
        assert splitter._chunk_overlap == 200

    def test_creates_splitter_with_overrides(self):
        """Explicit parameters should override config defaults."""
        splitter = create_splitter(chunk_size=500, chunk_overlap=50)
        assert splitter._chunk_size == 500
        assert splitter._chunk_overlap == 50


class TestSplitPages:
    """Verify page splitting behavior."""

    def _make_page(self, text: str, source: str = "test.pdf", page: int = 1) -> DocumentPage:
        """Helper to create a DocumentPage for testing."""
        return DocumentPage(text=text, metadata={"source": source, "page": page})

    def test_short_text_produces_single_chunk(self):
        """Text shorter than chunk_size should produce exactly one chunk."""
        page = self._make_page("This is a short document about depression treatment.")
        chunks = split_pages([page], chunk_size=1000, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0].text == page.text

    def test_long_text_produces_multiple_chunks(self):
        """Text longer than chunk_size should be split into multiple chunks."""
        # Create text that is clearly longer than chunk_size=100
        long_text = "Sentence about guidelines. " * 50  # ~1350 chars
        page = self._make_page(long_text)
        chunks = split_pages([page], chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunks_preserve_source_metadata(self):
        """Each chunk should carry forward its parent page's metadata."""
        page = self._make_page("Some text.", source="mhgap.pdf", page=5)
        chunks = split_pages([page], chunk_size=1000)
        assert chunks[0].metadata["source"] == "mhgap.pdf"
        assert chunks[0].metadata["page"] == 5

    def test_chunks_have_chunk_level_metadata(self):
        """Each chunk should have chunk_index and chunk_char_count."""
        page = self._make_page("A" * 500)
        chunks = split_pages([page], chunk_size=1000)
        assert "chunk_index" in chunks[0].metadata
        assert "chunk_char_count" in chunks[0].metadata
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["chunk_char_count"] == 500

    def test_chunk_ids_are_unique(self):
        """All chunk IDs within a batch should be unique."""
        long_text = "Word " * 500  # enough to produce multiple chunks
        pages = [
            self._make_page(long_text, source="doc_a.pdf", page=1),
            self._make_page(long_text, source="doc_b.pdf", page=1),
        ]
        chunks = split_pages(pages, chunk_size=200, chunk_overlap=20)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found"

    def test_chunk_id_format(self):
        """Chunk IDs should follow the source::page::chunk pattern."""
        page = self._make_page("Short text", source="nice.pdf", page=3)
        chunks = split_pages([page], chunk_size=1000)
        assert chunks[0].chunk_id == "nice.pdf::p3::c0"

    def test_multiple_pages_combined(self):
        """Splitting multiple pages should produce chunks from all pages."""
        pages = [
            self._make_page("Page one content.", source="doc.pdf", page=1),
            self._make_page("Page two content.", source="doc.pdf", page=2),
        ]
        chunks = split_pages(pages, chunk_size=1000)
        assert len(chunks) == 2
        assert chunks[0].metadata["page"] == 1
        assert chunks[1].metadata["page"] == 2

    def test_empty_page_list_returns_empty(self):
        """An empty input should produce an empty output."""
        chunks = split_pages([])
        assert chunks == []

    def test_overlap_creates_shared_content(self):
        """With overlap > 0, consecutive chunks from a long text should share content."""
        # Build text that will split into at least 2 chunks
        words = [f"word{i}" for i in range(200)]
        long_text = " ".join(words)
        page = self._make_page(long_text)
        chunks = split_pages([page], chunk_size=200, chunk_overlap=50)

        if len(chunks) >= 2:
            # Extract the last few whole words of chunk 0
            tail_words = chunks[0].text.split()[-3:]  # last 3 words
            tail_phrase = " ".join(tail_words)
            assert tail_phrase in chunks[1].text, (
                f"Expected overlap phrase '{tail_phrase}' in second chunk"
            )

    def test_returns_text_chunk_objects(self):
        """Output should be a list of TextChunk dataclass instances."""
        page = self._make_page("Hello world.")
        chunks = split_pages([page], chunk_size=1000)
        assert all(isinstance(c, TextChunk) for c in chunks)
