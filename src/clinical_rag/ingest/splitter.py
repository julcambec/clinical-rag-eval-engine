"""
Text splitting for clinical guideline documents.

Splits DocumentPage objects into smaller chunks with configurable size and
overlap, preserving metadata and adding chunk-level tracking info. Uses
LangChain's RecursiveCharacterTextSplitter, which tries to split along
natural boundaries (paragraphs, then sentences, then words) before
resorting to mid-word splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from clinical_rag.config import load_settings
from clinical_rag.ingest.loader import DocumentPage
from clinical_rag.ops.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """
    A chunk of text ready for embedding, with full provenance metadata.

    Attributes:
        text: The chunk's text content.
        metadata: Provenance info including source document, page number,
                  chunk index within the page, and character count.
        chunk_id: Globally unique identifier in the format
                  '{source}::p{page}::c{chunk_index}'.
    """

    text: str
    metadata: dict[str, str | int] = field(default_factory=dict)
    chunk_id: str = ""


def create_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with the given or config-default parameters.

    Args:
        chunk_size: Maximum characters per chunk. Defaults to config value.
        chunk_overlap: Overlap between consecutive chunks. Defaults to config value.
        separators: Split boundary characters. Defaults to config value.

    Returns:
        A configured RecursiveCharacterTextSplitter.
    """
    settings = load_settings()
    cfg = settings.retrieval.chunking

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or cfg.chunk_size,
        chunk_overlap=chunk_overlap or cfg.chunk_overlap,
        separators=separators or cfg.separators,
        length_function=len,
        is_separator_regex=False,
    )


def split_pages(
    pages: list[DocumentPage],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """
    Split a list of DocumentPages into TextChunks.

    Each chunk inherits its parent page's metadata and gets additional
    chunk-level metadata (chunk index within page, character count, and
    a globally unique chunk_id for deduplication).

    Args:
        pages: List of DocumentPage objects from the loader.
        chunk_size: Override for chunk size (uses config default if None).
        chunk_overlap: Override for chunk overlap (uses config default if None).

    Returns:
        List of TextChunk objects ready for indexing.
    """
    splitter = create_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks: list[TextChunk] = []
    total_input_chars = 0

    for page in pages:
        total_input_chars += len(page.text)
        split_texts = splitter.split_text(page.text)

        for chunk_idx, text in enumerate(split_texts):
            source = page.metadata.get("source", "unknown")
            page_num = page.metadata.get("page", 0)

            chunk_id = f"{source}::p{page_num}::c{chunk_idx}"

            chunk = TextChunk(
                text=text,
                metadata={
                    **page.metadata,  # inherit source, page
                    "chunk_index": chunk_idx,
                    "chunk_char_count": len(text),
                },
                chunk_id=chunk_id,
            )
            chunks.append(chunk)

    logger.info(
        "Split %d pages (%d chars) into %d chunks",
        len(pages),
        total_input_chars,
        len(chunks),
    )

    if chunks:
        sizes = [int(c.metadata["chunk_char_count"]) for c in chunks]
        logger.info(
            "Chunk sizes: min= %d, max= %d, mean= %d",
            min(sizes),
            max(sizes),
            sum(sizes) // len(sizes),
        )

    return chunks
