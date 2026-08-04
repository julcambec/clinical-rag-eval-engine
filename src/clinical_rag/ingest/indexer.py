"""
Indexing pipeline for clinical guideline chunks.

Stores text chunks in two complementary indexes:
- ChromaDB (via LangChain's Chroma wrapper): dense vector store using the
  configured local embeddings (BAAI/bge-small-en-v1.5), for semantic search.
- BM25: sparse keyword index for term-matching retrieval.

The two are combined via reciprocal rank fusion at retrieval time.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from clinical_rag.config import PROJECT_ROOT, load_settings
from clinical_rag.ingest.loader import load_guidelines
from clinical_rag.ingest.splitter import TextChunk, split_pages
from clinical_rag.ops.logging import get_logger, setup_logging
from clinical_rag.providers import get_embeddings

logger = get_logger(__name__)

# ---------------------------------
# Paths, derived from project root
# ---------------------------------

GUIDELINES_DIR = PROJECT_ROOT / "data" / "guidelines"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
BM25_DIR = PROJECT_ROOT / "data" / "bm25_index"


# ------------------
# ChromaDB indexing
# ------------------


def index_chunks_chroma(
    chunks: list[TextChunk],
    collection_name: str | None = None,
    persist_dir: Path | None = None,
) -> int:
    """
    Index text chunks into ChromaDB using the configured local embeddings.

    We use LangChain's Chroma wrapper (not the raw client) so that the SAME
    embeddings object is reused at retrieval time, and so documents/queries are
    embedded consistently. The collection is created with cosine distance, which
    pairs well with normalized bge vectors.
    """
    settings = load_settings()
    collection_name = collection_name or settings.retrieval.dense.collection_name
    persist_dir = persist_dir or CHROMA_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(settings)

    # Clean re-index: drop any existing collection first (avoids stale/mixed vectors).
    existing = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    try:
        existing.delete_collection()
        logger.info("Deleted existing collection '%s' for clean re-index", collection_name)
    except Exception:  # noqa: BLE001 - collection may simply not exist yet
        pass

    documents = [
        Document(
            page_content=chunk.text,
            metadata={**chunk.metadata, "chunk_id": chunk.chunk_id},
        )
        for chunk in chunks
    ]
    ids = [chunk.chunk_id for chunk in chunks]

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )

    logger.info(
        "ChromaDB indexing complete: %d chunks in collection '%s' (embeddings: %s)",
        len(documents),
        collection_name,
        settings.retrieval.embeddings.model,
    )
    return len(documents)


# ---------------
# BM25 indexing
# ---------------


def _tokenize(text: str) -> list[str]:
    """
    Whitespace + lowercase tokenizer for BM25.
    NOTE: retrieval/sparse.py MUST tokenize queries identically.
    """
    return text.lower().split()


def index_chunks_bm25(
    chunks: list[TextChunk],
    persist_dir: Path | None = None,
) -> int:
    """Build and persist a BM25 keyword index (model + aligned chunk data)."""
    persist_dir = persist_dir or BM25_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building BM25 index from %d chunks...", len(chunks))

    tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(persist_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    chunks_data = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": {k: str(v) for k, v in chunk.metadata.items()},
        }
        for chunk in chunks
    ]
    with open(persist_dir / "bm25_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    logger.info("BM25 indexing complete: %d chunks saved to %s", len(chunks), persist_dir)
    return len(chunks)


# ------------------------
# Full ingestion pipeline
# ------------------------


def run_ingestion(
    guidelines_dir: Path | None = None,
    chroma_dir: Path | None = None,
    bm25_dir: Path | None = None,
) -> dict[str, int]:
    """Run the complete ingestion pipeline: load -> split -> index."""
    guidelines_dir = guidelines_dir or GUIDELINES_DIR

    logger.info("=" * 60)
    logger.info("Starting ingestion pipeline")
    logger.info("=" * 60)

    logger.info("Stage 1/3: Loading PDFs from %s", guidelines_dir)
    pages = load_guidelines(guidelines_dir)

    logger.info("Stage 2/3: Splitting pages into chunks")
    chunks = split_pages(pages)

    logger.info("Stage 3/3: Indexing chunks")
    chroma_count = index_chunks_chroma(chunks, persist_dir=chroma_dir)
    bm25_count = index_chunks_bm25(chunks, persist_dir=bm25_dir)

    stats = {
        "pages_loaded": len(pages),
        "chunks_created": len(chunks),
        "chunks_indexed_chroma": chroma_count,
        "chunks_indexed_bm25": bm25_count,
    }

    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    for key, value in stats.items():
        logger.info("  %s: %d", key, value)
    logger.info("=" * 60)

    return stats


def main() -> None:
    """CLI entry point for `make ingest` / `clinical-rag-ingest`."""
    setup_logging(level="INFO", log_format="simple")
    run_ingestion()


if __name__ == "__main__":
    main()
