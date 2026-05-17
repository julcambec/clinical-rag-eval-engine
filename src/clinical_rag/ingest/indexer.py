"""
Indexing pipeline for clinical guideline chunks.

Stores text chunks in two complementary indexes:
- ChromaDB: dense vector store for embedding-based semantic search.
- BM25: sparse keyword index for term-matching retrieval.

The two indexes are used together via hybrid retrieval (reciprocal rank fusion)
during the retrieval stage.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from clinical_rag.config import PROJECT_ROOT, load_settings
from clinical_rag.ingest.loader import load_guidelines
from clinical_rag.ingest.splitter import TextChunk, split_pages
from clinical_rag.ops.logging import get_logger, setup_logging

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


def get_chroma_client(persist_dir: Path | None = None) -> chromadb.ClientAPI:
    """
    Create a persistent ChromaDB client.

    Args:
        persist_dir: Directory to persist the database. Defaults to data/chroma_db/.

    Returns:
        A ChromaDB PersistentClient.
    """
    persist_dir = persist_dir or CHROMA_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client


def index_chunks_chroma(
    chunks: list[TextChunk],
    collection_name: str | None = None,
    persist_dir: Path | None = None,
) -> int:
    """
    Index text chunks into ChromaDB with OpenAI embeddings.

    ChromaDB handles embedding generation internally when configured with
    an embedding function. We pass the raw text and it embeds + stores.

    However, for better control and consistency with our config, I use
    ChromaDB's default embedding function first, then plan to switch to
    OpenAI embeddings when the retrieval layer is built. For now, ChromaDB's
    built-in all-MiniLM-L6-v2 gets us a good working index.

    Args:
        chunks: List of TextChunk objects to index.
        collection_name: Name of the ChromaDB collection.
        persist_dir: Where to persist the database.

    Returns:
        Number of chunks indexed.
    """
    settings = load_settings()
    collection_name = collection_name or settings.retrieval.dense.collection_name

    client = get_chroma_client(persist_dir)

    # Delete existing collection to ensure clean re-indexing
    try:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s' for clean re-index", collection_name)
    except (ValueError, chromadb.errors.NotFoundError):
        pass  # Collection doesn't exist yet; that's fine

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Clinical guideline chunks for RAG retrieval"},
    )

    # ChromaDB has a batch size limit; add in batches of 500
    batch_size = 500
    total_indexed = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[
                {k: str(v) for k, v in chunk.metadata.items()}  # ChromaDB needs str values
                for chunk in batch
            ],
        )

        total_indexed += len(batch)
        logger.info(
            "Indexed batch %d/%d (%d chunks)",
            (i // batch_size) + 1,
            (len(chunks) + batch_size - 1) // batch_size,
            total_indexed,
        )

    logger.info(
        "ChromaDB indexing complete: %d chunks in collection '%s'",
        total_indexed,
        collection_name,
    )

    return total_indexed


# ---------------
# BM25 indexing
# ---------------


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercasing tokenizer for BM25.

    For large-scale deployment, I'd use another tokenizer (e.g., spaCy, NLTK),
    but for this first prototype this is sufficient and dependency-light.
    """
    return text.lower().split()


def index_chunks_bm25(
    chunks: list[TextChunk],
    persist_dir: Path | None = None,
) -> int:
    """
    Build and persist a BM25 keyword index from text chunks.

    Saves two files:
    - bm25_index.pkl: The serialized BM25Okapi model.
    - bm25_chunks.json: The chunk texts and metadata, aligned by index
      so that BM25 result indices map back to the correct chunks.

    Args:
        chunks: List of TextChunk objects to index.
        persist_dir: Where to save the index files.

    Returns:
        Number of chunks indexed.
    """
    persist_dir = persist_dir or BM25_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building BM25 index from %d chunks...", len(chunks))

    # Tokenize all chunks
    tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]

    # Build the BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Save the BM25 model
    bm25_path = persist_dir / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # Save chunk data alongside (we need this to map BM25 scores back to chunks)
    chunks_data = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": {k: str(v) for k, v in chunk.metadata.items()},
        }
        for chunk in chunks
    ]
    chunks_path = persist_dir / "bm25_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    logger.info(
        "BM25 indexing complete: %d chunks saved to %s",
        len(chunks),
        persist_dir,
    )

    return len(chunks)


# ------------------------
# Full ingestion pipeline
# ------------------------


def run_ingestion(
    guidelines_dir: Path | None = None,
    chroma_dir: Path | None = None,
    bm25_dir: Path | None = None,
) -> dict[str, int]:
    """
    Run the complete ingestion pipeline: load → split → index.

    This is the main entry point called by `make ingest`.

    Args:
        guidelines_dir: Path to the clinical guideline PDFs.
        chroma_dir: Where to persist ChromaDB.
        bm25_dir: Where to persist the BM25 index.

    Returns:
        Dictionary with pipeline stats (pages_loaded, chunks_created,
        chunks_indexed_chroma, chunks_indexed_bm25).
    """
    guidelines_dir = guidelines_dir or GUIDELINES_DIR

    logger.info("=" * 60)
    logger.info("Starting ingestion pipeline")
    logger.info("=" * 60)

    # Stage 1: Load PDFs
    logger.info("Stage 1/3: Loading PDFs from %s", guidelines_dir)
    pages = load_guidelines(guidelines_dir)

    # Stage 2: Split into chunks
    logger.info("Stage 2/3: Splitting pages into chunks")
    chunks = split_pages(pages)

    # Stage 3: Index into both stores
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


# ----------------
# CLI entry point
# ----------------


def main() -> None:
    """CLI entry point for `make ingest` / `clinical-rag-ingest`."""
    setup_logging(level="INFO", log_format="simple")
    run_ingestion()


if __name__ == "__main__":
    main()
