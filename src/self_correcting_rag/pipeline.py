"""Orchestrate PDF ingestion, chunking, embedding, and retrieval."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from self_correcting_rag.chunking import chunk_pages
from self_correcting_rag.models import DocumentChunk, DocumentPage
from self_correcting_rag.pdf_ingestion import extract_pdf_pages
from self_correcting_rag.retrieval import (
    Embedder,
    InMemoryVectorIndex,
    SearchResult,
)


@dataclass(frozen=True, slots=True)
class IndexedBook:
    """An indexed book ready for semantic search."""

    pages: tuple[DocumentPage, ...]
    chunks: tuple[DocumentChunk, ...]
    _index: InMemoryVectorIndex = field(repr=False)

    def search(
        self,
        query: str,
        *,
        limit: int = 4,
    ) -> list[SearchResult]:
        """Retrieve chunks relevant to a question."""

        return self._index.search(query, limit=limit)

    def save_index(self, path: str | Path) -> None:
        """Save the book's searchable vector index."""

        self._index.save(path)


def index_pdf(
    path: str | Path,
    embedder: Embedder,
    *,
    chunk_size: int = 300,
    overlap: int = 50,
) -> IndexedBook:
    """Convert a PDF into a searchable in-memory book index."""

    pages: Sequence[DocumentPage] = extract_pdf_pages(path)
    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    vector_index = InMemoryVectorIndex(embedder)
    vector_index.add(chunks)

    return IndexedBook(
        pages=tuple(pages),
        chunks=tuple(chunks),
        _index=vector_index,
    )
