"""Tests for the complete book-indexing pipeline."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from self_correcting_rag import pipeline
from self_correcting_rag.models import DocumentPage


class FakeEmbedder:
    """Create predictable semantic groups for pipeline testing."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "cat" in text.lower() else [0.0, 1.0] for text in texts]


def test_index_pdf_builds_searchable_book(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [
        DocumentPage(
            source="book.pdf",
            page_number=1,
            text="Cats are domestic animals",
        ),
        DocumentPage(
            source="book.pdf",
            page_number=2,
            text="RAG retrieves external evidence",
        ),
    ]

    def fake_extract_pdf_pages(path: str | Path) -> list[DocumentPage]:
        assert Path(path) == Path("book.pdf")
        return pages

    monkeypatch.setattr(
        pipeline,
        "extract_pdf_pages",
        fake_extract_pdf_pages,
    )

    indexed_book = pipeline.index_pdf(
        "book.pdf",
        FakeEmbedder(),
        chunk_size=4,
        overlap=0,
    )

    results = indexed_book.search(
        "How does RAG use evidence?",
        limit=1,
    )

    assert indexed_book.pages == tuple(pages)
    assert len(indexed_book.chunks) == 2
    assert results[0].chunk.start_page == 2
    assert results[0].chunk.text == "RAG retrieves external evidence"
