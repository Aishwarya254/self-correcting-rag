"""Tests for page-aware document chunking."""

import pytest

from self_correcting_rag.chunking import chunk_pages
from self_correcting_rag.models import DocumentChunk, DocumentPage


def test_chunk_pages_preserves_overlap_and_page_ranges() -> None:
    pages = [
        DocumentPage(
            source="book.pdf",
            page_number=1,
            text="one two three four five",
        ),
        DocumentPage(
            source="book.pdf",
            page_number=2,
            text="six seven eight",
        ),
    ]

    chunks = chunk_pages(pages, chunk_size=4, overlap=1)

    assert chunks == [
        DocumentChunk(
            source="book.pdf",
            chunk_index=0,
            start_page=1,
            end_page=1,
            text="one two three four",
        ),
        DocumentChunk(
            source="book.pdf",
            chunk_index=1,
            start_page=1,
            end_page=2,
            text="four five six seven",
        ),
        DocumentChunk(
            source="book.pdf",
            chunk_index=2,
            start_page=2,
            end_page=2,
            text="seven eight",
        ),
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (0, 0, "chunk_size must be positive"),
        (4, -1, "overlap cannot be negative"),
        (4, 4, "overlap must be smaller than chunk_size"),
    ],
)
def test_chunk_pages_rejects_invalid_settings(
    chunk_size: int,
    overlap: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_pages([], chunk_size=chunk_size, overlap=overlap)


def test_chunk_pages_ignores_pages_without_text() -> None:
    pages = [
        DocumentPage(
            source="book.pdf",
            page_number=1,
            text="",
        )
    ]

    assert chunk_pages(pages, chunk_size=4, overlap=1) == []


def test_chunk_pages_rejects_mixed_sources() -> None:
    pages = [
        DocumentPage(
            source="first.pdf",
            page_number=1,
            text="First source.",
        ),
        DocumentPage(
            source="second.pdf",
            page_number=1,
            text="Second source.",
        ),
    ]

    with pytest.raises(ValueError, match="all pages must come from the same source"):
        chunk_pages(pages, chunk_size=4, overlap=1)
