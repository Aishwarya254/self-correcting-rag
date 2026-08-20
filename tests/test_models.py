"""Tests for core document models."""

import pytest

from self_correcting_rag.models import DocumentChunk, DocumentPage


def test_document_page_stores_source_page_number_and_text() -> None:
    page = DocumentPage(
        source="example.pdf",
        page_number=3,
        text="Example page content.",
    )

    assert page.source == "example.pdf"
    assert page.page_number == 3
    assert page.text == "Example page content."


def test_document_page_rejects_non_positive_page_number() -> None:
    with pytest.raises(ValueError, match="page_number must be positive"):
        DocumentPage(
            source="example.pdf",
            page_number=0,
            text="Invalid page.",
        )


def test_document_chunk_stores_content_and_page_range() -> None:
    chunk = DocumentChunk(
        source="example.pdf",
        chunk_index=0,
        start_page=3,
        end_page=4,
        text="Content spanning two pages.",
    )

    assert chunk.source == "example.pdf"
    assert chunk.chunk_index == 0
    assert chunk.start_page == 3
    assert chunk.end_page == 4
    assert chunk.text == "Content spanning two pages."


def test_document_chunk_rejects_invalid_page_range() -> None:
    with pytest.raises(ValueError, match="end_page cannot precede start_page"):
        DocumentChunk(
            source="example.pdf",
            chunk_index=0,
            start_page=4,
            end_page=3,
            text="Invalid range.",
        )


def test_document_chunk_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="chunk_index cannot be negative"):
        DocumentChunk(
            source="example.pdf",
            chunk_index=-1,
            start_page=1,
            end_page=1,
            text="Invalid chunk.",
        )


def test_document_chunk_rejects_non_positive_start_page() -> None:
    with pytest.raises(ValueError, match="start_page must be positive"):
        DocumentChunk(
            source="example.pdf",
            chunk_index=0,
            start_page=0,
            end_page=1,
            text="Invalid chunk.",
        )
