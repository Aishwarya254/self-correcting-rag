"""Tests for core document models."""

import pytest
from self_correcting_rag.models import DocumentPage


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
