"""Tests for PDF ingestion."""

from pathlib import Path

import pytest

from self_correcting_rag import pdf_ingestion
from self_correcting_rag.models import DocumentPage


class FakePage:
    """A PDF page used to test extraction without a real book."""

    def __init__(self, text: str | None) -> None:
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


class FakeReader:
    """A predictable replacement for pypdf.PdfReader."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.pages = [
            FakePage("First page."),
            FakePage(None),
        ]


def test_extract_pdf_pages_preserves_page_numbers_and_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "book.pdf"
    pdf_path.touch()
    monkeypatch.setattr(pdf_ingestion, "PdfReader", FakeReader)

    pages = pdf_ingestion.extract_pdf_pages(pdf_path)

    assert pages == [
        DocumentPage(
            source=str(pdf_path),
            page_number=1,
            text="First page.",
        ),
        DocumentPage(
            source=str(pdf_path),
            page_number=2,
            text="",
        ),
    ]
