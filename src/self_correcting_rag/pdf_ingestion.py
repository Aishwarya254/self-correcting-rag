"""Extract page-level text and metadata from PDF documents."""

from pathlib import Path

from pypdf import PdfReader

from self_correcting_rag.models import DocumentPage
from self_correcting_rag.text_cleaning import clean_extracted_text


def extract_pdf_pages(path: str | Path) -> list[DocumentPage]:
    """Extract every page from a PDF while preserving its source and page number."""

    pdf_path = Path(path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    reader = PdfReader(pdf_path)

    return [
        DocumentPage(
            source=str(pdf_path),
            page_number=page_number,
            text=clean_extracted_text(page.extract_text() or ""),
        )
        for page_number, page in enumerate(reader.pages, start=1)
    ]
