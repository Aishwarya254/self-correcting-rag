"""Core data models used throughout the RAG pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """Text and source metadata extracted from one PDF page."""

    source: str
    page_number: int
    text: str

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must be positive")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A retrieval-sized piece of text with source page metadata."""

    source: str
    chunk_index: int
    start_page: int
    end_page: int
    text: str

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative")

        if self.start_page < 1:
            raise ValueError("start_page must be positive")

        if self.end_page < self.start_page:
            raise ValueError("end_page cannot precede start_page")
