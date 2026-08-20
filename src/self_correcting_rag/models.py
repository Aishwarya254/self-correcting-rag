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
