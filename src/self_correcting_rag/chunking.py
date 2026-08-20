"""Split cleaned PDF pages into overlapping retrieval chunks."""

from collections.abc import Sequence

from self_correcting_rag.models import DocumentChunk, DocumentPage


def chunk_pages(
    pages: Sequence[DocumentPage],
    *,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """Split pages into word-based chunks while retaining source page ranges."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not pages:
        return []

    sources = {page.source for page in pages}

    if len(sources) != 1:
        raise ValueError("all pages must come from the same source")

    words_with_pages: list[tuple[str, int]] = []

    for page in pages:
        words_with_pages.extend((word, page.page_number) for word in page.text.split())

    if not words_with_pages:
        return []

    source = pages[0].source
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words_with_pages), step):
        window = words_with_pages[start : start + chunk_size]

        chunks.append(
            DocumentChunk(
                source=source,
                chunk_index=len(chunks),
                start_page=window[0][1],
                end_page=window[-1][1],
                text=" ".join(word for word, _ in window),
            )
        )

        if start + chunk_size >= len(words_with_pages):
            break

    return chunks
