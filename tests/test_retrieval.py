"""Tests for semantic chunk retrieval."""

from collections.abc import Sequence

import pytest

from self_correcting_rag.models import DocumentChunk
from self_correcting_rag.retrieval import InMemoryVectorIndex


class FakeEmbedder:
    """Return predictable vectors without loading a real model."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = {
            "Cats are small domestic animals.": [1.0, 0.0],
            "RAG retrieves evidence before answering.": [0.0, 1.0],
            "How does RAG retrieve evidence?": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


def test_vector_index_returns_most_relevant_chunk() -> None:
    animal_chunk = DocumentChunk(
        source="book.pdf",
        chunk_index=0,
        start_page=1,
        end_page=1,
        text="Cats are small domestic animals.",
    )
    rag_chunk = DocumentChunk(
        source="book.pdf",
        chunk_index=1,
        start_page=2,
        end_page=2,
        text="RAG retrieves evidence before answering.",
    )
    index = InMemoryVectorIndex(FakeEmbedder())
    index.add([animal_chunk, rag_chunk])

    results = index.search(
        "How does RAG retrieve evidence?",
        limit=1,
    )

    assert len(results) == 1
    assert results[0].chunk == rag_chunk
    assert results[0].score == 1.0


def test_empty_vector_index_returns_no_results() -> None:
    index = InMemoryVectorIndex(FakeEmbedder())

    assert index.search("How does RAG retrieve evidence?") == []


def test_vector_index_rejects_invalid_limit() -> None:
    index = InMemoryVectorIndex(FakeEmbedder())

    with pytest.raises(ValueError, match="limit must be positive"):
        index.search(
            "How does RAG retrieve evidence?",
            limit=0,
        )
