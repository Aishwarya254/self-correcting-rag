"""Store and retrieve document chunks using semantic similarity."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from self_correcting_rag.content_detection import (
    is_navigation_query,
    is_table_of_contents,
)
from self_correcting_rag.models import DocumentChunk

_TABLE_OF_CONTENTS_PENALTY = 0.15


class Embedder(Protocol):
    """Interface required by semantic retrieval."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Convert text into normalized embedding vectors."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieved chunk and its semantic similarity score."""

    chunk: DocumentChunk
    score: float


class InMemoryVectorIndex:
    """A simple vector index for local development and testing."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        """Embed and store document chunks."""

        chunk_list = list(chunks)

        if not chunk_list:
            return

        vectors = self._embedder.embed([chunk.text for chunk in chunk_list])

        if len(vectors) != len(chunk_list):
            raise ValueError("embedder returned an unexpected number of vectors")

        self._chunks.extend(chunk_list)
        self._vectors.extend(vectors)

    def search(self, query: str, *, limit: int = 4) -> list[SearchResult]:
        """Return chunks with the highest dot-product similarity."""

        if limit < 1:
            raise ValueError("limit must be positive")

        if not self._chunks:
            return []

        query_vectors = self._embedder.embed([query])

        if len(query_vectors) != 1:
            raise ValueError("embedder did not return one query vector")

        query_vector = query_vectors[0]
        results = []

        for chunk, vector in zip(self._chunks, self._vectors, strict=True):
            if len(vector) != len(query_vector):
                raise ValueError("embedding dimensions do not match")

            score = sum(
                stored_value * query_value
                for stored_value, query_value in zip(
                    vector,
                    query_vector,
                    strict=True,
                )
            )

            if is_table_of_contents(chunk.text) and not is_navigation_query(query):
                score -= _TABLE_OF_CONTENTS_PENALTY

            results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:limit]
