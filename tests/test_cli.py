"""Tests for the command-line interface."""

from pathlib import Path

import pytest

from self_correcting_rag import cli
from self_correcting_rag.models import DocumentChunk
from self_correcting_rag.retrieval import SearchResult


class FakeEmbedder:
    """Avoid loading a real embedding model during CLI tests."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device


class FakeIndexedBook:
    """Return predictable search results."""

    def search(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[SearchResult]:
        assert query == "How does RAG use evidence?"
        assert limit == 1

        return [
            SearchResult(
                chunk=DocumentChunk(
                    source="book.pdf",
                    chunk_index=0,
                    start_page=2,
                    end_page=3,
                    text="RAG retrieves evidence before answering.",
                ),
                score=0.95,
            )
        ]


def test_search_command_prints_ranked_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_index_pdf(
        path: str | Path,
        embedder: object,
        *,
        chunk_size: int,
        overlap: int,
    ) -> FakeIndexedBook:
        assert Path(path) == Path("book.pdf")
        assert isinstance(embedder, FakeEmbedder)
        assert chunk_size == 300
        assert overlap == 50
        return FakeIndexedBook()

    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", FakeEmbedder)
    monkeypatch.setattr(cli, "index_pdf", fake_index_pdf)

    exit_code = cli.main(
        [
            "search",
            "book.pdf",
            "How does RAG use evidence?",
            "--limit",
            "1",
            "--device",
            "cpu",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "# Retrieved Evidence" in output
    assert "**Source:** book.pdf" in output
    assert "**Pages:** 2-3" in output
    assert "**Score:** 0.9500" in output
    assert "RAG retrieves evidence before answering." in output
