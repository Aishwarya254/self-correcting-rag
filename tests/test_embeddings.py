"""Tests for embedding text with a Sentence Transformers model."""

import pytest

from self_correcting_rag import embeddings


class FakeVectors:
    """Predictable embedding output used by the test."""

    def tolist(self) -> list[list[float]]:
        return [
            [1.0, 0.0],
            [0.0, 1.0],
        ]


def test_embedder_loads_model_and_normalizes_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            calls["model_name"] = model_name
            calls["device"] = device

        def encode(
            self,
            texts: list[str],
            *,
            normalize_embeddings: bool,
        ) -> FakeVectors:
            calls["texts"] = texts
            calls["normalize_embeddings"] = normalize_embeddings
            return FakeVectors()

    monkeypatch.setattr(
        embeddings,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = embeddings.SentenceTransformerEmbedder(
        model_name="test-model",
        device="cpu",
    )

    vectors = embedder.embed(["first text", "second text"])

    assert vectors == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert calls == {
        "model_name": "test-model",
        "device": "cpu",
        "texts": ["first text", "second text"],
        "normalize_embeddings": True,
    }


def test_embedder_handles_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            pass

        def encode(self, texts: list[str], *, normalize_embeddings: bool) -> None:
            raise AssertionError("encode should not be called for empty input")

    monkeypatch.setattr(
        embeddings,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = embeddings.SentenceTransformerEmbedder()

    assert embedder.embed([]) == []
