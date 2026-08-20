"""Generate normalized semantic embeddings for retrieval."""

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class SentenceTransformerEmbedder:
    """Embed text using a configurable Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return normalized embedding vectors for the supplied texts."""

        text_list = list(texts)

        if not text_list:
            return []

        vectors = self._model.encode(
            text_list,
            normalize_embeddings=True,
        )

        return vectors.tolist()
