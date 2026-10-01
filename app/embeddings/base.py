"""Embedder interface + factory."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):

    dim: int
    model_name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return shape (len(texts), dim)."""

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def get_embedder(provider: str, model_name: str, dim: int = 384,
                 batch_size: int = 32) -> Embedder:
    if provider == "hashing":
        from app.embeddings.hashing import HashingEmbedder
        return HashingEmbedder(dim=dim)
    if provider == "sentence-transformers":
        from app.embeddings.st_embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model_name, batch_size=batch_size)
    raise ValueError(f"unknown embedding provider: {provider!r}")
