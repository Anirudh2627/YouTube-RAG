"""Vector store abstraction.

The rest of the app only ever talks to this interface, so swapping Chroma →
Qdrant / FAISS / pgvector means writing one new class (and pointing a config
value at it), not touching retrieval or API code.

Contract:
  * vectors are L2-normalized float32; `score` is cosine similarity
  * metadata round-trips untouched
  * `where` filters use simple equality on metadata fields
    (e.g. {"video_id": "abc"}); backends translate to native syntax
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VectorHit:
    id: str
    score: float                       # cosine similarity
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    backend: str = "base"

    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str],
            metadatas: list[dict[str, Any]]) -> None: ...

    @abstractmethod
    def query(self, vector: np.ndarray, top_k: int = 10,
              where: dict[str, Any] | None = None) -> list[VectorHit]: ...

    @abstractmethod
    def delete_video(self, video_id: str) -> None: ...

    @abstractmethod
    def video_ids(self) -> list[str]: ...

    @abstractmethod
    def count(self, video_id: str | None = None) -> int: ...

    @abstractmethod
    def get_all_documents(self) -> list[tuple[str, str, str]]:
        """All stored chunks as (chunk_id, video_id, text) — used to build
        the BM25 index without re-ingesting."""

    @abstractmethod
    def get_metadata(self, chunk_id: str) -> dict[str, Any] | None: ...


def get_vector_store(backend: str, persist_dir: str | None = None,
                     collection: str = "youtube_rag") -> VectorStore:
    if backend == "chroma":
        from app.vectorstore.chroma_store import ChromaVectorStore
        return ChromaVectorStore(persist_dir=persist_dir, collection=collection)
    if backend == "memory":
        from app.vectorstore.memory_store import MemoryVectorStore
        return MemoryVectorStore(persist_path=persist_dir)
    raise ValueError(f"unknown vectorstore backend: {backend!r}")
