"""Chroma-backed vector store (default backend).

Chroma is embedded (no server needed) and persists to disk, which pairs well
with our per-video cache: one persistent collection holds all ingested videos,
and `video_id` metadata filters scope queries to a single video or the whole
collection.

Note: we pass *precomputed* embeddings (from our Embedder) instead of using
Chroma's built-in embedding functions — the embedding model must stay a
pipeline-level decision, configurable and testable, not a database detail.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.utils.logging import get_logger
from app.vectorstore.base import VectorHit, VectorStore

log = get_logger(__name__)


class ChromaVectorStore(VectorStore):
    backend = "chroma"

    def __init__(self, persist_dir: str | Path | None = None,
                 collection: str = "youtube_rag"):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        if persist_dir:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        else:
            self._client = chromadb.EphemeralClient(
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------
    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str],
            metadatas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        # Chroma rejects None values in metadata; stringify defensively.
        metas = [{k: ("" if v is None else v) for k, v in m.items()} for m in metadatas]
        batch = 500
        for i in range(0, len(ids), batch):
            self._col.upsert(
                ids=ids[i:i + batch],
                embeddings=vectors[i:i + batch].tolist(),
                documents=texts[i:i + batch],
                metadatas=metas[i:i + batch],
            )

    def query(self, vector: np.ndarray, top_k: int = 10,
              where: dict[str, Any] | None = None) -> list[VectorHit]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [np.asarray(vector, dtype=np.float32).tolist()],
            "n_results": min(top_k, max(self.count(), 1)) or top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        res = self._col.query(**kwargs)
        hits: list[VectorHit] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            # cosine distance ∈ [0, 2] → similarity ∈ [-1, 1]
            hits.append(VectorHit(id=_id, score=1.0 - float(dist),
                                  text=doc or "", metadata=dict(meta or {})))
        return hits

    def delete_video(self, video_id: str) -> None:
        self._col.delete(where={"video_id": video_id})

    def video_ids(self) -> list[str]:
        seen: set[str] = set()
        data = self._col.get(include=["metadatas"])
        for meta in data.get("metadatas") or []:
            vid = (meta or {}).get("video_id")
            if vid:
                seen.add(str(vid))
        return sorted(seen)

    def count(self, video_id: str | None = None) -> int:
        if video_id is None:
            return self._col.count()
        return len(self._col.get(where={"video_id": video_id}, include=[]).get("ids") or [])

    def get_all_documents(self) -> list[tuple[str, str, str]]:
        data = self._col.get(include=["documents", "metadatas"])
        out = []
        for _id, doc, meta in zip(data.get("ids") or [],
                                  data.get("documents") or [],
                                  data.get("metadatas") or []):
            out.append((_id, str((meta or {}).get("video_id", "")), doc or ""))
        return out

    def get_metadata(self, chunk_id: str) -> dict[str, Any] | None:
        data = self._col.get(ids=[chunk_id], include=["metadatas"])
        metas = data.get("metadatas") or []
        return dict(metas[0]) if metas else None
