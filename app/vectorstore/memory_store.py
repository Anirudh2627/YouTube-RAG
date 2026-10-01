"""In-memory numpy vector store.

Brute-force cosine search — perfectly fine below ~100k vectors and the right
tool for tests: zero dependencies, fully deterministic, instant. Optionally
persists to an .npz + json sidecar so the offline demo survives restarts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.vectorstore.base import VectorHit, VectorStore


class MemoryVectorStore(VectorStore):
    backend = "memory"

    def __init__(self, persist_path: str | Path | None = None):
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []
        self._texts: list[str] = []
        self._metas: list[dict[str, Any]] = []
        self._persist = Path(persist_path) if persist_path else None
        if self._persist:
            self._load()

    # ------------------------------------------------------------------
    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str],
            metadatas: list[dict[str, Any]]) -> None:
        for i, _id in enumerate(ids):
            if _id in self._ids:  # upsert semantics
                self.delete_ids([_id])
            self._ids.append(_id)
            self._vecs.append(np.asarray(vectors[i], dtype=np.float32))
            self._texts.append(texts[i])
            self._metas.append(dict(metadatas[i]))
        self._save()

    def query(self, vector: np.ndarray, top_k: int = 10,
              where: dict[str, Any] | None = None) -> list[VectorHit]:
        if not self._ids:
            return []
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        mat = np.vstack(self._vecs)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        sims = mat @ q
        idxs = np.argsort(-sims)
        hits: list[VectorHit] = []
        for i in idxs:
            meta = self._metas[i]
            if where and any(str(meta.get(k, "")) != str(v) for k, v in where.items()):
                continue
            hits.append(VectorHit(id=self._ids[i], score=float(sims[i]),
                                  text=self._texts[i], metadata=dict(meta)))
            if len(hits) >= top_k:
                break
        return hits

    def delete_ids(self, ids: list[str]) -> None:
        keep = [j for j, _id in enumerate(self._ids) if _id not in set(ids)]
        self._ids = [self._ids[j] for j in keep]
        self._vecs = [self._vecs[j] for j in keep]
        self._texts = [self._texts[j] for j in keep]
        self._metas = [self._metas[j] for j in keep]
        self._save()

    def delete_video(self, video_id: str) -> None:
        self.delete_ids([_id for _id, m in zip(self._ids, self._metas)
                         if m.get("video_id") == video_id])

    def video_ids(self) -> list[str]:
        return sorted({str(m.get("video_id")) for m in self._metas if m.get("video_id")})

    def count(self, video_id: str | None = None) -> int:
        if video_id is None:
            return len(self._ids)
        return sum(1 for m in self._metas if m.get("video_id") == video_id)

    def get_all_documents(self) -> list[tuple[str, str, str]]:
        return [(_id, str(m.get("video_id", "")), t)
                for _id, m, t in zip(self._ids, self._metas, self._texts)]

    def get_metadata(self, chunk_id: str) -> dict[str, Any] | None:
        try:
            return dict(self._metas[self._ids.index(chunk_id)])
        except ValueError:
            return None

    # ------------------------------------------------------ persistence
    def _save(self) -> None:
        if not self._persist:
            return
        self._persist.mkdir(parents=True, exist_ok=True)
        if self._vecs:
            np.save(self._persist / "vectors.npy", np.vstack(self._vecs))
        (self._persist / "meta.json").write_text(json.dumps({
            "ids": self._ids, "texts": self._texts, "metas": self._metas}))

    def _load(self) -> None:
        meta_f, vec_f = self._persist / "meta.json", self._persist / "vectors.npy"
        if not (meta_f.exists() and vec_f.exists()):
            return
        data = json.loads(meta_f.read_text())
        self._ids, self._texts, self._metas = data["ids"], data["texts"], data["metas"]
        vecs = np.load(vec_f)
        self._vecs = [vecs[i] for i in range(len(self._ids))]
