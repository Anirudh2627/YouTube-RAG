"""Retrieval pipeline: the stage-1/stage-2 orchestrator.

    query → embed → vector top_k ┐
                                 ├→ RRF fusion → reranker → top_n
    query → tokenize → BM25 top_k┘

Every intermediate ranking is kept in a `RetrievalResult` so debug mode can
show exactly why a chunk made it into (or fell out of) the final context.

The BM25 index is rebuilt lazily from the vector store's stored documents
and cached per scope (`__all__` or a single video_id). Rebuilds happen when
a video is (re)ingested — see `invalidate()`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.embeddings.base import Embedder
from app.models.schemas import Chunk, ScoredChunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import rrf_fuse
from app.retrieval.reranker import Reranker
from app.utils.logging import get_logger
from app.vectorstore.base import VectorHit, VectorStore

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    query: str
    final: list[ScoredChunk] = field(default_factory=list)
    vector_candidates: list[ScoredChunk] = field(default_factory=list)
    bm25_candidates: list[ScoredChunk] = field(default_factory=list)
    fused: list[ScoredChunk] = field(default_factory=list)
    mode: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)


def _hit_to_scored(hit: VectorHit) -> ScoredChunk:
    m = hit.metadata
    chunk = Chunk(
        chunk_id=str(m.get("chunk_id", hit.id)),
        video_id=str(m.get("video_id", "")),
        video_url=str(m.get("video_url", "")),
        title=str(m.get("title", "")),
        text=hit.text,
        start_time=float(m.get("start_time", 0.0)),
        end_time=float(m.get("end_time", 0.0)),
        index=int(m.get("index", 0)),
        extra=({"frame_path": m["frame_path"], "frame_time": m.get("frame_time")}
               if m.get("frame_path") else {}),
    )
    return ScoredChunk(chunk=chunk, vector_score=round(hit.score, 6))


class RetrievalPipeline:
    def __init__(self, store: VectorStore, embedder: Embedder,
                 reranker: Reranker | None = None,
                 mode: str = "hybrid",
                 top_k: int = 15, top_n: int = 4,
                 vector_weight: float = 1.0, bm25_weight: float = 1.0,
                 rrf_k: int = 60):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.mode = mode
        self.top_k = top_k
        self.top_n = top_n
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self._bm25_cache: dict[str, BM25Index] = {}
        self._scope_cache: dict[str, set[str]] = {}

    # ---------------------------------------------------------- BM25 side
    def invalidate(self, scope: str | None = None) -> None:
        """Drop cached BM25 indexes after ingestion changes."""
        if scope is None:
            self._bm25_cache.clear()
            self._scope_cache.clear()
        else:
            self._bm25_cache.pop(scope, None)
            self._bm25_cache.pop("__all__", None)
            self._scope_cache.pop(scope, None)
            self._scope_cache.pop("__all__", None)

    def _corpus(self) -> dict[str, tuple[str, str]]:
        """chunk_id → (video_id, text) for every stored chunk."""
        docs = self.store.get_all_documents()
        return {cid: (vid, text) for cid, vid, text in docs}

    def _bm25_index(self, scope: str) -> BM25Index:
        if scope not in self._bm25_cache:
            corpus = self._corpus()
            if scope != "__all__":
                subset = {cid: (vid, t) for cid, (vid, t) in corpus.items() if vid == scope}
            else:
                subset = corpus
            idx = BM25Index()
            idx.build([(cid, t) for cid, (_, t) in subset.items()])
            self._bm25_cache[scope] = idx
        return self._bm25_cache[scope]

    # ------------------------------------------------------------- search
    def retrieve(self, query: str, video_id: str | None = None,
                 top_k: int | None = None, top_n: int | None = None,
                 mode: str | None = None) -> RetrievalResult:
        t0 = time.perf_counter()
        mode = mode or self.mode
        top_k = top_k or self.top_k
        top_n = top_n or self.top_n
        where = {"video_id": video_id} if video_id else None
        result = RetrievalResult(query=query, mode=mode)

        # ---------------- stage 1a: vector
        if mode in ("vector", "hybrid"):
            t = time.perf_counter()
            qv = self.embedder.embed_query(query)
            hits = self.store.query(qv, top_k=top_k, where=where)
            result.vector_candidates = [_hit_to_scored(h) for h in hits]
            result.timings_ms["embed+vector"] = (time.perf_counter() - t) * 1000

        # ---------------- stage 1b: BM25
        if mode in ("bm25", "hybrid"):
            t = time.perf_counter()
            scope = video_id or "__all__"
            bm25 = self._bm25_index(scope)
            raw = bm25.search(query, top_k=top_k)
            by_id = {sc.chunk.chunk_id: sc for sc in result.vector_candidates}
            bm25_scored: list[ScoredChunk] = []
            for cid, score in raw:
                sc = by_id.get(cid)
                if sc is None:
                    sc = self._load_chunk(cid)
                    if sc is None:
                        continue
                else:
                    sc = ScoredChunk(chunk=sc.chunk.model_copy(),
                                     vector_score=sc.vector_score)
                sc.bm25_score = round(float(score), 4)
                bm25_scored.append(sc)
            result.bm25_candidates = bm25_scored
            result.timings_ms["bm25"] = (time.perf_counter() - t) * 1000

        # ---------------- fusion
        if mode == "vector":
            result.fused = result.vector_candidates
        elif mode == "bm25":
            result.fused = result.bm25_candidates
        else:
            result.fused = rrf_fuse(
                [result.vector_candidates, result.bm25_candidates],
                weights=[self.vector_weight, self.bm25_weight],
                k=self.rrf_k, top_k=top_k,
            )

        # ---------------- stage 2: rerank
        t = time.perf_counter()
        if self.reranker is not None and result.fused:
            result.final = self.reranker.rerank(query, result.fused, top_n)
        else:
            result.final = result.fused[:top_n]
            for i, sc in enumerate(result.final):
                sc.final_rank = i + 1
        result.timings_ms["rerank"] = (time.perf_counter() - t) * 1000
        result.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return result

    # ----------------------------------------------------------------
    def _load_chunk(self, chunk_id: str) -> ScoredChunk | None:
        """Materialize a ScoredChunk for a BM25-only hit (not in vector top)."""
        corpus = self._corpus()
        if chunk_id not in corpus:
            return None
        vid, text = corpus[chunk_id]
        meta = self.store.get_metadata(chunk_id)
        if meta is None:
            return None
        chunk = Chunk(
            chunk_id=chunk_id, video_id=vid,
            video_url=str(meta.get("video_url", "")),
            title=str(meta.get("title", "")),
            text=text,
            start_time=float(meta.get("start_time", 0.0)),
            end_time=float(meta.get("end_time", 0.0)),
            index=int(meta.get("index", 0)),
        )
        return ScoredChunk(chunk=chunk)
