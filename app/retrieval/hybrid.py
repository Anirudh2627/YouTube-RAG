"""Hybrid retrieval: fuse vector and BM25 rankings with Reciprocal Rank
Fusion (RRF).

Why RRF instead of score-weighted merging?
  * Cosine similarity (0..1-ish) and BM25 (unbounded) live on incomparable
    scales; normalizing them is fiddly and corpus-dependent.
  * RRF only uses *ranks*: score(d) = Σ_lists w / (k + rank(d)). It is
    parameter-light (one k, default 60 from the original paper), robust, and
    empirically strong — which is why Elasticsearch/OpenSearch/Weaviate all
    ship it as their default hybrid strategy.

Weighted variant: we multiply each list's contribution by a configurable
weight so the semantic/lexical balance can be tuned per workload.
"""
from __future__ import annotations

from collections import defaultdict

from app.models.schemas import ScoredChunk


def rrf_fuse(rankings: list[list[ScoredChunk]],
             weights: list[float] | None = None,
             k: int = 60,
             top_k: int | None = None) -> list[ScoredChunk]:
    """Fuse multiple ranked lists of ScoredChunk (same chunk ids).

    `rankings[0]` is expected to carry the vector scores, `rankings[1]` the
    BM25 scores; fused_score is written back onto each surviving chunk.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = defaultdict(float)
    by_id: dict[str, ScoredChunk] = {}
    per_list_score: dict[str, dict[str, float | None]] = defaultdict(dict)

    for ranking, w in zip(rankings, weights):
        for rank, sc in enumerate(ranking):
            cid = sc.chunk.chunk_id
            fused[cid] += w / (k + rank + 1)
            by_id.setdefault(cid, sc)
            if sc.vector_score is not None:
                per_list_score[cid]["vector"] = sc.vector_score
            if sc.bm25_score is not None:
                per_list_score[cid]["bm25"] = sc.bm25_score

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if top_k:
        ordered = ordered[:top_k]
    out: list[ScoredChunk] = []
    for i, (cid, score) in enumerate(ordered):
        sc = by_id[cid]
        sc.fused_score = round(score, 6)
        # attach whichever stage-1 scores we saw for debug mode
        if sc.vector_score is None:
            sc.vector_score = per_list_score[cid].get("vector")
        if sc.bm25_score is None:
            sc.bm25_score = per_list_score[cid].get("bm25")
        out.append(sc)
    return out
