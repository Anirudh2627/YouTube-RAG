"""Rerankers: stage-2 of the retrieval pipeline.

Stage 1 (vector / BM25) uses *independent* query & doc encodings — fast but
approximate. A cross-encoder reranker feeds (query, document) pairs jointly
through a transformer, which is far more accurate but O(candidates) forward
passes — hence: retrieve wide (top_k≈15), rerank narrow (top_n≈4).

Two implementations:
  * CrossEncoderReranker — real model (default: ms-marco-MiniLM-L-6-v2).
  * HeuristicReranker — deterministic lexical-overlap fallback for tests /
    environments without model downloads. Weaker, but honest about it.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.models.schemas import ScoredChunk
from app.utils.logging import get_logger

log = get_logger(__name__)


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk],
               top_n: int) -> list[ScoredChunk]:
        """Return candidates reordered & truncated, with rerank_score set."""


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_len: int = 512,
        stage1_weight: float = 0.10,
    ):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, max_length=max_len)
        self.model_name = model_name
        self.stage1_weight = stage1_weight
        log.info("loaded reranker %s", model_name)

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_n: int,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        pairs = [(query, sc.text[:2000]) for sc in candidates]
        logits = self._model.predict(pairs, show_progress_bar=False)

        import math

        # Cross-encoder relevance score.
        cross_scores = []
        for sc, logit in zip(candidates, logits):
            score = 1.0 / (1.0 + math.exp(-float(logit)))
            sc.rerank_score = round(score, 6)
            cross_scores.append(score)

        # Preserve some information from stage-1 retrieval.
        # Rank-based prior is deliberately used instead of mixing raw
        # vector/BM25/RRF scores because those scores have different scales.
        n = len(candidates)
        for i, sc in enumerate(candidates):
            stage1_prior = 1.0 if n == 1 else 1.0 - (i / (n - 1))

            cross_score = cross_scores[i]

            combined_score = (
                (1.0 - self.stage1_weight) * cross_score
                + self.stage1_weight * stage1_prior
            )

            # Keep the cross-encoder score for debugging/API output,
            # while using the combined score only for final ordering.
            sc._combined_rerank_score = combined_score

        ranked = sorted(
            candidates,
            key=lambda s: s._combined_rerank_score,
            reverse=True,
        )

        for i, sc in enumerate(ranked[:top_n]):
            sc.final_rank = i + 1

        return ranked[:top_n]


class HeuristicReranker(Reranker):
    """Deterministic fallback: token-F1 overlap with the query, blended with
    the stage-1 score. Used in tests and when no reranker model is available."""
    name = "heuristic"

    _re = re.compile(r"[a-z0-9']+")

    def __init__(self, stage1_weight: float = 0.3):
        self.stage1_weight = stage1_weight

    def _toks(self, s: str) -> set[str]:
        return set(self._re.findall(s.lower()))

    def rerank(self, query: str, candidates: list[ScoredChunk],
               top_n: int) -> list[ScoredChunk]:
        q = self._toks(query)
        for sc in candidates:
            d = self._toks(sc.text)
            if not q or not d:
                f1 = 0.0
            else:
                inter = len(q & d)
                prec = inter / len(d)
                rec = inter / len(q)
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            stage1 = sc.fused_score if sc.fused_score is not None else (sc.vector_score or 0.0)
            stage1 = max(0.0, min(1.0, stage1))
            sc.rerank_score = round((1 - self.stage1_weight) * f1
                                    + self.stage1_weight * stage1, 6)
        ranked = sorted(candidates, key=lambda s: s.rerank_score or 0.0, reverse=True)
        for i, sc in enumerate(ranked[:top_n]):
            sc.final_rank = i + 1
        return ranked[:top_n]


def get_reranker(enabled: bool, model_name: str,
                 allow_fallback: bool = True) -> Reranker | None:
    if not enabled:
        return None
    try:
        return CrossEncoderReranker(model_name)
    except Exception as e:  # no network / no torch → fall back, don't crash
        if not allow_fallback:
            raise
        log.warning("CrossEncoder unavailable (%s); using HeuristicReranker", e)
        return HeuristicReranker()
