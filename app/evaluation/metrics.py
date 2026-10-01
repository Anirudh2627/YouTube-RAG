
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class GoldItem:
    """One gold relevance interval on a video's timeline."""
    video_id: str
    start: float
    end: float


def chunk_is_relevant(start: float, end: float, gold: list[GoldItem],
                      video_id: str | None = None, pad_s: float = 2.0) -> bool:
    for g in gold:
        if video_id and g.video_id and g.video_id != video_id:
            continue
        if start < g.end + pad_s and end + pad_s > g.start:
            return True
    return False


def relevance_vector(ranked_spans: list[tuple[float, float, str]],
                     gold: list[GoldItem]) -> list[int]:
    """1/0 relevance for each ranked chunk (start, end, video_id)."""
    return [int(chunk_is_relevant(s, e, gold, vid)) for s, e, vid in ranked_spans]


# metrics

def precision_at_k(rel: list[int], k: int) -> float:
    top = rel[:k]
    return sum(top) / k if top else 0.0


def recall_at_k(rel: list[int], k: int, n_relevant: int | None = None) -> float:
    """n_relevant = total relevant chunks in the corpus. When unknown, we use
    the number of relevant items inside the retrieved list (hit-rate style);
    the runner always passes the true corpus count."""
    hits = sum(rel[:k])
    denom = n_relevant if n_relevant else sum(rel)
    return hits / denom if denom else 0.0


def hit_rate_at_k(rel: list[int], k: int) -> float:
    return 1.0 if any(rel[:k]) else 0.0


def reciprocal_rank(rel: list[int]) -> float:
    for i, r in enumerate(rel):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(rel: list[int], k: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel[:k]))
    ideal = sorted(rel, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


@dataclass
class RetrievalMetrics:
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    hit_rate: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)
    n_questions: int = 0

    def summary(self) -> dict[str, float]:
        out = {
            "MRR": round(self.mrr, 4),
            "n_questions": self.n_questions,
        }
        for k, v in self.recall.items():
            out[f"Recall@{k}"] = round(v, 4)
        for k, v in self.precision.items():
            out[f"Precision@{k}"] = round(v, 4)
        for k, v in self.hit_rate.items():
            out[f"HitRate@{k}"] = round(v, 4)
        for k, v in self.ndcg.items():
            out[f"nDCG@{k}"] = round(v, 4)
        return out


def aggregate(per_query: list[tuple[list[int], int]],
              ks: tuple[int, ...] = (1, 3, 5, 10)) -> RetrievalMetrics:
    """per_query: list of (relevance_vector, n_relevant_in_corpus)."""
    m = RetrievalMetrics(n_questions=len(per_query))
    if not per_query:
        return m
    for k in ks:
        m.recall[k] = sum(recall_at_k(rel, k, n) for rel, n in per_query) / len(per_query)
        m.precision[k] = sum(precision_at_k(rel, k) for rel, n in per_query) / len(per_query)
        m.hit_rate[k] = sum(hit_rate_at_k(rel, k) for rel, n in per_query) / len(per_query)
        m.ndcg[k] = sum(ndcg_at_k(rel, k) for rel, n in per_query) / len(per_query)
    m.mrr = sum(reciprocal_rank(rel) for rel, _ in per_query) / len(per_query)
    return m
