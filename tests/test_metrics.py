"""Metric correctness against hand-computed examples."""
import math

from app.evaluation.metrics import (aggregate, hit_rate_at_k, ndcg_at_k,
                                    precision_at_k, recall_at_k,
                                    reciprocal_rank, relevance_vector,
                                    chunk_is_relevant, GoldItem)


def test_precision_recall_hit():
    rel = [1, 0, 1, 0, 0]
    assert precision_at_k(rel, 5) == 2 / 5
    assert precision_at_k(rel, 1) == 1.0
    assert recall_at_k(rel, 3, n_relevant=4) == 0.5     # 2 of 4 gold found
    assert hit_rate_at_k([0, 0, 1], 3) == 1.0
    assert hit_rate_at_k([0, 0, 0], 3) == 0.0


def test_mrr():
    assert reciprocal_rank([0, 0, 1, 0]) == 1 / 3
    assert reciprocal_rank([1, 0]) == 1.0
    assert reciprocal_rank([0, 0]) == 0.0


def test_ndcg_known_value():
    rel = [1, 0, 1]
    # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5; IDCG = 1 + 1/log2(3)
    dcg = 1 + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert abs(ndcg_at_k(rel, 3) - dcg / idcg) < 1e-9
    assert ndcg_at_k([1], 1) == 1.0
    assert ndcg_at_k([0], 3) == 0.0


def test_aggregate():
    per_q = [([1, 0, 0], 1), ([0, 0, 1], 2)]
    m = aggregate(per_q, ks=(1, 3))
    assert m.n_questions == 2
    assert abs(m.recall[1] - (1.0 + 0.0) / 2) < 1e-9
    assert abs(m.mrr - (1.0 + 1 / 3) / 2) < 1e-9
    s = m.summary()
    assert "MRR" in s and "Recall@3" in s


def test_interval_relevance():
    gold = [GoldItem(video_id="v", start=100, end=150)]
    assert chunk_is_relevant(90, 120, gold, "v")       # overlaps
    assert chunk_is_relevant(140, 200, gold, "v")      # overlaps
    assert not chunk_is_relevant(0, 90, gold, "v")     # before (pad=2)
    assert not chunk_is_relevant(160, 200, gold, "v")  # after
    # other video's gold doesn't match
    assert not chunk_is_relevant(100, 150, [GoldItem(video_id="w", start=100, end=150)], "v")


def test_relevance_vector():
    gold = [GoldItem(video_id="v", start=100, end=150)]
    spans = [(100, 120, "v"), (0, 50, "v"), (140, 160, "v")]
    assert relevance_vector(spans, gold) == [1, 0, 1]
