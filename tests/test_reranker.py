import pytest

from app.models.schemas import Chunk, ScoredChunk
from app.retrieval.reranker import HeuristicReranker, get_reranker


def sc(cid: str, text: str, vector_score: float = 0.5) -> ScoredChunk:
    chunk = Chunk(chunk_id=cid, video_id="v", video_url="u", text=text,
                  start_time=0, end_time=1, index=0)
    return ScoredChunk(chunk=chunk, vector_score=vector_score)


def test_heuristic_reranker_orders_by_relevance():
    r = HeuristicReranker()
    cands = [
        sc("c1", "unrelated cooking pasta with tomato sauce", 0.6),
        sc("c2", "the attention mechanism uses queries keys and values", 0.5),
        sc("c3", "attention attention attention", 0.4),
    ]
    out = r.rerank("how does attention work with queries and keys", cands, top_n=2)
    assert len(out) == 2
    assert out[0].chunk.chunk_id == "c2"        # best content match wins
    assert out[0].final_rank == 1
    assert all(o.rerank_score is not None for o in out)


def test_heuristic_scores_in_unit_range():
    r = HeuristicReranker()
    out = r.rerank("some query words", [sc("a", "some query words here")], top_n=1)
    assert 0.0 <= out[0].rerank_score <= 1.0


def test_get_reranker_disabled():
    assert get_reranker(False, "whatever") is None


@pytest.mark.slow
def test_cross_encoder_reranker():
    r = get_reranker(True, "cross-encoder/ms-marco-MiniLM-L-6-v2",
                     allow_fallback=False)
    cands = [
        sc("bad", "the weather in Paris is rainy today"),
        sc("good", "self-attention connects all positions with constant path length"),
    ]
    out = r.rerank("why is self-attention better than recurrence for long sequences",
                   cands, top_n=2)
    assert out[0].chunk.chunk_id == "good"
    assert out[0].rerank_score > out[1].rerank_score


def test_cross_encoder_preserves_strong_stage1_candidates():
    from app.retrieval.reranker import CrossEncoderReranker

    reranker = object.__new__(CrossEncoderReranker)
    reranker.stage1_weight = 0.10

    class FakeModel:
        def predict(self, pairs, show_progress_bar=False):
            # Cross-encoder prefers candidate 3,
            # while stage-1 strongly preferred candidate 1.
            return [-2.0, -3.0, 2.0]

    reranker._model = FakeModel()

    candidates = [
        sc("c1", "Important explanation of the MLP architecture.", 0.90),
        sc("c2", "Some unrelated information.", 0.50),
        sc("c3", "Another unrelated section.", 0.10),
    ]

    result = reranker.rerank(
        "What is the MLP architecture?",
        candidates,
        top_n=2,
    )

    assert len(result) == 2
    assert result[0].chunk.chunk_id == "c3"
    assert result[0].final_rank == 1
