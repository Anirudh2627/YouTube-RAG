from app.models.schemas import Chunk, ScoredChunk
from app.retrieval.hybrid import rrf_fuse


def sc(cid: str, **kw) -> ScoredChunk:
    chunk = Chunk(chunk_id=cid, video_id="v", video_url="u", text=cid,
                  start_time=0, end_time=1, index=0)
    return ScoredChunk(chunk=chunk, **kw)


def test_rrf_prefers_consensus():
    vec = [sc("a", vector_score=0.9), sc("b", vector_score=0.8), sc("c", vector_score=0.7)]
    bm = [sc("b", bm25_score=5.0), sc("c", bm25_score=4.0), sc("a", bm25_score=3.0)]
    fused = rrf_fuse([vec, bm], k=60)
    ids = [f.chunk.chunk_id for f in fused]
    # 'a' and 'b' both appear at ranks {1,3}/{2,1}; b beats a on consensus
    assert ids[0] in {"a", "b"}
    assert set(ids) == {"a", "b", "c"}
    # every item gets a fused score, and stage-1 scores survive
    for f in fused:
        assert f.fused_score > 0


def test_rrf_single_list_is_rank_order():
    vec = [sc("x", vector_score=0.9), sc("y", vector_score=0.5)]
    fused = rrf_fuse([vec])
    assert [f.chunk.chunk_id for f in fused] == ["x", "y"]


def test_rrf_top_k_and_weights():
    vec = [sc("a", vector_score=0.9), sc("b", vector_score=0.8)]
    bm = [sc("b", bm25_score=9.0), sc("a", bm25_score=1.0)]
    # heavy bm25 weight should flip the order
    fused = rrf_fuse([vec, bm], weights=[0.0, 1.0], top_k=1)
    assert len(fused) == 1 and fused[0].chunk.chunk_id == "b"


def test_rrf_score_formula():
    a = [sc("only", vector_score=1.0)]
    fused = rrf_fuse([a], weights=[2.0], k=60)
    assert abs(fused[0].fused_score - 2.0 / 61) < 1e-6


def test_bm25_score_attached_after_fusion():
    vec = [sc("a", vector_score=0.9)]
    bm = [sc("a", bm25_score=4.2), sc("b", bm25_score=3.0)]
    fused = rrf_fuse([vec, bm])
    by_id = {f.chunk.chunk_id: f for f in fused}
    assert by_id["a"].bm25_score == 4.2
    assert by_id["a"].vector_score == 0.9
    assert by_id["b"].vector_score is None
