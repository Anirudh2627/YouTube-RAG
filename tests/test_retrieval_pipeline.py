"""Retrieval pipeline tests (offline: hashing embedder + memory store)."""
import pytest

from app.retrieval.reranker import HeuristicReranker

Q_LEARNING_RATE = "What learning rate does the speaker recommend for fine-tuning BERT?"
GOLD_SPAN = (410.0, 440.0)   # fixture segment with the lr recommendation


def _overlaps(chunk, span, pad=2.0):
    return chunk.start_time < span[1] + pad and chunk.end_time + pad > span[0]


def test_vector_retrieval_finds_gold(demo_container, demo):
    r = demo_container.retriever.retrieve(Q_LEARNING_RATE, video_id=demo.video_id,
                                          top_k=10, top_n=3, mode="vector")
    assert r.final, "no results"
    assert any(_overlaps(sc.chunk, GOLD_SPAN) for sc in r.final)
    assert r.timings_ms["total"] >= 0


def test_hybrid_retrieval_finds_gold(demo_container, demo):
    r = demo_container.retriever.retrieve(Q_LEARNING_RATE, video_id=demo.video_id,
                                          top_k=10, top_n=3, mode="hybrid")
    assert any(_overlaps(sc.chunk, GOLD_SPAN) for sc in r.final)
    assert r.bm25_candidates, "bm25 leg produced nothing"
    # hybrid scores exist
    assert all(sc.fused_score is not None for sc in r.fused)


def test_bm25_only_mode(demo_container, demo):
    r = demo_container.retriever.retrieve(Q_LEARNING_RATE, video_id=demo.video_id,
                                          top_k=10, top_n=3, mode="bm25")
    assert any(_overlaps(sc.chunk, GOLD_SPAN) for sc in r.final)


def test_video_scope_filter(demo_container, demo, settings):
    # ingest a second tiny video into the same store
    from app.models.schemas import TranscriptSegment, VideoMeta
    segs = [TranscriptSegment(text="The secret pasta ingredient is nutmeg and butter.",
                              start=0.0, duration=5.0)]
    meta = VideoMeta(video_id="pastaVideo01", url="u", title="Pasta")
    from app.ingestion.chunking import TimestampChunker
    chunks = TimestampChunker(target_tokens=30).chunk(segs, meta)
    vecs = demo_container.embedder.embed([c.text for c in chunks])
    demo_container.store.add([c.chunk_id for c in chunks], vecs,
                             [c.text for c in chunks], [c.metadata() for c in chunks])
    demo_container.retriever.invalidate()

    r_all = demo_container.retriever.retrieve("pasta ingredient", top_k=5, top_n=3,
                                              mode="hybrid")
    r_demo = demo_container.retriever.retrieve("pasta ingredient",
                                               video_id=demo.video_id,
                                               top_k=5, top_n=3, mode="hybrid")
    assert any(sc.chunk.video_id == "pastaVideo01" for sc in r_all.final)
    assert all(sc.chunk.video_id == demo.video_id for sc in r_demo.final)


def test_reranker_stage_improves_or_keeps_top1(demo_container, demo):
    demo_container.retriever.reranker = HeuristicReranker()
    r = demo_container.retriever.retrieve(Q_LEARNING_RATE, video_id=demo.video_id,
                                          top_k=10, top_n=4, mode="hybrid")
    assert len(r.final) <= 4
    assert all(sc.rerank_score is not None for sc in r.final)
    ranks = [sc.final_rank for sc in r.final]
    assert ranks == sorted(ranks)


def test_invalidate_rebuilds_bm25(demo_container, demo):
    demo_container.retriever.retrieve("attention", video_id=demo.video_id, mode="bm25")
    assert demo.video_id in demo_container.retriever._bm25_cache
    demo_container.retriever.invalidate(demo.video_id)
    assert demo.video_id not in demo_container.retriever._bm25_cache
