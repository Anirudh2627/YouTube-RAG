"""End-to-end tests: full pipeline on the offline fixture, no network.

Covers the golden path a real user takes:
  process video → cached re-process → multi-turn grounded chat with
  clickable timestamp citations → cross-video collection retrieval →
  evaluation harness runs and produces a report.
"""
import json

from app.evaluation.judges import HeuristicJudge
from app.evaluation.runner import run_evaluation
from tests.conftest import FIXTURE_PATH, FIXTURE_URL

GOLD_LR = (410.0, 440.0)   # fixture segment: fine-tuning lr recommendation


def test_full_user_journey(container):
    # 1. process
    meta, cached, elapsed = container.videos.process(FIXTURE_URL)
    assert not cached and meta.n_chunks > 5 and elapsed >= 0

    # 2. cache hit on re-process
    meta2, cached2, _ = container.videos.process(FIXTURE_URL)
    assert cached2 and meta2.video_id == meta.video_id

    # 3. grounded question → answer + clickable timestamp citations
    resp = container.engine.answer(
        "What learning rate does the speaker recommend for fine-tuning?",
        video_id=meta.video_id, debug=True)
    assert resp.answer and resp.sources
    assert any(s.start_time < GOLD_LR[1] + 2 and s.end_time > GOLD_LR[0] - 2
               for s in resp.sources)
    md = resp.answer_markdown
    assert "youtube.com/watch?v=" in md and "&t=" in md
    assert "**Relevant sections:**" in md

    # 4. follow-up in same conversation gets contextualized
    resp2 = container.engine.answer("Why does it need warmup?",
                                    video_id=meta.video_id,
                                    conversation_id=resp.conversation_id,
                                    debug=True)
    assert resp2.debug.rewritten_query
    assert resp2.conversation_id == resp.conversation_id

    # 5. chunks on disk carry required metadata fields
    chunks = container.videos.get_chunks(meta.video_id)
    required = {"chunk_id", "video_id", "video_url", "title",
                "start_time", "end_time", "index", "text"}
    assert required <= set(chunks[0].model_dump())


def test_cross_video_collection(container, monkeypatch):
    """Two videos in one store; a collection-wide question retrieves from both."""
    container.videos.process(FIXTURE_URL)

    # fabricate a second cached video directly through the service internals
    from app.ingestion.chunking import TimestampChunker
    from app.models.schemas import TranscriptSegment, VideoMeta
    segs = [
        TranscriptSegment(text=("Quantization reduces model weights to eight bits, "
                                "cutting memory usage by roughly four times while "
                                "keeping most of the accuracy."), start=0.0, duration=12.0),
        TranscriptSegment(text=("We measured the quantized transformer on device and "
                                "it ran twice as fast with a small accuracy drop."),
                         start=12.0, duration=12.0),
    ]
    meta = VideoMeta(video_id="quantVideo01", url="https://www.youtube.com/watch?v=quantVideo01",
                     title="Quantization in practice")
    chunks = TimestampChunker(target_tokens=60).chunk(segs, meta)
    vecs = container.embedder.embed([c.text for c in chunks])
    container.store.add([c.chunk_id for c in chunks], vecs, [c.text for c in chunks],
                        [c.metadata() for c in chunks])
    container.retriever.invalidate()
    cdir = container.videos.cache_dir("quantVideo01")
    cdir.mkdir(parents=True, exist_ok=True)
    meta.n_chunks = len(chunks)
    (cdir / "meta.json").write_text(meta.model_dump_json())
    (cdir / "chunks.json").write_text(json.dumps([c.model_dump() for c in chunks]))

    # collection-wide query (no video_id) about quantization
    resp = container.engine.answer("How much memory does quantization save?")
    assert "quantVideo01" in resp.video_ids

    # library listing shows both videos
    ids = {m.video_id for m in container.videos.list_videos()}
    assert {"demoLctr001", "quantVideo01"} <= ids


def test_evaluation_harness_runs(container, tmp_path):
    """The evaluation runner produces a complete report on the demo dataset."""
    dataset = json.loads((FIXTURE_PATH.parent.parent / "eval" / "demo_dataset.json").read_text())
    # the dataset points at the same fixture (relative path in the dataset)
    assert dataset["video_source"].endswith("data/fixtures/demo_lecture.json")
    report = run_evaluation(container, FIXTURE_PATH.parent.parent / "eval" / "demo_dataset.json",
                            judge=HeuristicJudge(), top_k=10)
    assert report.n_questions == len(dataset["questions"])
    assert 0.0 <= report.retrieval_stage2["MRR"] <= 1.0
    assert 0.0 <= report.generation["faithfulness"] <= 1.0
    assert len(report.per_question) == report.n_questions
    # a config snapshot is recorded for reproducibility
    assert report.config["embedder"] == "hashing"


def test_playlist_ingestion(container, monkeypatch):
    """process_playlist ingests each id; per-item failures don't kill the batch."""
    monkeypatch.setattr(container.videos, "playlist_video_ids",
                        lambda url, limit=20: [FIXTURE_URL])
    results = container.videos.process_playlist("https://www.youtube.com/playlist?list=PLfake")
    assert len(results) == 1
    meta, cached, _ = results[0]
    assert meta.video_id == "demoLctr001"
