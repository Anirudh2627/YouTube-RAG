import pytest

from app.ingestion.chunking import (FixedTokenChunker, SentenceChunker,
                                    TimestampChunker, approx_token_count,
                                    get_chunker, split_sentences)
from app.models.schemas import TranscriptSegment, VideoMeta


def make_segments(n=10, words=12, step=6.0):
    segs = []
    for i in range(n):
        text = " ".join(f"w{i}_{j}" for j in range(words)) + "."
        segs.append(TranscriptSegment(text=text, start=i * step, duration=step))
    return segs


META = VideoMeta(video_id="testvid1234", url="https://www.youtube.com/watch?v=testvid1234",
                 title="t")


@pytest.mark.parametrize("cls", [TimestampChunker, SentenceChunker, FixedTokenChunker])
def test_chunkers_cover_timeline(cls):
    segs = make_segments()
    chunks = cls(target_tokens=30, max_tokens=50, overlap_tokens=8).chunk(segs, META)
    assert chunks, "produced no chunks"
    # starts non-decreasing, spans valid, ids unique
    for c in chunks:
        assert c.end_time >= c.start_time
        assert c.video_id == "testvid1234"
    assert [c.start_time for c in chunks] == sorted(c.start_time for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # first chunk starts at video start; last ends at/near video end
    assert chunks[0].start_time == pytest.approx(0.0, abs=1.0)
    assert chunks[-1].end_time >= segs[-1].start


@pytest.mark.parametrize("cls", [TimestampChunker, SentenceChunker])
def test_sentence_boundary_respected(cls):
    segs = make_segments(words=10)
    chunks = cls(target_tokens=25, max_tokens=40, overlap_tokens=0).chunk(segs, META)
    for c in chunks:
        # no chunk may start with a lowercase continuation fragment
        assert c.text[0].isalnum()


def test_no_tiny_trailing_chunks():
    segs = make_segments(n=12, words=40, step=10.0)
    chunks = TimestampChunker(target_tokens=60, max_tokens=90,
                              overlap_tokens=15, min_tokens=24).chunk(segs, META)
    for c in chunks:
        assert approx_token_count(c.text) >= 24


def test_overlap_creates_shared_content():
    # caption-like short segments: multiple sentences per chunk so the
    # sentence-level overlap can actually kick in
    segs = make_segments(n=16, words=8, step=4.0)
    chunks = TimestampChunker(target_tokens=40, max_tokens=60,
                              overlap_tokens=20).chunk(segs, META)
    shared = 0
    for a, b in zip(chunks, chunks[1:]):
        ta, tb = set(a.text.split()), set(b.text.split())
        shared += len(ta & tb) > 0
    assert shared >= len(chunks) - 2, "overlap should repeat tokens between neighbors"


def test_metadata_fields_present():
    segs = make_segments(n=4)
    c = TimestampChunker(target_tokens=20).chunk(segs, META)[0]
    md = c.metadata()
    for key in ["chunk_id", "video_id", "video_url", "title",
                "start_time", "end_time", "index"]:
        assert key in md


def test_empty_input():
    assert TimestampChunker().chunk([], META) == []


def test_get_chunker_validation():
    assert isinstance(get_chunker("timestamp"), TimestampChunker)
    with pytest.raises(ValueError):
        get_chunker("nonsense")


def test_split_sentences_abbreviations():
    out = split_sentences("Dr. Smith went to Washington. He arrived at 3 p.m. Then he left.")
    assert out[0].startswith("Dr. Smith")
    assert len(out) == 3
