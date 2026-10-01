import pytest

from app.ingestion.chunking import (
    TimestampChunker,
    approx_token_count,
    get_chunker,
    split_sentences,
)
from app.models.schemas import TranscriptSegment, VideoMeta


def make_segments(n=10, words=12, step=6.0):
    segs = []

    for i in range(n):
        text = " ".join(
            f"w{i}_{j}" for j in range(words)
        ) + "."

        segs.append(
            TranscriptSegment(
                text=text,
                start=i * step,
                duration=step,
            )
        )

    return segs


META = VideoMeta(
    video_id="testvid1234",
    url="https://www.youtube.com/watch?v=testvid1234",
    title="t",
)


def test_timestamp_chunker_covers_timeline():
    segs = make_segments()

    chunks = TimestampChunker(
        target_tokens=30,
        max_tokens=50,
        overlap_tokens=8,
    ).chunk(segs, META)

    assert chunks, "produced no chunks"

    # Starts are non-decreasing and every chunk has valid metadata.
    for chunk in chunks:
        assert chunk.end_time >= chunk.start_time
        assert chunk.video_id == "testvid1234"

    assert [
        chunk.start_time for chunk in chunks
    ] == sorted(
        chunk.start_time for chunk in chunks
    )

    # Chunk IDs must be unique.
    assert len(
        {chunk.chunk_id for chunk in chunks}
    ) == len(chunks)

    # First chunk starts near the beginning of the video.
    assert chunks[0].start_time == pytest.approx(
        0.0,
        abs=1.0,
    )

    # Last chunk reaches the end of the transcript.
    assert chunks[-1].end_time >= segs[-1].start


def test_sentence_boundary_respected():
    segs = make_segments(words=10)

    chunks = TimestampChunker(
        target_tokens=25,
        max_tokens=40,
        overlap_tokens=0,
    ).chunk(segs, META)

    for chunk in chunks:
        # Chunks should not begin with a lowercase continuation fragment.
        assert chunk.text[0].isalnum()


def test_no_tiny_trailing_chunks():
    segs = make_segments(
        n=12,
        words=40,
        step=10.0,
    )

    chunks = TimestampChunker(
        target_tokens=60,
        max_tokens=90,
        overlap_tokens=15,
        min_tokens=24,
    ).chunk(segs, META)

    for chunk in chunks:
        assert approx_token_count(chunk.text) >= 24


def test_overlap_creates_shared_content():
    # Caption-like short segments: multiple sentences per chunk
    # allow sentence-level overlap to take effect.
    segs = make_segments(
        n=16,
        words=8,
        step=4.0,
    )

    chunks = TimestampChunker(
        target_tokens=40,
        max_tokens=60,
        overlap_tokens=20,
    ).chunk(segs, META)

    shared = 0

    for first, second in zip(
        chunks,
        chunks[1:],
    ):
        first_tokens = set(first.text.split())
        second_tokens = set(second.text.split())

        shared += len(
            first_tokens & second_tokens
        ) > 0

    assert shared >= len(chunks) - 2, (
        "overlap should repeat tokens between neighbors"
    )


def test_metadata_fields_present():
    segs = make_segments(n=4)

    chunk = TimestampChunker(
        target_tokens=20
    ).chunk(segs, META)[0]

    metadata = chunk.metadata()

    for key in [
        "chunk_id",
        "video_id",
        "video_url",
        "title",
        "start_time",
        "end_time",
        "index",
    ]:
        assert key in metadata


def test_empty_input():
    assert TimestampChunker().chunk([], META) == []


def test_get_chunker_returns_timestamp_chunker():
    chunker = get_chunker(
        target_tokens=160,
        max_tokens=240,
        overlap_tokens=40,
    )

    assert isinstance(
        chunker,
        TimestampChunker,
    )


def test_split_sentences_abbreviations():
    output = split_sentences(
        "Dr. Smith went to Washington. "
        "He arrived at 3 p.m. Then he left."
    )

    assert output[0].startswith("Dr. Smith")
    assert len(output) == 3