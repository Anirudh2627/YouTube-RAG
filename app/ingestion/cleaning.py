"""Transcript cleaning: turns raw caption segments into tidy segments.

Pipeline: unicode normalization → artifact removal ([Music], ♪, URLs) →
whitespace normalization → consecutive-duplicate removal. Timestamps are
untouched — cleaning must never shift a segment's position on the timeline.
"""
from __future__ import annotations

from app.models.schemas import TranscriptSegment
from app.utils.text import clean_segment_text, dedupe_consecutive


def clean_segments(segments: list[TranscriptSegment], aggressive: bool = False) -> list[TranscriptSegment]:
    """Return a cleaned copy of the segment list (empty segments dropped)."""
    dicts = [
        {"text": clean_segment_text(s.text, aggressive=aggressive),
         "start": s.start, "duration": s.duration}
        for s in segments
    ]
    dicts = dedupe_consecutive(dicts)
    return [
        TranscriptSegment(text=d["text"], start=d["start"], duration=d["duration"])
        for d in dicts if d["text"]
    ]
