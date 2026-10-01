from app.ingestion.cleaning import clean_segments
from app.models.schemas import TranscriptSegment
from app.utils.text import clean_segment_text, dedupe_consecutive, fix_shouting


def test_removes_music_and_applause():
    assert clean_segment_text("[Music] hello world [Applause]") == "hello world"
    assert clean_segment_text("♪♪ some lyrics ♪") == "some lyrics"


def test_normalizes_whitespace_and_unicode():
    assert clean_segment_text("  too   many \t spaces \u200b ") == "too many spaces"


def test_fixes_all_caps_shouting():
    out = fix_shouting("THIS WHOLE SEGMENT IS SHOUTING AT THE VIEWER LOUDLY")
    assert out.isupper() is False
    assert "NASA" in fix_shouting("NASA LAUNCHES THE ROCKET TODAY EVERYONE CHEERS")


def test_keeps_short_caps():
    assert clean_segment_text("YES!") == "YES!"


def test_dedupe_consecutive():
    segs = [{"text": "hello world", "start": 0}, {"text": "hello world", "start": 1},
            {"text": "next line", "start": 2}]
    assert len(dedupe_consecutive(segs)) == 2


def test_clean_segments_preserves_timestamps_and_drops_empty():
    segs = [
        TranscriptSegment(text="[Music]", start=0.0, duration=2.0),
        TranscriptSegment(text="real   content", start=2.0, duration=3.0),
    ]
    out = clean_segments(segs)
    assert len(out) == 1
    assert out[0].text == "real content"
    assert out[0].start == 2.0 and out[0].duration == 3.0
