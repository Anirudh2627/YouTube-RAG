"""Time formatting helpers shared by ingestion, citations, and the UI."""
from __future__ import annotations


def format_timestamp(seconds: float | int) -> str:
    """Seconds -> human-readable `[H:]MM:SS` (YouTube style).

    >>> format_timestamp(0), format_timestamp(75), format_timestamp(3725)
    ('0:00', '1:15', '1:02:05')
    """
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_timestamp(text: str) -> int:
    """`[H:]MM:SS` or plain seconds -> int seconds.

    >>> parse_timestamp("12:43"), parse_timestamp("1:02:05"), parse_timestamp("90")
    (763, 3725, 90)
    """
    text = text.strip()
    if text.isdigit():
        return int(text)
    parts = text.split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"cannot parse timestamp: {text!r}")
    nums = [int(p) for p in parts]
    seconds = 0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds
