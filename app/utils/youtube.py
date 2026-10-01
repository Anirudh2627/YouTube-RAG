"""YouTube URL parsing and watch-URL construction.

Handles the common URL shapes users paste:
    https://www.youtube.com/watch?v=VIDEO_ID&t=123s
    https://youtu.be/VIDEO_ID
    https://www.youtube.com/shorts/VIDEO_ID
    https://www.youtube.com/live/VIDEO_ID
    https://www.youtube.com/embed/VIDEO_ID
    https://m.youtube.com/watch?v=VIDEO_ID
    raw 11-character video IDs
"""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class InvalidYouTubeURLError(ValueError):
    """Raised when a string cannot be resolved to a YouTube video id."""


def extract_video_id(url_or_id: str) -> str:
    """Extract the 11-character video id from a YouTube URL (or bare id)."""
    raw = (url_or_id or "").strip()
    if _VIDEO_ID_RE.match(raw):
        return raw

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")

    if host in {"youtube.com", "youtube-nocookie.com"}:
        if parsed.path == "/watch":
            vid = parse_qs(parsed.query).get("v", [None])[0]
            if vid and _VIDEO_ID_RE.match(vid):
                return vid
        for prefix in ("/shorts/", "/live/", "/embed/", "/v/"):
            if parsed.path.startswith(prefix):
                vid = parsed.path[len(prefix):].split("/")[0].split("?")[0]
                if _VIDEO_ID_RE.match(vid):
                    return vid
    elif host in {"youtu.be",}:
        vid = parsed.path.lstrip("/").split("/")[0]
        if _VIDEO_ID_RE.match(vid):
            return vid

    raise InvalidYouTubeURLError(
        f"Could not extract a YouTube video id from: {url_or_id!r}"
    )


def watch_url(video_id: str, t_seconds: int | None = None) -> str:
    """Canonical watch URL, optionally deep-linked to a timestamp."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    if t_seconds is not None:
        url += f"&t={int(t_seconds)}s"
    return url

