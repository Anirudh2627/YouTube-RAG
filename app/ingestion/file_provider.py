"""File-based transcript provider — loads a local JSON fixture.

Used by tests, offline demos, and the evaluation harness so the full
pipeline can run without network access. Fixture format:

    {
      "meta": {"video_id": "...", "title": "...", "channel": "...",
               "duration_s": 1234, "language": "en"},
      "segments": [{"text": "...", "start": 0.0, "duration": 2.5}, ...]
    }
"""
from __future__ import annotations

import json
from pathlib import Path

from app.ingestion.base import TranscriptProvider, TranscriptResult
from app.ingestion.youtube_transcript import TranscriptUnavailableError
from app.models.schemas import TranscriptSegment, VideoMeta
from app.utils.youtube import watch_url


class FileTranscriptProvider(TranscriptProvider):
    name = "file"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch(self, video_id: str | None = None, language: str | None = None) -> TranscriptResult:
        if not self.path.exists():
            raise TranscriptUnavailableError(f"Fixture not found: {self.path}")
        data = json.loads(self.path.read_text())
        segments = [TranscriptSegment(**s) for s in data["segments"]]
        meta_d = data.get("meta", {})
        meta_d.setdefault("video_id", video_id or self.path.stem)
        meta_d.setdefault("url", watch_url(meta_d["video_id"]))
        meta_d.setdefault("n_segments", len(segments))
        meta_d.setdefault("duration_s", segments[-1].end if segments else None)
        meta_d.setdefault("source", "fixture")
        meta = VideoMeta(**meta_d)
        return TranscriptResult(segments=segments, meta=meta)
