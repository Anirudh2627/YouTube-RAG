"""Transcript provider interface.

Providers are interchangeable: YouTube API, yt-dlp subtitles, or a local
file (used by tests and offline demos). All return the same shape.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import TranscriptSegment, VideoMeta


class TranscriptResult:
    def __init__(self, segments: list[TranscriptSegment], meta: VideoMeta):
        self.segments = segments
        self.meta = meta


class TranscriptProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, video_id: str, language: str | None = None) -> TranscriptResult:
        """Fetch transcript segments + video metadata for a video id."""
        raise NotImplementedError
