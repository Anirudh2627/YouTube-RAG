"""Transcript extraction via the `youtube-transcript-api` package.

Primary provider: it hits YouTube's timed-text endpoint directly (no video
download needed), so it is fast and cheap. It fails when a video has no
captions or when YouTube blocks the client IP — `YtDlpProvider` is the
fallback. Both raise `TranscriptUnavailableError` so the service layer can
degrade gracefully.

Supports both the modern (>=1.0) instance-based API and the legacy (0.6.x)
static-method API, since the package had a breaking change between them.
"""
from __future__ import annotations

from app.ingestion.base import TranscriptProvider, TranscriptResult
from app.models.schemas import TranscriptSegment, VideoMeta
from app.utils.logging import get_logger
from app.utils.youtube import watch_url

log = get_logger(__name__)


class TranscriptUnavailableError(RuntimeError):
    """No transcript could be obtained for this video."""


def _snippets_to_segments(raw) -> list[TranscriptSegment]:
    """Normalize both dict-based (legacy) and object-based (>=1.0) snippets."""
    segments: list[TranscriptSegment] = []
    for item in raw:
        if isinstance(item, dict):
            text, start, dur = item.get("text", ""), item.get("start", 0.0), item.get("duration", 0.0)
        else:  # FetchedTranscriptSnippet
            text, start, dur = getattr(item, "text", ""), getattr(item, "start", 0.0), getattr(item, "duration", 0.0)
        segments.append(
            TranscriptSegment(text=str(text), start=float(start), duration=float(dur))
        )
    return segments


class YouTubeTranscriptProvider(TranscriptProvider):
    name = "youtube-transcript-api"

    def __init__(self, prefer_manual: bool = True):
        # Human-written captions are cleaner than ASR ones when available.
        self.prefer_manual = prefer_manual

    # ------------------------------------------------------------------
    def fetch(self, video_id: str, language: str | None = "en") -> TranscriptResult:
        import youtube_transcript_api as ytt

        langs = [language] if language else ["en"]
        errors = ytt.YouTubeTranscriptApiException if hasattr(ytt, "YouTubeTranscriptApiException") else Exception

        try:
            segments, used_lang = self._fetch_modern(video_id, langs)
        except (AttributeError, TypeError):
            segments, used_lang = self._fetch_legacy(video_id, langs)
        except errors as e:
            etype = type(e).__name__
            if etype in {"TranscriptsDisabled", "NoTranscriptFound", "IpBlocked",
                         "RequestBlocked", "VideoUnavailable", "VideoUnplayable"}:
                raise TranscriptUnavailableError(f"{etype} for {video_id}: {e}") from e
            raise TranscriptUnavailableError(f"Transcript fetch failed for {video_id}: {e}") from e

        if not segments:
            raise TranscriptUnavailableError(f"Empty transcript for {video_id}")

        meta = VideoMeta(
            video_id=video_id,
            url=watch_url(video_id),
            language=used_lang,
            n_segments=len(segments),
            duration_s=segments[-1].end,
            source=self.name,
        )
        return TranscriptResult(segments=segments, meta=meta)

    # ------------------------------------------------- modern API (>=1.0)
    def _fetch_modern(self, video_id: str, langs: list[str]):
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        if not hasattr(api, "list"):  # legacy package installed
            raise AttributeError("legacy API")
        listing = api.list(video_id)

        transcript = None
        manually = getattr(listing, "manually_created_transcripts", None)
        if self.prefer_manual and manually:
            transcript = listing.find_manually_created_transcript(langs)
        else:
            try:
                transcript = listing.find_transcript(langs)
            except Exception:
                transcript = next(iter(listing))          # any language
                if transcript.is_translatable and "en" not in (transcript.language_code,):
                    try:
                        transcript = transcript.translate("en")
                    except Exception:
                        pass
        fetched = transcript.fetch()
        return _snippets_to_segments(fetched), transcript.language_code

    # --------------------------------------------------- legacy API (0.6.x)
    def _fetch_legacy(self, video_id: str, langs: list[str]):
        from youtube_transcript_api import YouTubeTranscriptApi

        listing = YouTubeTranscriptApi.list_transcripts(video_id)
        if self.prefer_manual:
            try:
                transcript = listing.find_manually_created_transcript(langs)
            except Exception:
                transcript = listing.find_transcript(langs)
        else:
            transcript = listing.find_transcript(langs)
        raw = transcript.fetch()
        return _snippets_to_segments(raw), transcript.language_code
