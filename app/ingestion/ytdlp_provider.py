"""Fallback transcript provider using yt-dlp subtitle extraction.

Used when `youtube-transcript-api` fails (e.g. IP blocks, unusual caption
formats). yt-dlp is more robust at negotiating with YouTube but slower.
It also gives us full video metadata (title, channel, duration), which the
primary provider does not.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from app.ingestion.base import TranscriptProvider, TranscriptResult
from app.ingestion.youtube_transcript import TranscriptUnavailableError
from app.models.schemas import TranscriptSegment, VideoMeta
from app.utils.logging import get_logger
from app.utils.youtube import watch_url

log = get_logger(__name__)


class YtDlpProvider(TranscriptProvider):
    name = "yt-dlp"

    def __init__(self, ytdlp_bin: str = "yt-dlp", timeout_s: int = 120,
                 cookies_file: str | None = None):
        self.bin = ytdlp_bin
        self.timeout_s = timeout_s
        # Netscape-format cookies file — the standard workaround when YouTube
        # blocks datacenter IPs ("Sign in to confirm you're not a bot").
        # Configure via YTDLP_COOKIES_FILE=/path/to/cookies.txt
        self.cookies_file = cookies_file

    def _base_cmd(self) -> list[str]:
        cmd = [self.bin, "--no-warnings"]
        if self.cookies_file:
            cmd += ["--cookies", self.cookies_file]
        return cmd

    # ------------------------------------------------------------------
    def fetch(self, video_id: str, language: str | None = "en") -> TranscriptResult:
        url = watch_url(video_id)
        with tempfile.TemporaryDirectory() as td:
            out_tmpl = str(Path(td) / "subs")
            cmd = self._base_cmd() + [
                "--skip-download", "--write-auto-subs", "--write-subs",
                "--sub-langs", f"{language or 'en'}.*", "--sub-format", "json3/srv3/vtt",
                "-o", out_tmpl, url,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=self.timeout_s)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                raise TranscriptUnavailableError(f"yt-dlp failed to run: {e}") from e

            files = list(Path(td).glob("*.json3")) or list(Path(td).glob("*.vtt"))
            if not files:
                raise TranscriptUnavailableError(
                    f"yt-dlp found no subtitles for {video_id} "
                    f"(exit={proc.returncode}): {proc.stderr[:300]}"
                )
            segments = self._parse_subtitle_file(files[0])

        if not segments:
            raise TranscriptUnavailableError(f"yt-dlp produced empty transcript for {video_id}")

        meta = self._fetch_metadata(video_id) or VideoMeta(video_id=video_id, url=url)
        meta.n_segments = len(segments)
        meta.duration_s = meta.duration_s or segments[-1].end
        meta.language = language or "en"
        meta.source = self.name
        return TranscriptResult(segments=segments, meta=meta)

    # ------------------------------------------------------------------
    def _fetch_metadata(self, video_id: str) -> VideoMeta | None:
        try:
            proc = subprocess.run(
                self._base_cmd() + ["--dump-single-json", "--skip-download",
                                      "--no-download-archive", watch_url(video_id)],
                capture_output=True, text=True, timeout=self.timeout_s,
            )
            if proc.returncode != 0:
                return None
            info = json.loads(proc.stdout)
            return VideoMeta(
                video_id=video_id,
                url=watch_url(video_id),
                title=info.get("title", ""),
                channel=info.get("channel") or info.get("uploader", ""),
                duration_s=float(info.get("duration") or 0) or None,
            )
        except Exception as e:  # metadata is best-effort
            log.warning("yt-dlp metadata fetch failed for %s: %s", video_id, e)
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_subtitle_file(path: Path) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        if path.suffix == ".json3":
            data = json.loads(path.read_text())
            for ev in data.get("events", []):
                segs = ev.get("segs") or []
                text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ")
                if not text.strip():
                    continue
                start = ev.get("tStartMs", 0) / 1000.0
                dur = ev.get("dDurationMs", 0) / 1000.0
                segments.append(TranscriptSegment(text=text, start=start, duration=dur))
        else:  # .vtt — strip cue headers/tags crudely
            import re
            for block in re.split(r"\n\n+", path.read_text(errors="ignore")):
                lines = [l for l in block.splitlines()
                         if l.strip() and "-->" not in l and not l.startswith(("WEBVTT", "Kind:", "Language:", "NOTE"))]
                text = re.sub(r"<[^>]+>", "", " ".join(lines)).strip()
                if text:
                    m = re.search(r"(\d+):(\d+):(\d+)\.(\d+)", block)
                    start = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                             + int(m.group(4)) / 1000) if m else 0.0
                    segments.append(TranscriptSegment(text=text, start=start, duration=0.0))
        return segments
