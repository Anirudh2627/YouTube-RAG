"""Multimodal support: extract representative video frames per chunk.

Design:
  * One keyframe per chunk, taken at the chunk midpoint, saved to
    data/cache/<video_id>/frames/<chunk_id>.jpg
  * The frame path is attached to the chunk's metadata, so it flows through
    the vector store into Sources and the UI — retrieval itself stays
    text-based (transcript), frames are *additional evidence* displayed next
    to the cited moment ("what slide was on screen at 12:43?").
  * Fully optional: requires yt-dlp + ffmpeg. If either is missing (or the
    video can't be downloaded, e.g. datacenter IP blocks), ingestion simply
    proceeds text-only. `available()` reports capability.

Implementation: yt-dlp resolves a low-res (≤360p) video stream URL, then
ffmpeg seeks directly to the timestamp and grabs a single frame — no full
video download.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.models.schemas import Chunk
from app.utils.logging import get_logger
from app.utils.youtube import watch_url

log = get_logger(__name__)


class FrameExtractor:
    def __init__(self, frames_root: str | Path, ytdlp_bin: str = "yt-dlp",
                 timeout_s: int = 45):
        self.frames_root = Path(frames_root)
        self.bin = ytdlp_bin
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    @staticmethod
    def available() -> bool:
        return bool(shutil.which("ffmpeg") and shutil.which("yt-dlp"))

    def frames_dir(self, video_id: str) -> Path:
        return self.frames_root / video_id / "frames"

    # ------------------------------------------------------------------
    def _stream_url(self, video_id: str) -> str | None:
        try:
            proc = subprocess.run(
                [self.bin, "-f", "bv*[height<=360]/b[height<=360]/b",
                 "--print", "url", "--no-warnings", "--skip-download",
                 watch_url(video_id)],
                capture_output=True, text=True, timeout=self.timeout_s)
            if proc.returncode == 0 and proc.stdout.strip().startswith("http"):
                return proc.stdout.strip().splitlines()[0]
        except Exception as e:
            log.warning("stream url resolution failed for %s: %s", video_id, e)
        return None

    def extract_frame(self, stream_url: str, t_seconds: float,
                      out_path: Path) -> bool:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", f"{max(0.0, t_seconds):.2f}", "-i", stream_url,
                 "-frames:v", "1", "-q:v", "3", str(out_path)],
                capture_output=True, text=True, timeout=self.timeout_s)
            return proc.returncode == 0 and out_path.exists()
        except Exception as e:
            log.warning("ffmpeg frame grab failed at %.1fs: %s", t_seconds, e)
            return False

    # ------------------------------------------------------------------
    def attach_frames(self, video_id: str, chunks: list[Chunk]) -> int:
        """Extract one frame per chunk (at its midpoint); returns # extracted.
        Mutates chunk.extra['frame_path'] / ['frame_time']."""
        if not self.available():
            log.info("frame extraction unavailable (need ffmpeg + yt-dlp on PATH)")
            return 0
        stream = self._stream_url(video_id)
        if not stream:
            return 0
        fdir = self.frames_dir(video_id)
        n = 0
        for c in chunks:
            t = (c.start_time + c.end_time) / 2
            out = fdir / f"{c.chunk_id}.jpg"
            if out.exists() or self.extract_frame(stream, t, out):
                c.extra["frame_path"] = str(out)
                c.extra["frame_time"] = round(t, 2)
                n += 1
        log.info("extracted %d/%d frames for %s", n, len(chunks), video_id)
        return n
