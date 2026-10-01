"""VideoService: ingestion orchestration + caching.

    URL → video_id → cache hit? ──yes──→ load cached meta, reuse vector store
                        │no
                        ↓
        fetch transcript (provider chain) → clean → chunk → embed → store
                        ↓
        persist meta.json / segments.json / chunks.json under
        data/cache/<video_id>/  (the cache key is the video id)

The on-disk cache means restarts are instant and re-ingestion only happens
on `force=True` or when the cache is missing/corrupt. The BM25 side-index
rebuilds lazily from stored documents — no second source of truth.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.embeddings.base import Embedder
from app.ingestion.base import TranscriptProvider
from app.ingestion.chunking import get_chunker
from app.ingestion.cleaning import clean_segments
from app.ingestion.file_provider import FileTranscriptProvider
from app.ingestion.youtube_transcript import (TranscriptUnavailableError,
                                              YouTubeTranscriptProvider)
from app.models.schemas import Chunk, VideoMeta
from app.utils.logging import get_logger
from app.utils.youtube import extract_video_id, watch_url
from app.vectorstore.base import VectorStore

log = get_logger(__name__)


class VideoService:
    def __init__(self, settings: Settings, embedder: Embedder,
                 store: VectorStore,
                 providers: list[TranscriptProvider] | None = None,
                 on_ingested=None):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.providers = providers if providers is not None else self._default_providers()
        self.on_ingested = on_ingested   # callback(video_id) e.g. BM25 invalidation

    def _default_providers(self) -> list[TranscriptProvider]:
        providers: list[TranscriptProvider] = [YouTubeTranscriptProvider()]
        try:
            import os
            from app.ingestion.ytdlp_provider import YtDlpProvider
            providers.append(YtDlpProvider(cookies_file=os.environ.get("YTDLP_COOKIES_FILE")))
        except Exception:
            pass
        return providers

    # ------------------------------------------------------------------
    def cache_dir(self, video_id: str) -> Path:
        return Path(self.settings.cache_dir) / video_id

    def get_meta(self, video_id: str) -> VideoMeta | None:
        f = self.cache_dir(video_id) / "meta.json"
        if not f.exists():
            return None
        try:
            return VideoMeta(**json.loads(f.read_text()))
        except Exception:
            log.warning("corrupt cache meta for %s; ignoring", video_id)
            return None

    def get_chunks(self, video_id: str) -> list[Chunk]:
        f = self.cache_dir(video_id) / "chunks.json"
        if not f.exists():
            return []
        return [Chunk(**c) for c in json.loads(f.read_text())]

    def list_videos(self) -> list[VideoMeta]:
        metas = []
        root = Path(self.settings.cache_dir)
        if root.exists():
            for d in sorted(root.iterdir()):
                meta = self.get_meta(d.name)
                if meta:
                    metas.append(meta)
        return metas

    # ------------------------------------------------------------------
    def process(self, url_or_path: str, language: str | None = "en",
                force: bool = False,
                ) -> tuple[VideoMeta, bool, float]:
        """Returns (meta, cached, elapsed_seconds)."""
        t0 = time.perf_counter()

        # fixture/file support for offline demos & tests: "fixture:<path>"
        provider_override: TranscriptProvider | None = None
        if url_or_path.startswith("fixture:"):
            provider_override = FileTranscriptProvider(url_or_path.split(":", 1)[1])
            # resolve the real video id from the fixture meta so the cache key
            # matches what gets persisted (fall back to the filename stem)
            try:
                video_id = provider_override.fetch().meta.video_id
            except Exception:
                video_id = provider_override.path.stem[:11]
        else:
            video_id = extract_video_id(url_or_path)

        # ---- cache check
        if not force:
            meta = self.get_meta(video_id)
            if meta and self.store.count(video_id) > 0:
                return meta, True, time.perf_counter() - t0

        # ---- fetch transcript (provider chain, or override)
        result = None
        errors: list[str] = []
        chain = [provider_override] if provider_override else self.providers
        for provider in chain:
            try:
                result = provider.fetch(video_id, language)
                break
            except TranscriptUnavailableError as e:
                errors.append(f"{provider.name}: {e}")
                log.warning("transcript provider %s failed: %s", provider.name, e)
        if result is None:
            raise TranscriptUnavailableError(
                "All transcript providers failed:\n  " + "\n  ".join(errors))

        # ---- enrich metadata (best effort via yt-dlp)
        meta = result.meta
        if not meta.title and not provider_override:
            meta = self._enrich_meta(meta)
        meta.url = watch_url(meta.video_id)

        # ---- clean → chunk → embed → store
        segments = clean_segments(result.segments)
        chunker = get_chunker(
        target_tokens=self.settings.chunk_target_tokens,
        max_tokens=self.settings.chunk_max_tokens,
        overlap_tokens=self.settings.chunk_overlap_tokens,
)
        chunks = chunker.chunk(segments, meta)
        if not chunks:
            raise TranscriptUnavailableError(f"Chunking produced nothing for {video_id}")

        log.info("embedding %d chunks for %s ...", len(chunks), meta.video_id)
        vectors = self.embedder.embed([c.text for c in chunks])

        self.store.delete_video(meta.video_id)   # idempotent re-ingest
        self.store.add(
            ids=[c.chunk_id for c in chunks],
            vectors=vectors,
            texts=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
        )

        # ---- persist cache artifacts
        meta.n_segments = len(segments)
        meta.n_chunks = len(chunks)
        meta.ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cdir = self.cache_dir(meta.video_id)
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "meta.json").write_text(meta.model_dump_json(indent=2))
        (cdir / "segments.json").write_text(json.dumps(
            [s.model_dump() for s in segments]))
        (cdir / "chunks.json").write_text(json.dumps(
            [c.model_dump() for c in chunks]))

        if self.on_ingested:
            self.on_ingested(meta.video_id)

        return meta, False, time.perf_counter() - t0

    # ------------------------------------------------------------------
    def _enrich_meta(self, meta: VideoMeta) -> VideoMeta:
        """Best-effort title/channel/duration via yt-dlp (no download)."""
        try:
            from app.ingestion.ytdlp_provider import YtDlpProvider
            info = YtDlpProvider(timeout_s=30)._fetch_metadata(meta.video_id)
            if info:
                meta.title = info.title or meta.title
                meta.channel = info.channel or meta.channel
                meta.duration_s = info.duration_s or meta.duration_s
        except Exception as e:
            log.warning("metadata enrichment failed: %s", e)
        return meta

    def delete(self, video_id: str) -> bool:
        meta = self.get_meta(video_id)
        self.store.delete_video(video_id)
        cdir = self.cache_dir(video_id)
        if cdir.exists():
            import shutil
            shutil.rmtree(cdir, ignore_errors=True)
        if self.on_ingested:
            self.on_ingested(video_id)
        return meta is not None
