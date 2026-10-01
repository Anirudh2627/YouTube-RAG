"""Video endpoints: processing, listing, inspection, sources."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_container
from app.ingestion.youtube_transcript import TranscriptUnavailableError
from app.models.schemas import (Chunk, PlaylistItemResult,
                                ProcessPlaylistRequest,
                                ProcessPlaylistResponse, ProcessVideoRequest,
                                ProcessVideoResponse, VideoMeta)
from app.services.container import Container
from app.utils.youtube import InvalidYouTubeURLError

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("/process", response_model=ProcessVideoResponse,
             summary="Ingest a YouTube video (or serve from cache)")
def process_video(req: ProcessVideoRequest,
                  c: Container = Depends(get_container)) -> ProcessVideoResponse:
    try:
        meta, cached, elapsed = c.videos.process(
            req.url, language=req.language, force=req.force,
            chunk_strategy=req.chunk_strategy,
        )
    except InvalidYouTubeURLError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TranscriptUnavailableError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return ProcessVideoResponse(video=meta, cached=cached, elapsed_s=round(elapsed, 2))


@router.post("/process-playlist", response_model=ProcessPlaylistResponse,
             summary="Ingest every video of a playlist into the shared store")
def process_playlist(req: ProcessPlaylistRequest,
                     c: Container = Depends(get_container)) -> ProcessPlaylistResponse:
    try:
        raw = c.videos.process_playlist(req.url, language=req.language,
                                        limit=req.limit, force=req.force)
    except TranscriptUnavailableError as e:
        raise HTTPException(status_code=422, detail=str(e))
    results = [
        PlaylistItemResult(video=m, cached=cached, elapsed_s=round(el, 2))
        if not isinstance(m, str) else PlaylistItemResult(error=m)
        for m, cached, el in raw
    ]
    return ProcessPlaylistResponse(n_requested=len(results), results=results)


@router.get("", response_model=list[VideoMeta], summary="List ingested videos")
def list_videos(c: Container = Depends(get_container)) -> list[VideoMeta]:
    return c.videos.list_videos()


@router.get("/{video_id}", response_model=VideoMeta,
            summary="Metadata for one ingested video")
def get_video(video_id: str, c: Container = Depends(get_container)) -> VideoMeta:
    meta = c.videos.get_meta(video_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"video {video_id!r} not ingested")
    return meta


@router.get("/{video_id}/sources", response_model=list[Chunk],
            summary="All timestamped chunks stored for a video")
def get_sources(video_id: str, c: Container = Depends(get_container)) -> list[Chunk]:
    if c.videos.get_meta(video_id) is None:
        raise HTTPException(status_code=404, detail=f"video {video_id!r} not ingested")
    return c.videos.get_chunks(video_id)


@router.delete("/{video_id}", status_code=204, summary="Delete a video and its data")
def delete_video(video_id: str, c: Container = Depends(get_container)) -> None:
    if not c.videos.delete(video_id):
        raise HTTPException(status_code=404, detail=f"video {video_id!r} not found")
