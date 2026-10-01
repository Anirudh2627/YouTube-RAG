"""Core data models shared across the pipeline and exposed by the API.

Keeping one canonical set of Pydantic models means ingestion → storage →
retrieval → generation → API all speak the same language, and the API
schema is never out of sync with internal representations.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.utils.timefmt import format_timestamp
from app.utils.youtube import watch_url


class TranscriptSegment(BaseModel):
    """One caption segment as extracted from YouTube."""
    text: str
    start: float          # seconds
    duration: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration


class VideoMeta(BaseModel):
    """Metadata about an ingested video."""
    video_id: str
    url: str
    title: str = ""
    channel: str = ""
    duration_s: float | None = None
    language: str | None = None
    n_segments: int = 0
    n_chunks: int = 0
    source: Literal["youtube-transcript-api", "yt-dlp", "file", "fixture"] = "youtube-transcript-api"
    ingested_at: str = ""


class Chunk(BaseModel):
    """A timestamp-aware chunk — the atomic unit of retrieval."""
    chunk_id: str
    video_id: str
    video_url: str
    title: str = ""
    text: str
    start_time: float
    end_time: float
    index: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"[{format_timestamp(self.start_time)}–{format_timestamp(self.end_time)}]"

    def metadata(self) -> dict[str, Any]:
        """Flat metadata dict suitable for vector-store payloads."""
        meta = {
            "chunk_id": self.chunk_id,
            "video_id": self.video_id,
            "video_url": self.video_url,
            "title": self.title,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "index": self.index,
        }
        if self.extra.get("frame_path"):
            meta["frame_path"] = self.extra["frame_path"]
            meta["frame_time"] = self.extra.get("frame_time", self.start_time)
        return meta


class ScoredChunk(BaseModel):
    """A chunk with retrieval/rerank scores attached (for debug mode)."""
    chunk: Chunk
    vector_score: float | None = None     # cosine similarity in [-1, 1]
    bm25_score: float | None = None
    fused_score: float | None = None      # e.g. RRF score
    rerank_score: float | None = None     # cross-encoder logit (sigmoided 0..1)
    final_rank: int | None = None

    @property
    def text(self) -> str:
        return self.chunk.text


class Source(BaseModel):
    """A citation presented to the user."""
    video_id: str
    title: str = ""
    start_time: float
    end_time: float
    text: str
    url: str = ""
    timestamp_label: str = ""
    score: float | None = None
    frame_path: str | None = None      # multimodal: keyframe for this moment

    @classmethod
    def from_scored(cls, sc: ScoredChunk) -> "Source":
        c = sc.chunk
        score = sc.rerank_score if sc.rerank_score is not None else (
            sc.fused_score if sc.fused_score is not None else sc.vector_score
        )
        return cls(
            video_id=c.video_id,
            title=c.title,
            start_time=c.start_time,
            end_time=c.end_time,
            text=c.text,
            url=watch_url(c.video_id, int(c.start_time)),
            timestamp_label=format_timestamp(c.start_time),
            score=score,
            frame_path=c.extra.get("frame_path"),
        )


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# ------------------------------------------------------------------ API I/O

class ProcessVideoRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL or bare video id")
    language: str | None = Field(None, description="Transcript language code, e.g. 'en'")
    force: bool = Field(False, description="Re-ingest even if cached")
    chunk_strategy: Literal["timestamp", "sentence", "token"] | None = None


class ProcessPlaylistRequest(BaseModel):
    url: str = Field(..., description="YouTube playlist URL")
    language: str | None = None
    limit: int = Field(20, ge=1, le=50, description="Max videos to ingest")
    force: bool = False


class PlaylistItemResult(BaseModel):
    video: VideoMeta | None = None
    error: str | None = None
    cached: bool = False
    elapsed_s: float = 0.0


class ProcessPlaylistResponse(BaseModel):
    n_requested: int
    results: list[PlaylistItemResult]


class ProcessVideoResponse(BaseModel):
    video: VideoMeta
    cached: bool
    elapsed_s: float


class ChatRequest(BaseModel):
    query: str
    video_id: str | None = Field(None, description="Restrict to one video; omit to search the whole collection")
    conversation_id: str | None = Field(None, description="Omit to start a new conversation")
    top_k: int | None = Field(None, description="Override stage-1 candidate count")
    top_n: int | None = Field(None, description="Override reranked context size")
    debug: bool = Field(False, description="Include full retrieval trace in the response")


class DebugTrace(BaseModel):
    """Everything needed to explain why an answer was produced."""
    original_query: str
    rewritten_query: str | None = None
    retrieval_mode: str = ""
    candidates: list[ScoredChunk] = Field(default_factory=list)   # stage 1
    reranked: list[ScoredChunk] = Field(default_factory=list)     # stage 2
    final_context: str = ""
    llm_model: str = ""
    timings_ms: dict[str, float] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    answer_markdown: str = ""      # same answer with clickable [mm:ss] citations
    sources: list[Source] = Field(default_factory=list)
    cited_indices: list[int] = Field(default_factory=list,
                                     description="0-based indices into sources that the answer actually cited")
    conversation_id: str
    video_ids: list[str] = Field(default_factory=list)
    debug: DebugTrace | None = None
