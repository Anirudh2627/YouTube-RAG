"""Chat endpoint: grounded Q&A over ingested video(s)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_container
from app.models.schemas import ChatRequest, ChatResponse
from app.services.container import Container

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse,
             summary="Ask a question grounded in video transcripts")
def chat(req: ChatRequest, c: Container = Depends(get_container)) -> ChatResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if req.video_id and c.videos.get_meta(req.video_id) is None:
        raise HTTPException(status_code=404,
                            detail=f"video {req.video_id!r} not ingested")
    try:
        return c.engine.answer(
            query=req.query,
            video_id=req.video_id,
            conversation_id=req.conversation_id,
            top_k=req.top_k, top_n=req.top_n,
            debug=req.debug or c.settings.debug,
        )
    except Exception as e:  # never leak stack traces to clients
        raise HTTPException(status_code=500, detail=f"chat failed: {e}")


@router.get("/health", summary="Liveness + pipeline summary")
def health(c: Container = Depends(get_container)) -> dict:
    return {
        "status": "ok",
        "embedding_model": c.embedder.model_name,
        "vectorstore": c.store.backend,
        "llm": f"{c.llm.name}:{c.llm.model}",
        "reranker": c.reranker.name if c.reranker else "disabled",
        "videos": c.store.video_ids(),
        "chunks": c.store.count(),
    }
