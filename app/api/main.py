"""FastAPI application factory.

Run with:  uvicorn app.api.main:app --reload
Interactive docs at /docs (Swagger) and /redoc.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_chat import router as chat_router
from app.api.routes_videos import router as videos_router
from app.config import get_settings

DESCRIPTION = """
Grounded question-answering over YouTube video transcripts.

Pipeline: transcript extraction → cleaning → timestamp-aware chunking →
embeddings (BGE) → ChromaDB → hybrid retrieval (vector + BM25, RRF) →
cross-encoder reranking → grounded LLM answer with clickable timestamp
citations.
"""


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="YouTube RAG Assistant",
        version="0.1.0",
        description=DESCRIPTION,
        debug=settings.debug,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten in production
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(videos_router)
    app.include_router(chat_router)
    return app


app = create_app()
