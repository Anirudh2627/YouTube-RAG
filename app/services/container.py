"""Application composition root.

A single factory wires settings → embedder → vector store → BM25/hybrid
retriever → reranker → LLM → rewriter → engine → services. The FastAPI app,
the CLI scripts, and the evaluation harness all build through here, so there
is exactly one definition of "the pipeline" — no drift between entrypoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.embeddings.base import Embedder, get_embedder
from app.generation.base import LLM, get_llm
from app.generation.rag_engine import RAGEngine
from app.generation.rewriter import QueryRewriter
from app.memory.conversation import ConversationStore
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker import Reranker, get_reranker
from app.services.video_service import VideoService
from app.utils.logging import get_logger, setup_logging
from app.vectorstore.base import VectorStore, get_vector_store

log = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    llm: LLM
    reranker: Reranker | None
    retriever: RetrievalPipeline
    conversations: ConversationStore
    engine: RAGEngine
    videos: VideoService
    extras: dict[str, Any] = field(default_factory=dict)


def build_container(settings: Settings | None = None,
                    lazy_llm: bool = False) -> Container:
    """Construct the full pipeline. Heavy components (embedding model,
    cross-encoder) load on first construction; `lazy_llm=True` defers real
    LLM instantiation until an API key exists (falls back to mock)."""
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    embedder = get_embedder(
        settings.embedding_provider, settings.embedding_model,
        dim=settings.embedding_dim, batch_size=settings.embedding_batch_size,
    )

    store = get_vector_store(
        settings.vectorstore_backend,
        persist_dir=str(settings.vectorstore_dir) if settings.chroma_persist else None,
    )

    reranker = get_reranker(
        settings.reranker_enabled,
        settings.reranker_model,
        allow_fallback=False,
    )

    retriever = RetrievalPipeline(
        store=store, embedder=embedder, reranker=reranker,
        mode=settings.retrieval_mode, top_k=settings.retrieval_top_k,
        top_n=settings.rerank_top_n,
        vector_weight=settings.hybrid_vector_weight,
        bm25_weight=settings.hybrid_bm25_weight,
        rrf_k=settings.rrf_k,
    )

    # LLM: use mock when no key is configured so the app still boots offline
    provider = settings.llm_provider
    if provider != "mock" and not settings.llm_api_key:
        log.warning("LLM_API_KEY not set — falling back to MockLLM "
                    "(extractive, offline). Set the key for real answers.")
        provider = "mock"
    llm = get_llm(
        provider, settings.llm_model if provider != "mock" else "mock",
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url_resolved if provider != "mock" else None,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
    )

    conversations = ConversationStore(ttl_minutes=settings.conversation_ttl_minutes)
    rewriter = QueryRewriter(enabled=settings.rewriter_enabled,
                             max_history_turns=settings.history_turns_for_rewrite)
    engine = RAGEngine(retriever=retriever, llm=llm, conversations=conversations,
                       rewriter=rewriter)
    videos = VideoService(settings=settings, embedder=embedder, store=store,
                          on_ingested=retriever.invalidate)

    return Container(settings=settings, embedder=embedder, store=store, llm=llm,
                     reranker=reranker, retriever=retriever,
                     conversations=conversations, engine=engine, videos=videos)
