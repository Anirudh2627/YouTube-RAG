"""Application configuration.

All settings come from environment variables (optionally loaded from a `.env`
file). Nothing is hardcoded; every tunable knob of the RAG pipeline is exposed
here so components stay replaceable without touching call sites.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ paths
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "cache")
    vectorstore_dir: Path = Field(default=PROJECT_ROOT / "data" / "vectorstore")

    # ------------------------------------------------------------- embeddings
    embedding_provider: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    # ------------------------------------------------------------ vector store
    vectorstore_backend: Literal["chroma", "memory"] = "chroma"
    chroma_persist: bool = True

    # ------------------------------------------------------------- retrieval
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    retrieval_top_k: int = 15          # stage-1 candidates
    rerank_top_n: int = 4              # stage-2 survivors handed to the LLM
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_vector_weight: float = 1.0
    hybrid_bm25_weight: float = 1.0
    rrf_k: int = 60                    # reciprocal-rank-fusion constant

    # --------------------------------------------------------------- chunking
    chunk_target_tokens: int = 160
    chunk_max_tokens: int = 240
    chunk_overlap_tokens: int = 40

    # -------------------------------------------------------------------- LLM
    llm_provider: Literal["groq", "openai-compatible", "mock"] = "groq"
    llm_api_key: str | None = None
    llm_model: str = "llama-3.3-70b-versatile"
    llm_base_url: str | None = None    # override for OpenAI-compatible servers
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 60.0

    # Small/cheap model used for query rewriting (can equal the main model).
    rewriter_model: str = "llama-3.1-8b-instant"
    rewriter_enabled: bool = True

    # ----------------------------------------------------------- conversation
    history_turns_for_rewrite: int = 4
    conversation_ttl_minutes: int = 120

    # ------------------------------------------------------------------- misc
    debug: bool = True
    log_level: str = "INFO"

    # Convenience -----------------------------------------------------------
    @property
    def llm_base_url_resolved(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url.rstrip("/")
        if self.llm_provider == "groq":
            return "https://api.groq.com/openai/v1"
        return "https://api.openai.com/v1"

    @property
    def llm_available(self) -> bool:
        """True when a real LLM can be called (mock provider always is)."""
        return self.llm_provider == "mock" or bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
