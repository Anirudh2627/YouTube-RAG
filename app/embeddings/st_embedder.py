"""Real embeddings via sentence-transformers.

Default model: BAAI/bge-small-en-v1.5 (384-dim, strong MTEB performance for
its size). Configurable to bge-base-en-v1.5, all-MiniLM-L6-v2, etc.

BGE models recommend prefixing *queries* with an instruction for retrieval
("Represent this sentence for searching relevant passages:"). We apply it in
`embed_query` only — documents get no prefix, matching the model card.
"""
from __future__ import annotations

import numpy as np

from app.embeddings.base import Embedder
from app.utils.logging import get_logger

log = get_logger(__name__)

_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        # Lazy model load: importing this module must not download weights.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        log.info("loading embedding model %s ...", model_name)
        self._model = SentenceTransformer(model_name)
        get_dim = getattr(self._model, "get_embedding_dimension", None) \
            or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self._is_bge = "bge" in model_name.lower()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,       # cosine sim == dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        if self._is_bge:
            text = _BGE_QUERY_INSTRUCTION + text
        return self.embed([text])[0]
