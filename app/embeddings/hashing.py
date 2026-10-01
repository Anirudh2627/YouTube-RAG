"""Deterministic hashing embedder — for tests and offline CI only.

Maps bag-of-words (unigrams + bigrams) through a stable hash into a fixed
dimensional space, then L2-normalizes. Semantically weak (no synonymy), but
*exact keyword overlap* still yields high cosine similarity, which is enough
to exercise retrieval plumbing deterministically without downloading models.

Never use this in production: set EMBEDDING_PROVIDER=sentence-transformers.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from app.embeddings.base import Embedder

_WORD_RE = re.compile(r"[a-z0-9']+")


class HashingEmbedder(Embedder):
    model_name = "hashing"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = _WORD_RE.findall(text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in self._tokens(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 32) % 2 == 0 else -1.0
                out[row, idx] += sign
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out
