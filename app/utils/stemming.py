"""Minimal suffix stemmer for lexical matching heuristics.

Deliberately tiny (no NLTK/snowball dependency): strips common English
suffixes so "training"≈"train", "networks"≈"network", "divided"≈"divide".
Used by the offline MockLLM matcher and the HeuristicJudge, where exact-token
matching would otherwise undercount real overlaps. Not used by BM25 or
embeddings (they don't need it / handle morphology their own way).
"""
from __future__ import annotations

_SUFFIXES = ("ing", "ed", "es", "s", "ly")


def stem(tok: str) -> str:
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def stemmed_tokens(text: str, tokenizer) -> set[str]:
    return {stem(t) for t in tokenizer(text)}
