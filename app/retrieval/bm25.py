"""Okapi BM25 implemented from scratch (inverted index).

Why hand-rolled? It is ~60 lines, removes a dependency, and keeps the
hybrid-fusion logic fully inspectable — BM25 is one of those components you
should own, not import blindly.

Scoring per query term t and document d:
    idf(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d|/avgdl))
with idf(t) = log(1 + (N - df + 0.5) / (df + 0.5)).
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9']*")

# Small English stopword list — BM25's IDF already downweights frequent
# words, but removing them shrinks the index and helps exact-phrase-ish
# queries like "learning rate".
STOPWORDS = frozenset("""a an and are as at be but by for if in into is it its
no not of on or such that the their then there these they this to was will
with you your we our us can cannot do does did have has had been being am i
me my he she him her his they them its whats what when where who whom how why
than which very also most more some any only so too just now again once here
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._doc_ids: list[str] = []
        self._doc_len: list[int] = []
        self._tf: list[Counter] = []
        self._inv: dict[str, list[int]] = {}   # term -> doc positions
        self._avgdl = 0.0
        self._idf: dict[str, float] = {}

    # ------------------------------------------------------------------
    def build(self, docs: list[tuple[str, str]]) -> None:
        """docs: list of (doc_id, text)."""
        self._doc_ids, self._doc_len, self._tf, self._inv = [], [], [], {}
        for pos, (doc_id, text) in enumerate(docs):
            toks = tokenize(text)
            self._doc_ids.append(doc_id)
            self._doc_len.append(len(toks))
            self._tf.append(Counter(toks))
            for term in set(toks):
                self._inv.setdefault(term, []).append(pos)
        n = len(docs)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0
        self._idf = {
            t: math.log(1 + (n - len(pl) + 0.5) / (len(pl) + 0.5))
            for t, pl in self._inv.items()
        }

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10,
               allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """Returns [(doc_id, score)] sorted desc. `allowed` optionally
        restricts scoring to a subset of doc ids (e.g. one video)."""
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            postings = self._inv.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for pos in postings:
                doc_id = self._doc_ids[pos]
                if allowed is not None and doc_id not in allowed:
                    continue
                f = self._tf[pos][term]
                denom = f + self.k1 * (1 - self.b + self.b * self._doc_len[pos] / (self._avgdl or 1))
                scores[pos] = scores.get(pos, 0.0) + idf * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self._doc_ids[pos], s) for pos, s in ranked]

    def __len__(self) -> int:
        return len(self._doc_ids)
