"""Deterministic offline LLM for tests, CI, and key-less demos.

It is *extractive*, not generative: it selects the sentences from the
provided context that best match the query (IDF-weighted token overlap, so
rare informative words dominate common ones) and stitches them into an
answer with [Cn] citation markers. Selected sentences are extended by one
following sentence, because narration usually states a topic and then
substantiates it.

This exercises the whole pipeline end-to-end (prompt construction → citation
parsing → grounding behavior) without a network call, and it can never
hallucinate beyond the context — the failure mode real LLMs have.

It also implements the refusal and off-topic behaviors from the system
prompt so those paths are testable too.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from app.generation.base import LLM
from app.retrieval.bm25 import tokenize as _content_tokenize
from app.utils.stemming import stem as _stem

REFUSAL = "I couldn't find enough information about that in the video."
OFF_TOPIC = ("This assistant answers questions about the content of the loaded "
             "YouTube video(s). Your question appears to be unrelated to them.")

_CTX_BLOCK_RE = re.compile(r"\[C(\d+)\]\s*\(([^)]*)\)\s*(.*?)(?=\n\[C\d+\]|\Z)", re.S)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(s: str) -> set[str]:
    """Stemmed content tokens (stopwords stripped)."""
    return {_stem(t) for t in _content_tokenize(s)}


class MockLLM(LLM):
    name = "mock"
    model = "mock-extractive-v1"

    def complete(self, messages: list[dict], temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        user_msg = next((m["content"] for m in reversed(messages)
                         if m["role"] == "user"), "")
        query = ""
        m = re.search(r"QUESTION:\s*(.*?)\s*(?:CONTEXT|CONVERSATION|$)", user_msg, re.S)
        if m:
            query = m.group(1).strip()

        blocks = _CTX_BLOCK_RE.findall(user_msg)
        if not blocks:
            return REFUSAL

        # flatten to (marker, block_sentences, sent_idx)
        sentences: list[tuple[str, list[str], int]] = []
        for n, _label, text in blocks:
            sents = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
            for i in range(len(sents)):
                sentences.append((f"[C{n}]", sents, i))
        if not sentences:
            return REFUSAL

        # sentence-level IDF so rare query words (e.g. "optimizer") outweigh
        # common ones (e.g. "recommend") when picking the best sentence
        sent_toks = [_tokenize(sents[idx]) for _, sents, idx in sentences]
        df: Counter = Counter()
        for toks in sent_toks:
            df.update(toks)
        N = len(sentences)
        idf = {t: math.log(1 + N / (1 + c)) for t, c in df.items()}

        q_toks = _tokenize(query)
        q_idf = sum(idf.get(t, math.log(1 + N)) for t in q_toks) or 1.0

        best_per_marker: dict[str, tuple[float, float, str]] = {}
        raw_best = 0.0
        for (marker, sents, idx), toks in zip(sentences, sent_toks):
            matched = q_toks & toks
            if not matched:
                continue
            ratio = len(matched) / len(q_toks)
            raw_best = max(raw_best, ratio)
            # acceptance gate: strong coverage, or moderate coverage with
            # several distinct matches (guards 1-word coincidences)
            acceptable = ratio >= 0.3 or (ratio >= 0.2 and len(matched) >= 3)
            if not acceptable:
                continue
            wscore = sum(idf.get(t, math.log(1 + N)) for t in matched) / q_idf
            window = " ".join(sents[idx: idx + 2])   # topic + substantiation
            cur = best_per_marker.get(marker)
            if cur is None or wscore > cur[0]:
                best_per_marker[marker] = (wscore, ratio, window)

        ranked = sorted(best_per_marker.items(), key=lambda kv: kv[1][0], reverse=True)
        parts: list[tuple[str, str]] = []          # (window, marker)
        seen: set[str] = set()
        for marker, (_ws, _r, window) in ranked:
            if window in seen:                    # dedupe overlapping chunks
                continue
            seen.add(window)
            parts.append((window, marker))
            if len(parts) >= 3:
                break
        if parts:
            return " ".join(f"{w} {mk}" for w, mk in parts)

        # nothing matched: "topic present but answer absent" → refusal;
        # no lexical connection at all → off-topic
        if raw_best >= 0.12:
            return REFUSAL
        return OFF_TOPIC
