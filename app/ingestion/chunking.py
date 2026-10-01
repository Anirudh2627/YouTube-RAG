"""Timestamp-aware chunking.

Why not "split every N characters"?
  * It cuts mid-sentence, producing fragments that embed poorly and confuse
    the LLM ("...and the reason we do this is be" | "cause attention lets").
  * It ignores the timeline: a chunk must map back to a contiguous video
    interval so citations are meaningful.

Strategy (default: `TimestampChunker`)
  1. Walk caption segments in timeline order, accumulating text.
  2. When the accumulated token count passes the target, cut at the last
     sentence boundary — never mid-sentence unless a single sentence exceeds
     the hard max.
  3. Interpolate sub-segment timestamps: inside one caption segment, a
     character offset maps linearly onto [start, end], so every sentence
     (even one spanning segments) gets an accurate start/end time.
  4. Overlap by whole sentences (≈ chunk_overlap_tokens) so answers that
     straddle a boundary are still retrievable from one chunk.

Also provided for experiments/ablations:
  * SentenceChunker — sentence-boundary chunking on the concatenated text.
  * FixedTokenChunker — naive fixed-size baseline (what NOT to ship).

Token counting is approximate (words + punctuation pieces); exact BPE
counts don't matter for chunk sizing, and staying tokenizer-free keeps the
ingestion stage independent of any embedding/LLM vendor.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

from app.models.schemas import Chunk, TranscriptSegment, VideoMeta
from app.utils.youtube import watch_url

# --------------------------------------------------------------- utilities

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_PLACEHOLDER = "\uFF0E"   # unicode full stop used to shield abbreviation dots
# Titles never end a sentence → shield the final dot entirely.
_TITLES = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "fig.", "no."}
# These CAN end a sentence ("... at 3 p.m. Then ...") → shield inner dots only.
_ENDERS = {"e.g.", "i.e.", "etc.", "vs.", "p.m.", "a.m."}


def approx_token_count(text: str) -> int:
    """Rough token estimate: ~1 token per word + a few for punctuation."""
    words = text.split()
    return sum(1 + (len(w) > 12) for w in words)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Abbreviations are shielded before splitting: titles keep their dot
    ("Dr. Smith" stays one sentence), enders keep only their final dot as a
    potential boundary ("3 p.m. Then" splits, "e.g. attention" doesn't —
    the lookahead requires an uppercase/digit/quote continuation).
    """
    protected = text
    for ab in sorted(_TITLES, key=len, reverse=True):
        protected = re.sub(re.escape(ab),
                           lambda m: m.group(0)[:-1] + _PLACEHOLDER,
                           protected, flags=re.IGNORECASE)
    for ab in sorted(_ENDERS, key=len, reverse=True):
        protected = re.sub(re.escape(ab),
                           lambda m: m.group(0)[:-1].replace(".", _PLACEHOLDER) + ".",
                           protected, flags=re.IGNORECASE)
    parts = _SENT_SPLIT_RE.split(protected)
    return [p.replace(_PLACEHOLDER, ".").strip() for p in parts if p.strip()]


class _TimeIndex:
    """Maps a global character offset in the concatenated transcript to a
    timestamp, using linear interpolation inside each caption segment."""

    def __init__(self, segments: list[TranscriptSegment]):
        self._segments = segments
        self._offsets: list[int] = []
        off = 0
        for s in segments:
            self._offsets.append(off)
            off += len(s.text) + 1  # +1 for the joining space
        self.total_chars = off

    def char_to_time(self, offset: int) -> float:
        import bisect
        i = bisect.bisect_right(self._offsets, offset) - 1
        i = max(0, min(i, len(self._segments) - 1))
        seg = self._segments[i]
        local = offset - self._offsets[i]
        frac = local / max(1, len(seg.text))
        dur = seg.duration if seg.duration > 0 else 1.0
        return round(seg.start + min(1.0, frac) * dur, 3)


def _chunk_id(video_id: str, index: int, text: str) -> str:
    h = hashlib.sha1(f"{video_id}|{index}|{text[:64]}".encode()).hexdigest()[:12]
    return f"{video_id}-{index:04d}-{h}"


# ----------------------------------------------------------------- chunkers

class Chunker(ABC):
    strategy: str = "base"

    def __init__(self, target_tokens: int = 160, max_tokens: int = 240,
                 overlap_tokens: int = 40, min_tokens: int = 24):
        self.target = target_tokens
        self.max = max_tokens
        self.overlap = overlap_tokens
        self.min = min_tokens

    # public entry point: template method -------------------------------
    def chunk(self, segments: list[TranscriptSegment], meta: VideoMeta) -> list[Chunk]:
        """Chunk, then merge sub-`min_tokens` tail fragments backwards and
        re-index so chunk ids stay dense and ordered."""
        chunks = self._chunk_raw(segments, meta)
        chunks = self._merge_tiny(chunks)
        for i, c in enumerate(chunks):
            c.index = i
            c.chunk_id = _chunk_id(meta.video_id, i, c.text)
        return chunks

    @abstractmethod
    def _chunk_raw(self, segments: list[TranscriptSegment], meta: VideoMeta) -> list[Chunk]:
        ...

    def _merge_tiny(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge any chunk smaller than self.min tokens into its neighbor
        (preferring the previous chunk, so timestamps stay anchored)."""
        merged: list[Chunk] = []
        for c in chunks:
            if merged and approx_token_count(c.text) < self.min:
                prev = merged[-1]
                prev.text = (prev.text + " " + c.text).strip()
                prev.end_time = max(prev.end_time, c.end_time)
            elif (not merged and approx_token_count(c.text) < self.min):
                merged.append(c)  # first chunk: try to merge with the next one
            else:
                if (len(merged) == 1 and approx_token_count(merged[0].text) < self.min):
                    c.text = (merged[0].text + " " + c.text).strip()
                    c.start_time = min(c.start_time, merged[0].start_time)
                    merged = [c]
                else:
                    merged.append(c)
        # edge case: single tiny chunk left alone is fine (short video)
        return merged

    # shared helpers ---------------------------------------------------
    def _make_chunk(self, meta: VideoMeta, index: int, text: str,
                    start: float, end: float, extra: dict | None = None) -> Chunk:
        return Chunk(
            chunk_id=_chunk_id(meta.video_id, index, text),
            video_id=meta.video_id,
            video_url=meta.url or watch_url(meta.video_id),
            title=meta.title,
            text=text.strip(),
            start_time=round(start, 3),
            end_time=round(end, 3),
            index=index,
            extra=extra or {"strategy": self.strategy},
        )


class TimestampChunker(Chunker):
    """Default: merge caption segments, cut at sentence boundaries, keep
    exact timestamp spans, overlap by whole sentences."""
    strategy = "timestamp"

    def _chunk_raw(self, segments: list[TranscriptSegment], meta: VideoMeta) -> list[Chunk]:
        if not segments:
            return []
        time_idx = _TimeIndex(segments)

        # Build a flat list of (sentence, char_offset) so every sentence can
        # be located on the timeline even when it spans caption segments.
        sentences: list[tuple[str, int]] = []
        offset = 0
        buf = ""
        buf_off = 0
        for s in segments:
            piece = (buf + " " + s.text).strip() if buf else s.text
            if not buf:
                buf_off = offset
            complete = split_sentences(piece)
            # The last "sentence" may be incomplete (no terminal punctuation);
            # keep buffering it with the next segment.
            for sent in complete[:-1]:
                sentences.append((sent, buf_off + piece.find(sent)))
            tail = complete[-1] if complete else ""
            if tail and re.search(r"[.!?]$", tail):
                sentences.append((tail, buf_off + piece.find(tail)))
                buf, buf_off = "", offset + len(s.text) + 1
            else:
                buf, buf_off = piece, buf_off
            offset += len(s.text) + 1
        if buf.strip():
            sentences.append((buf.strip(), buf_off))

        # Greedily pack sentences into chunks with sentence-level overlap.
        chunks: list[Chunk] = []
        i = 0
        while i < len(sentences):
            texts: list[str] = []
            n_tok = 0
            j = i
            while j < len(sentences):
                sent, soff = sentences[j]
                t = approx_token_count(sent)
                if texts and n_tok + t > self.target:
                    break
                texts.append(sent)
                n_tok += t
                j += 1
                if n_tok >= self.max:  # hard cap
                    break
            text = " ".join(texts)
            start = time_idx.char_to_time(sentences[i][1])
            end_off = sentences[j - 1][1] + len(sentences[j - 1][0])
            end = time_idx.char_to_time(min(end_off, time_idx.total_chars - 1))
            chunks.append(self._make_chunk(meta, len(chunks), text, start, max(end, start + 0.5)))

            # overlap: step back so the next chunk re-includes trailing
            # sentences worth ~overlap tokens
            back, ov = j, 0
            while back > i + 1 and ov < self.overlap:
                back -= 1
                ov += approx_token_count(sentences[back][0])
            i = back if ov > 0 else j
        return chunks


class SentenceChunker(Chunker):
    """Concatenate everything, split into sentences, pack with overlap.
    Same idea as TimestampChunker but ignores caption-segment structure —
    useful as an ablation."""
    strategy = "sentence"

    def _chunk_raw(self, segments: list[TranscriptSegment], meta: VideoMeta) -> list[Chunk]:
        full = " ".join(s.text for s in segments)
        time_idx = _TimeIndex(segments)
        # locate each sentence by search offset into the concatenated text
        sents = split_sentences(full)
        offsets, cursor = [], 0
        for sent in sents:
            idx = full.find(sent, cursor)
            offsets.append(idx if idx >= 0 else cursor)
            cursor = (idx if idx >= 0 else cursor) + len(sent)

        chunks, i = [], 0
        while i < len(sents):
            texts, n_tok, j = [], 0, i
            while j < len(sents):
                t = approx_token_count(sents[j])
                if texts and n_tok + t > self.target:
                    break
                texts.append(sents[j]); n_tok += t; j += 1
                if n_tok >= self.max:
                    break
            start = time_idx.char_to_time(offsets[i])
            end = time_idx.char_to_time(min(offsets[j - 1] + len(sents[j - 1]),
                                            time_idx.total_chars - 1))
            chunks.append(self._make_chunk(meta, len(chunks), " ".join(texts),
                                           start, max(end, start + 0.5)))
            back, ov = j, 0
            while back > i + 1 and ov < self.overlap:
                back -= 1; ov += approx_token_count(sents[back])
            i = back if ov > 0 else j
        return chunks


class FixedTokenChunker(Chunker):
    """Naive fixed-size baseline: pack caption segments until N tokens,
    overlap by trailing segments. Ignores sentence boundaries — included
    for ablation studies, not for production use."""
    strategy = "token"

    def _chunk_raw(self, segments: list[TranscriptSegment], meta: VideoMeta) -> list[Chunk]:
        chunks: list[Chunk] = []
        cur: list[TranscriptSegment] = []
        n_tok = 0

        def flush() -> None:
            nonlocal cur, n_tok
            if not cur:
                return
            text = " ".join(s.text for s in cur)
            chunks.append(self._make_chunk(meta, len(chunks), text,
                                           cur[0].start, cur[-1].end or cur[-1].start + 1))
            # overlap: keep trailing segments (~overlap tokens)
            keep: list[TranscriptSegment] = []
            ov = 0
            for s in reversed(cur):
                if ov >= self.overlap:
                    break
                keep.insert(0, s); ov += approx_token_count(s.text)
            cur, n_tok = keep, ov

        for s in segments:
            t = approx_token_count(s.text)
            if cur and n_tok + t > self.target:
                flush()
            cur.append(s); n_tok += t
            if n_tok >= self.max:
                flush()
        flush()
        return chunks


STRATEGIES: dict[str, type[Chunker]] = {
    "timestamp": TimestampChunker,
    "sentence": SentenceChunker,
    "token": FixedTokenChunker,
}


def get_chunker(strategy: str, **kwargs) -> Chunker:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown chunking strategy {strategy!r}; "
                         f"choose from {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](**kwargs)
