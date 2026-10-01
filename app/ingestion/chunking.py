"""Timestamp-aware chunking.

Why not "split every N characters"?
  * It cuts mid-sentence, producing fragments that embed poorly and confuse
    the LLM ("...and the reason we do this is be" | "cause attention lets").
  * It ignores the timeline: a chunk must map back to a contiguous video
    interval so citations are meaningful.

Strategy:
  1. Walk caption segments in timeline order, accumulating text.
  2. When the accumulated token count passes the target, cut at the last
     sentence boundary — never mid-sentence unless a single sentence exceeds
     the hard max.
  3. Interpolate sub-segment timestamps: inside one caption segment, a
     character offset maps linearly onto [start, end], so every sentence
     (even one spanning segments) gets an accurate start/end time.
  4. Overlap by whole sentences (≈ chunk_overlap_tokens) so answers that
     straddle a boundary are still retrievable from one chunk.

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

#utilities

_SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])"
)

_PLACEHOLDER = "\uFF0E"

# Titles never end a sentence → shield the final dot entirely.
_TITLES = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "st.",
    "jr.",
    "sr.",
    "fig.",
    "no.",
}

# These CAN end a sentence ("... at 3 p.m. Then ...")
# → shield inner dots only.
_ENDERS = {
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "p.m.",
    "a.m.",
}


def approx_token_count(text: str) -> int:
    """Rough token estimate: ~1 token per word + a few for punctuation."""
    words = text.split()
    return sum(1 + (len(word) > 12) for word in words)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Abbreviations are shielded before splitting:
      - titles keep their dot ("Dr. Smith" stays one sentence)
      - enders keep only their final dot as a potential boundary
        ("3 p.m. Then" splits, "e.g. attention" doesn't)
    """
    protected = text

    # Protect titles such as "Dr." and "Mr."
    for abbreviation in sorted(_TITLES, key=len, reverse=True):
        protected = re.sub(
            re.escape(abbreviation),
            lambda match: match.group(0)[:-1] + _PLACEHOLDER,
            protected,
            flags=re.IGNORECASE,
        )

    # Protect internal dots in abbreviations such as "e.g." and "p.m."
    for abbreviation in sorted(_ENDERS, key=len, reverse=True):
        protected = re.sub(
            re.escape(abbreviation),
            lambda match: (
                match.group(0)[:-1].replace(".", _PLACEHOLDER) + "."
            ),
            protected,
            flags=re.IGNORECASE,
        )

    parts = _SENT_SPLIT_RE.split(protected)

    return [
        part.replace(_PLACEHOLDER, ".").strip()
        for part in parts
        if part.strip()
    ]


class _TimeIndex:
    """Map a global character offset to a transcript timestamp.

    The transcript is treated as one concatenated string. Each caption
    segment owns a character range, and offsets inside that range are
    linearly interpolated between the segment's start and end time.
    """

    def __init__(self, segments: list[TranscriptSegment]):
        self._segments = segments
        self._offsets: list[int] = []

        offset = 0

        for segment in segments:
            self._offsets.append(offset)
            offset += len(segment.text) + 1  # +1 for joining space

        self.total_chars = offset

    def char_to_time(self, offset: int) -> float:
        """Convert a transcript character offset into a timestamp."""
        import bisect

        index = bisect.bisect_right(self._offsets, offset) - 1

        index = max(
            0,
            min(index, len(self._segments) - 1),
        )

        segment = self._segments[index]

        local_offset = offset - self._offsets[index]

        fraction = local_offset / max(1, len(segment.text))

        duration = (
            segment.duration
            if segment.duration > 0
            else 1.0
        )

        return round(
            segment.start + min(1.0, fraction) * duration,
            3,
        )


def _chunk_id(
    video_id: str,
    index: int,
    text: str,
) -> str:
    """Generate a deterministic chunk ID."""
    digest = hashlib.sha1(
        f"{video_id}|{index}|{text[:64]}".encode()
    ).hexdigest()[:12]

    return f"{video_id}-{index:04d}-{digest}"


# ----------------------------------------------------------------- chunker


class Chunker(ABC):
    """Base class for transcript chunkers."""

    strategy: str = "base"

    def __init__(
        self,
        target_tokens: int = 160,
        max_tokens: int = 240,
        overlap_tokens: int = 40,
        min_tokens: int = 24,
    ):
        self.target = target_tokens
        self.max = max_tokens
        self.overlap = overlap_tokens
        self.min = min_tokens

    def chunk(
        self,
        segments: list[TranscriptSegment],
        meta: VideoMeta,
    ) -> list[Chunk]:
        """Create chunks and normalize their indexes and IDs.

        Very small tail fragments are merged into neighboring chunks.
        """
        chunks = self._chunk_raw(segments, meta)

        chunks = self._merge_tiny(chunks)

        for index, chunk in enumerate(chunks):
            chunk.index = index
            chunk.chunk_id = _chunk_id(
                meta.video_id,
                index,
                chunk.text,
            )

        return chunks

    @abstractmethod
    def _chunk_raw(
        self,
        segments: list[TranscriptSegment],
        meta: VideoMeta,
    ) -> list[Chunk]:
        """Implement the actual chunking algorithm."""
        raise NotImplementedError

    def _merge_tiny(
        self,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Merge chunks smaller than the minimum token count.

        The previous chunk is preferred so that timestamps remain naturally
        anchored to the earlier content.
        """
        merged: list[Chunk] = []

        for chunk in chunks:
            # Small chunk after an existing chunk:
            # merge it into the previous chunk.
            if (
                merged
                and approx_token_count(chunk.text) < self.min
            ):
                previous = merged[-1]

                previous.text = (
                    previous.text + " " + chunk.text
                ).strip()

                previous.end_time = max(
                    previous.end_time,
                    chunk.end_time,
                )

            # First chunk is tiny.
            elif (
                not merged
                and approx_token_count(chunk.text) < self.min
            ):
                merged.append(chunk)

            else:
                # If the first chunk was tiny and we now have a second
                # chunk, merge the first into the second.
                if (
                    len(merged) == 1
                    and approx_token_count(merged[0].text) < self.min
                ):
                    chunk.text = (
                        merged[0].text + " " + chunk.text
                    ).strip()

                    chunk.start_time = min(
                        chunk.start_time,
                        merged[0].start_time,
                    )

                    merged = [chunk]

                else:
                    merged.append(chunk)

        # A single tiny chunk is fine for a very short video.
        return merged

    def _make_chunk(
        self,
        meta: VideoMeta,
        index: int,
        text: str,
        start: float,
        end: float,
        extra: dict | None = None,
    ) -> Chunk:
        """Create a Chunk object with common metadata."""
        return Chunk(
            chunk_id=_chunk_id(
                meta.video_id,
                index,
                text,
            ),
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
    """Timestamp-aware sentence-based chunker.

    This is the only chunking strategy used by the application.

    It:
      - preserves sentence boundaries
      - preserves video timestamps
      - supports sentence-level overlap
      - handles sentences spanning multiple caption segments
      - produces deterministic chunk IDs
    """

    strategy = "timestamp"

    def _chunk_raw(
        self,
        segments: list[TranscriptSegment],
        meta: VideoMeta,
    ) -> list[Chunk]:
        if not segments:
            return []

        time_index = _TimeIndex(segments)

        # -----------------------------------------------------------
        # Build a flat list of:
        #
        #     (sentence_text, character_offset)
        #
        # This allows every sentence to be mapped back to the
        # original transcript timeline.
        # -----------------------------------------------------------

        sentences: list[tuple[str, int]] = []

        offset = 0
        buffer = ""
        buffer_offset = 0

        for segment in segments:
            piece = (
                (buffer + " " + segment.text).strip()
                if buffer
                else segment.text
            )

            if not buffer:
                buffer_offset = offset

            complete_sentences = split_sentences(piece)

            # The final sentence may be incomplete.
            # Keep it in the buffer until the next segment.
            for sentence in complete_sentences[:-1]:
                sentences.append(
                    (
                        sentence,
                        buffer_offset + piece.find(sentence),
                    )
                )

            tail = (
                complete_sentences[-1]
                if complete_sentences
                else ""
            )

            # If the final part ends with punctuation, it is complete.
            if tail and re.search(r"[.!?]$", tail):
                sentences.append(
                    (
                        tail,
                        buffer_offset + piece.find(tail),
                    )
                )

                buffer = ""
                buffer_offset = offset + len(segment.text) + 1

            else:
                # Keep incomplete sentence for the next segment.
                buffer = piece
                buffer_offset = buffer_offset

            offset += len(segment.text) + 1

        # Add any remaining transcript text.
        if buffer.strip():
            sentences.append(
                (
                    buffer.strip(),
                    buffer_offset,
                )
            )

        # -----------------------------------------------------------
        # Pack sentences into chunks.
        # -----------------------------------------------------------

        chunks: list[Chunk] = []

        i = 0

        while i < len(sentences):
            texts: list[str] = []
            token_count = 0

            j = i

            while j < len(sentences):
                sentence, _ = sentences[j]

                sentence_tokens = approx_token_count(sentence)

                # Once we have content, don't exceed target size.
                if (
                    texts
                    and token_count + sentence_tokens > self.target
                ):
                    break

                texts.append(sentence)

                token_count += sentence_tokens

                j += 1

                # Hard maximum.
                if token_count >= self.max:
                    break

            text = " ".join(texts)

            # Start timestamp.
            start = time_index.char_to_time(
                sentences[i][1]
            )

            # End timestamp.
            last_sentence = sentences[j - 1]

            end_offset = (
                last_sentence[1]
                + len(last_sentence[0])
            )

            end = time_index.char_to_time(
                min(
                    end_offset,
                    time_index.total_chars - 1,
                )
            )

            # Ensure the chunk has a non-zero duration.
            end = max(
                end,
                start + 0.5,
            )

            chunks.append(
                self._make_chunk(
                    meta=meta,
                    index=len(chunks),
                    text=text,
                    start=start,
                    end=end,
                )
            )

            # -------------------------------------------------------
            # Sentence-level overlap.
            #
            # Move backwards from the end of the current chunk until
            # we have approximately overlap_tokens worth of text.
            # -------------------------------------------------------

            back = j
            overlap = 0

            while (
                back > i + 1
                and overlap < self.overlap
            ):
                back -= 1

                overlap += approx_token_count(
                    sentences[back][0]
                )

            i = (
                back
                if overlap > 0
                else j
            )

        return chunks


def get_chunker(**kwargs) -> Chunker:
    """Return the application's timestamp-aware chunker.

    There is intentionally no strategy parameter anymore.
    TimestampChunker is the single production chunking strategy.
    """
    return TimestampChunker(**kwargs)