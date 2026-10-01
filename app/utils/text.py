"""Text normalization utilities for transcript cleaning.

Auto-generated captions are noisy: repeated music tags, `[Applause]`,
ALL-CAPS shouting, run-on whitespace, duplicated segments (YouTube's
rolling-caption artifacts). We clean conservatively — the goal is to
remove noise without destroying content the retriever needs.
"""
from __future__ import annotations

import re
import unicodedata

# Bracketed stage directions: [Music], (applause), ___, etc.
_BRACKET_RE = re.compile(r"[\[\(](?:music|applause|laughter|laughing|clapping|"
                         r"intro|outro|inaudible|crosstalk|silence|pause|"
                         r"blank audio|foreign language|upbeat [a-z ]*|"
                         r"video plays|no audio)[\]\)]", re.IGNORECASE)
_ANY_BRACKET_SHORT_RE = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]")
_MUSIC_NOTE_RE = re.compile(r"[♪♫]+")
_WS_RE = re.compile(r"\s+")
_MULTI_PUNCT_RE = re.compile(r"([!?.,;:])\1{2,}")
_URL_RE = re.compile(r"https?://\S+")


def normalize_unicode(text: str) -> str:
    """NFKC-normalize and strip zero-width/control characters."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t ")


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def remove_artifacts(text: str) -> str:
    """Remove caption artifacts (music tags, applause markers, URLs)."""
    text = _BRACKET_RE.sub(" ", text)
    text = _MUSIC_NOTE_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return text


def fix_shouting(text: str) -> str:
    """Auto-captions sometimes emit whole segments in CAPS; downcase them
    (sentence-case) unless the segment is short or contains acronyms."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return text
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    if upper_ratio > 0.8:
        # keep known acronyms intact
        return re.sub(
            r"\b[A-Z]{2,}\b|\S+",
            lambda m: m.group(0) if m.group(0).isupper() and len(m.group(0)) <= 6
            else m.group(0).lower(),
            text,
        )
    return text


def clean_segment_text(text: str, aggressive: bool = False) -> str:
    """Full cleaning pass for one transcript segment."""
    text = normalize_unicode(text)
    text = remove_artifacts(text)
    text = fix_shouting(text)
    if aggressive:
        text = _ANY_BRACKET_SHORT_RE.sub(" ", text)
    text = _MULTI_PUNCT_RE.sub(r"\1", text)
    text = normalize_whitespace(text)
    return text


def dedupe_consecutive(segments: list[dict]) -> list[dict]:
    """Drop exact-duplicate consecutive segments (rolling-caption artifact).

    `segments` items are dicts with at least a `text` key; other keys are kept.
    """
    out: list[dict] = []
    prev = None
    for seg in segments:
        norm = normalize_whitespace(seg.get("text", ""))
        if norm and norm == prev:
            continue
        if norm:
            out.append(seg)
            prev = norm
    return out
