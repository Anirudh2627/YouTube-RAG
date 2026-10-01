from __future__ import annotations

from app.models.schemas import ScoredChunk
from app.utils.timefmt import format_timestamp

REFUSAL_SENTENCE = "I couldn't find enough information about that in the video."

SYSTEM_PROMPT = f"""\
You are a precise assistant that answers questions about YouTube videos using
ONLY the transcript excerpts provided to you.

Rules:
1. Ground every statement in the provided context. Never use outside
   knowledge to add facts, numbers, or claims that the transcript does not
   contain. If your background knowledge disagrees with the transcript,
   follow the transcript.
2. Cite sources inline with the context markers, e.g. [C1] or [C2][C3].
   Every factual sentence must carry at least one citation.
3. If the context does not contain the answer, reply EXACTLY:
   "{REFUSAL_SENTENCE}"
   Do not guess, do not partially answer, do not apologize beyond that.
4. If the question is unrelated to the video content (small talk, other
   topics, requests to do unrelated tasks), explain that you are designed to
   answer questions about the current video(s) and invite a video-related
   question.
5. Prefer short, well-structured answers. Use markdown lists for
   enumerations. Keep the speaker's terminology.
6. When the video spans multiple videos in the context, attribute claims to
   the right video title in your prose."""

REWRITE_SYSTEM_PROMPT = """\
You rewrite a follow-up question into a standalone search query.

Given the recent conversation about a YouTube video and the user's new
question, output ONE self-contained search query that could be understood
without the conversation. Resolve pronouns ("it", "that", "they") and
ellipses using the conversation. Keep the user's intent and important
technical terms unchanged. If the question is already standalone, output it
verbatim. Output ONLY the rewritten query — no explanations, no quotes."""


def build_context_block(scored_chunks: list[ScoredChunk],
                        max_chars_per_chunk: int = 1600) -> str:
    parts = []
    for i, sc in enumerate(scored_chunks, start=1):
        c = sc.chunk
        header = (f"[C{i}] ({format_timestamp(c.start_time)}–"
                  f"{format_timestamp(c.end_time)})")
        if c.title:
            header += f" from \"{c.title}\""
        text = c.text[:max_chars_per_chunk]
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def build_user_message(query: str, context: str,
                       history_snippet: str | None = None) -> str:
    parts = []
    if history_snippet:
        parts.append(f"CONVERSATION SO FAR (for reference only):\n{history_snippet}")
    parts.append(f"QUESTION: {query}")
    parts.append(f"CONTEXT (transcript excerpts):\n{context}")
    parts.append("Answer the QUESTION using only the CONTEXT, with [Cn] citations.")
    return "\n\n".join(parts)
