"""The RAG engine: end-to-end query orchestration.

    query → QueryRewriter → RetrievalPipeline → context blocks → LLM
          → citation post-processing → ChatResponse (+ DebugTrace)

Citation handling is deliberately two-layered:
  * The LLM writes [C1]-style markers next to claims (cheap, verifiable).
  * `render_citations` maps each marker to a real timestamped YouTube URL.
    Markers pointing outside the provided sources are dropped, and any
    unused source is still listed under "Relevant sections" — so the UI can
    always render clickable timestamps even if the model forgets a marker.
"""
from __future__ import annotations

import re
import time

from app.generation.base import LLM
from app.generation.prompts import (REFUSAL_SENTENCE, SYSTEM_PROMPT,
                                    build_context_block, build_user_message)
from app.generation.rewriter import QueryRewriter
from app.memory.conversation import ConversationStore
from app.models.schemas import (ChatResponse, ConversationTurn, DebugTrace,
                                ScoredChunk, Source)
from app.retrieval.pipeline import RetrievalPipeline
from app.utils.logging import get_logger
from app.utils.timefmt import format_timestamp
from app.utils.youtube import watch_url

log = get_logger(__name__)

_CITATION_RE = re.compile(r"\[C(\d+)\]")


class RAGEngine:
    def __init__(self, retriever: RetrievalPipeline, llm: LLM,
                 conversations: ConversationStore,
                 rewriter: QueryRewriter | None = None,
                 max_context_chunks: int = 6,
                 include_history_in_prompt: int = 2):
        self.retriever = retriever
        self.llm = llm
        self.conversations = conversations
        self.rewriter = rewriter or QueryRewriter(llm=None, enabled=False)
        self.max_context_chunks = max_context_chunks
        self.include_history_in_prompt = include_history_in_prompt

    # ------------------------------------------------------------------
    def answer(self, query: str, video_id: str | None = None,
               conversation_id: str | None = None,
               top_k: int | None = None, top_n: int | None = None,
               debug: bool = False) -> ChatResponse:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        # ---- conversation setup
        conv_id = conversation_id or self.conversations.create_id()
        history = self.conversations.get(conv_id)

        # ---- 1. query rewriting (follow-up → standalone)
        t = time.perf_counter()
        rewritten, rewrite_method = self.rewriter.rewrite(query, history)
        timings["rewrite"] = (time.perf_counter() - t) * 1000

        # ---- 2. retrieval
        result = self.retriever.retrieve(rewritten, video_id=video_id,
                                         top_k=top_k, top_n=top_n)
        final_chunks = result.final[: self.max_context_chunks]
        timings.update(result.timings_ms)

        # ---- 3. grounded generation
        t = time.perf_counter()
        sources = [Source.from_scored(sc) for sc in final_chunks]
        answer = self._generate(query, rewritten, final_chunks, history, timings)
        timings["llm"] = (time.perf_counter() - t) * 1000
        timings["total"] = (time.perf_counter() - t0) * 1000

        # ---- 4. citations → clickable timestamps
        answer_md, cited = render_citations(answer, sources)
        sources_md = render_sources_section(sources, cited)
        if sources_md:
            answer_md = f"{answer_md}\n\n{sources_md}"

        # ---- 5. persist turn (store the clean answer, not the markdown)
        self.conversations.append(conv_id, ConversationTurn(role="user", content=query))
        self.conversations.append(conv_id, ConversationTurn(role="assistant",
                                                            content=strip_markers(answer)))

        resp = ChatResponse(
            answer=strip_markers(answer),
            answer_markdown=answer_md,
            sources=sources,
            cited_indices=sorted(cited),
            conversation_id=conv_id,
            video_ids=sorted({s.video_id for s in sources}),
        )
        if debug:
            resp.debug = DebugTrace(
                original_query=query,
                rewritten_query=rewritten if rewritten != query else None,
                retrieval_mode=result.mode + (f"+{rewrite_method}" if rewrite_method else ""),
                candidates=result.fused,
                reranked=final_chunks,
                final_context=build_context_block(final_chunks),
                llm_model=self.llm.model or self.llm.name,
                timings_ms={k: round(v, 1) for k, v in timings.items()},
            )
        return resp

    # ------------------------------------------------------------------
    def _generate(self, query: str, rewritten: str,
                  chunks: list[ScoredChunk],
                  history: list[ConversationTurn],
                  timings: dict[str, float]) -> str:
        if not chunks:
            return REFUSAL_SENTENCE
        context = build_context_block(chunks)
        snippet = None
        if self.include_history_in_prompt and history:
            turns = history[-self.include_history_in_prompt * 2:]
            snippet = "\n".join(f"{t.role.upper()}: {t.content[:300]}" for t in turns)
        user_msg = build_user_message(rewritten or query, context, snippet)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}]
        try:
            return self.llm.complete(messages)
        except Exception as e:
            log.error("LLM generation failed: %s", e)
            return REFUSAL_SENTENCE


# ------------------------------------------------------------ citation utils

def strip_markers(text: str) -> str:
    """Remove [Cn] markers, tidy the whitespace they leave behind."""
    text = _CITATION_RE.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def render_citations(answer: str, sources: list[Source]) -> tuple[str, set[int]]:
    """Replace [Cn] markers with clickable `[mm:ss](watch-url)` links.

    Returns (markdown, set of 0-based source indices actually cited).
    Out-of-range markers are silently dropped (LLM mistakes happen).
    """
    cited: set[int] = set()

    def repl(m: re.Match) -> str:
        n = int(m.group(1)) - 1
        if 0 <= n < len(sources):
            cited.add(n)
            s = sources[n]
            label = format_timestamp(s.start_time)
            if len({x.video_id for x in sources}) > 1 and s.title:
                # multi-video: make clear which video the timestamp is from
                short = s.title if len(s.title) <= 40 else s.title[:37] + "..."
                label = f"{label} · {short}"
            return f"[{label}]({s.url})"
        return ""

    md = _CITATION_RE.sub(repl, answer)
    md = re.sub(r"[ \t]{2,}", " ", md).strip()
    return md, cited


def render_sources_section(sources: list[Source], cited: set[int]) -> str:
    """Render the '**Relevant sections:**' block with clickable timestamps."""
    if not sources:
        return ""
    lines = ["**Relevant sections:**"]
    for i, s in enumerate(sources):
        mark = "↳ cited" if i in cited else ""
        title_part = f" — *{s.title}*" if (
            s.title and len({x.video_id for x in sources}) > 1) else ""
        score = f" (score {s.score:.3f})" if s.score is not None else ""
        lines.append(f"- [{s.timestamp_label}]({s.url}){title_part} {mark}{score}".rstrip())
    return "\n".join(lines)
