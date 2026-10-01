"""Query rewriting: make follow-up questions standalone before retrieval.

    history + "why is it useful?"  →  "why is the attention mechanism useful?"

Primary path: a small, cheap LLM (e.g. llama-3.1-8b-instant on Groq) with a
strict one-line-output prompt. Fallback path (no key / LLM failure): a
conservative heuristic that only rewrites when the query is *detectably*
context-dependent (short + starts with a pronoun/determiner/wh-fragment),
by appending the topic of the previous user question. Rewriting must never
make things worse — when in doubt, pass the query through.
"""
from __future__ import annotations

import re

from app.generation.base import LLM
from app.generation.prompts import REWRITE_SYSTEM_PROMPT
from app.models.schemas import ConversationTurn
from app.utils.logging import get_logger

log = get_logger(__name__)

_CONTEXT_DEPENDENT_START = re.compile(
    r"^\s*(why|how|what|when|where|which|who|is|are|does|do|did|can|could|"
    r"should|would|will|it|its|this|that|these|those|they|them|their|he|she|"
    r"the same|same|more|less|and|but|also|any|others?)\b", re.IGNORECASE)

_ANAPHORA_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|him|her|his|"
    r"same|also|more|else|again|there)\b", re.IGNORECASE)


def needs_rewrite(query: str, history: list[ConversationTurn]) -> bool:
    """Heuristic: rewrite only when the query is *detectably* context-dependent.

    Signals: explicit anaphora ("it", "that", "those", "more"...), or a very
    short question (≤6 words) starting with a wh-/auxiliary word. Longer
    wh-questions without anaphora are treated as standalone — rewriting them
    could only hurt.
    """
    if not history:
        return False
    q = query.strip()
    words = q.split()
    if len(words) > 14:
        return False
    if _ANAPHORA_RE.search(q):
        return True
    return len(words) <= 6 and bool(_CONTEXT_DEPENDENT_START.match(q))


class QueryRewriter:
    def __init__(self, llm: LLM | None = None, enabled: bool = True,
                 max_history_turns: int = 4):
        self.llm = llm
        self.enabled = enabled
        self.max_history_turns = max_history_turns

    # ------------------------------------------------------------------
    def rewrite(self, query: str, history: list[ConversationTurn]) -> tuple[str, str | None]:
        """Returns (effective_query, method) where method ∈
        {None, 'llm', 'heuristic', 'passthrough'}."""
        if not self.enabled or not needs_rewrite(query, history):
            return query, None
        if self.llm is not None:
            try:
                return self._rewrite_llm(query, history), "llm"
            except Exception as e:
                log.warning("LLM rewrite failed (%s); using heuristic", e)
        return self._rewrite_heuristic(query, history), "heuristic"

    # ------------------------------------------------------------------
    def _rewrite_llm(self, query: str, history: list[ConversationTurn]) -> str:
        snippet = "\n".join(
            f"{t.role.upper()}: {t.content[:400]}" for t in history[-self.max_history_turns:]
        )
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user",
             "content": f"CONVERSATION:\n{snippet}\n\nNEW QUESTION: {query}\n\nSTANDALONE QUERY:"},
        ]
        out = self.llm.complete(messages, temperature=0.0, max_tokens=80)
        # sanitize: one line, no quotes/prefixes
        out = out.strip().strip('"').splitlines()[0].strip()
        out = re.sub(r"^(standalone query|rewritten query|query)\s*[:\-]\s*", "", out,
                     flags=re.IGNORECASE)
        if not 3 <= len(out.split()) <= 60:   # nonsense guard → passthrough
            return query
        return out

    def _rewrite_heuristic(self, query: str, history: list[ConversationTurn]) -> str:
        prev_user = next((t.content for t in reversed(history) if t.role == "user"), None)
        if not prev_user:
            return query
        topic = prev_user.strip().rstrip("?").strip()
        if len(topic.split()) > 20:
            topic = " ".join(topic.split()[:20])
        return f"{query.rstrip('?')} (in the context of this earlier question: {topic})"
