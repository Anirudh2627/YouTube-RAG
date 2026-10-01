"""
The rewriter converts short follow-up questions into retrieval-friendly
queries using the previous user question as context.

Example:
    "Why is it useful?"
        ->
    "Why is it useful (in the context of this earlier question:
    what is the attention mechanism)"
"""

from __future__ import annotations

import re

from app.models.schemas import ConversationTurn


_CONTEXT_DEPENDENT_START = re.compile(
    r"^\s*(why|how|what|when|where|which|who|is|are|does|do|did|can|could|"
    r"should|would|will|it|its|this|that|these|those|they|them|their|he|she|"
    r"the same|same|more|less|and|but|also|any|others?)\b",
    re.IGNORECASE,
)

_ANAPHORA_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|him|her|his|"
    r"same|also|more|else|again|there)\b",
    re.IGNORECASE,
)


def needs_rewrite(
    query: str,
    history: list[ConversationTurn],
) -> bool:
    """Return True when a query is likely dependent on conversation history.

    Rewriting is attempted only when:
    - conversation history exists,
    - the query is reasonably short, and
    - it contains an explicit contextual reference or looks like a
      short follow-up question.

    Longer standalone questions are passed through unchanged.
    """
    if not history:
        return False

    q = query.strip()
    words = q.split()

    if len(words) > 14:
        return False

    if _ANAPHORA_RE.search(q):
        return True

    return len(words) <= 6 and bool(
        _CONTEXT_DEPENDENT_START.match(q)
    )


class QueryRewriter:
    """Rewrite context-dependent follow-up questions using conversation history."""

    def __init__(
        self,
        enabled: bool = True,
        max_history_turns: int = 4,
    ):
        self.enabled = enabled
        self.max_history_turns = max_history_turns

    def rewrite(
        self,
        query: str,
        history: list[ConversationTurn],
    ) -> tuple[str, str | None]:
        """Return (effective_query, method).

        method is one of:
            None       - no rewriting was required
            heuristic  - heuristic contextual rewriting
            passthrough - reserved for future explicit passthrough handling
        """
        if not self.enabled or not needs_rewrite(query, history):
            return query, None

        return self._rewrite_heuristic(query, history), "heuristic"

    def _rewrite_heuristic(
        self,
        query: str,
        history: list[ConversationTurn],
    ) -> str:
        """Append the previous user question as retrieval context."""
        recent_history = history[-self.max_history_turns :]

        prev_user = next(
            (
                turn.content
                for turn in reversed(recent_history)
                if turn.role == "user"
            ),
            None,
        )

        if not prev_user:
            return query

        topic = prev_user.strip().rstrip("?").strip()

        if len(topic.split()) > 20:
            topic = " ".join(topic.split()[:20])

        return (
            f"{query.rstrip('?')} "
            f"(in the context of this earlier question: {topic})"
        )