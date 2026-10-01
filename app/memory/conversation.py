"""Conversation store with TTL.

In-memory is the right default for a single-process demo service; the
interface is storage-agnostic so Redis/Postgres can back it in production
without touching the engine.
"""
from __future__ import annotations

import threading
import time
import uuid

from app.models.schemas import ConversationTurn


class ConversationStore:
    def __init__(self, ttl_minutes: int = 120):
        self._ttl_s = ttl_minutes * 60
        self._lock = threading.Lock()
        self._data: dict[str, tuple[float, list[ConversationTurn]]] = {}

    def create_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def get(self, conversation_id: str) -> list[ConversationTurn]:
        with self._lock:
            self._evict()
            return list(self._data.get(conversation_id, (0, []))[1])

    def append(self, conversation_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            self._evict()
            ts, turns = self._data.get(conversation_id, (time.time(), []))
            turns = list(turns) + [turn]
            # cap history length so memory stays bounded
            self._data[conversation_id] = (time.time(), turns[-40:])

    def history_snippet(self, conversation_id: str, max_turns: int = 4) -> str | None:
        turns = self.get(conversation_id)[-max_turns:]
        if not turns:
            return None
        return "\n".join(f"{t.role.upper()}: {t.content}" for t in turns)

    def _evict(self) -> None:
        now = time.time()
        stale = [k for k, (ts, _) in self._data.items() if now - ts > self._ttl_s]
        for k in stale:
            del self._data[k]
