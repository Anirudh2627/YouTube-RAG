"""Deterministic generation evaluation for the RAG system.

The heuristic judge is offline and reproducible. It evaluates:
- answer correctness
- faithfulness
- context relevance
- citation accuracy
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.evaluation.metrics import GoldItem, chunk_is_relevant
from app.generation.prompts import REFUSAL_SENTENCE
from app.retrieval.bm25 import tokenize
from app.utils.logging import get_logger
from app.utils.stemming import stem

log = get_logger(__name__)

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class JudgeInput:
    question: str
    expected_answer: str
    expected_refusal: bool = False
    gold: list[GoldItem] = field(default_factory=list)

    answer: str = ""

    # (text, start, end, video_id)
    context_chunks: list[tuple[str, float, float, str]] = field(
        default_factory=list
    )

    # (start, end, video_id)
    cited_spans: list[tuple[float, float, str]] = field(
        default_factory=list
    )


@dataclass
class JudgeScore:
    answer_correctness: float
    faithfulness: float
    context_relevance: float
    citation_accuracy: float
    refused: bool = False
    refusal_correct: bool | None = None
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in self.__dict__.items()
        }


class Judge(ABC):
    """Base interface for generation judges."""

    name = "base"

    @abstractmethod
    def score(self, ji: JudgeInput) -> JudgeScore:
        ...


class HeuristicJudge(Judge):
    """Lexical, deterministic and offline generation evaluator."""

    name = "heuristic"

    def __init__(self, support_threshold: float = 0.6):
        self.support_threshold = support_threshold

    def score(self, ji: JudgeInput) -> JudgeScore:
        refused = (
            REFUSAL_SENTENCE.lower() in ji.answer.lower()
            or (
                "unrelated" in ji.answer.lower()
                and len(ji.answer) < 200
            )
        )

        # Handle questions where refusal is the expected behaviour.
        if ji.expected_refusal:
            return JudgeScore(
                answer_correctness=1.0 if refused else 0.0,
                faithfulness=1.0 if refused else 0.0,
                context_relevance=0.0,
                citation_accuracy=(
                    1.0
                    if refused and not ji.cited_spans
                    else 0.0
                    if not refused
                    else 1.0
                ),
                refused=refused,
                refusal_correct=refused,
                detail="refusal expected",
            )

        # Penalise refusing an answerable question.
        if refused:
            return JudgeScore(
                answer_correctness=0.0,
                faithfulness=0.0,
                context_relevance=self._context_relevance(ji),
                citation_accuracy=0.0,
                refused=True,
                refusal_correct=None,
                detail="refused an answerable question",
            )

        correctness = self._keyword_coverage(
            ji.answer,
            ji.expected_answer,
        )

        faithfulness = self._faithfulness(
            ji.answer,
            ji.context_chunks,
        )

        context_relevance = self._context_relevance(ji)

        citation_accuracy = self._citation_accuracy(ji)

        return JudgeScore(
            answer_correctness=correctness,
            faithfulness=faithfulness,
            context_relevance=context_relevance,
            citation_accuracy=citation_accuracy,
            refused=False,
            refusal_correct=None,
        )

    @staticmethod
    def _keyword_coverage(
        answer: str,
        expected: str,
    ) -> float:
        """Measure how many expected answer tokens appear in the answer."""

        expected_tokens = {
            stem(token)
            for token in tokenize(expected)
        }

        if not expected_tokens:
            return 0.0

        answer_tokens = {
            stem(token)
            for token in tokenize(answer)
        }

        return len(expected_tokens & answer_tokens) / len(expected_tokens)

    def _faithfulness(
        self,
        answer: str,
        context: list[tuple[str, float, float, str]],
    ) -> float:
        """Measure how many answer sentences are supported by context."""

        context_tokens = [
            set(tokenize(text))
            for text, *_ in context
        ]

        if not context_tokens:
            return 0.0

        sentences = [
            sentence
            for sentence in _SENT_RE.split(answer)
            if len(tokenize(sentence)) >= 3
        ]

        if not sentences:
            return 0.0

        supported = 0

        for sentence in sentences:
            sentence_tokens = set(tokenize(sentence))

            if any(
                len(sentence_tokens & context_tokens_chunk)
                / max(1, len(sentence_tokens))
                >= self.support_threshold
                for context_tokens_chunk in context_tokens
            ):
                supported += 1

        return supported / len(sentences)

    @staticmethod
    def _context_relevance(ji: JudgeInput) -> float:
        """Measure how many retrieved chunks overlap with the question."""

        if not ji.context_chunks:
            return 0.0

        question_tokens = set(tokenize(ji.question))

        if not question_tokens:
            return 0.0

        relevant = 0

        for text, *_ in ji.context_chunks:
            context_tokens = set(tokenize(text))

            if (
                len(question_tokens & context_tokens)
                / len(question_tokens)
                >= 0.3
            ):
                relevant += 1

        return relevant / len(ji.context_chunks)

    @staticmethod
    def _citation_accuracy(ji: JudgeInput) -> float:
        """Average citation precision and citation coverage."""

        if not ji.gold:
            return 1.0 if not ji.cited_spans else 0.0

        if not ji.cited_spans:
            return 0.0

        good_citations = [
            span
            for span in ji.cited_spans
            if chunk_is_relevant(
                span[0],
                span[1],
                ji.gold,
                span[2],
            )
        ]

        precision = len(good_citations) / len(ji.cited_spans)

        coverage = 1.0 if good_citations else 0.0

        return (precision + coverage) / 2
