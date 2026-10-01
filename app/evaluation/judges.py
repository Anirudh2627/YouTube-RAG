"""Generation judges: heuristic (offline, deterministic) and LLM-as-a-judge.

Heuristic judge (default in CI / no-API-key mode):
  * answer_correctness  — content-keyword coverage of the expected answer
  * faithfulness        — fraction of answer sentences lexically supported
                          by the retrieved context (proxy for entailment)
  * context_relevance   — fraction of context chunks sharing content with Q
  * citation_accuracy   — fraction of cited sources that overlap a gold
                          interval (precision) and fraction of questions
                          where ≥1 cited source is gold (coverage)

LLM judge:
  Same four axes, scored 0..1 by the configured LLM with a rubric prompt and
  JSON output. Known limitations (documented in README): self-preference
  bias, verbosity bias, position sensitivity, coarse calibration, and
  non-determinism even at temperature 0. It complements — not replaces —
  the deterministic metrics.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.evaluation.metrics import GoldItem, chunk_is_relevant
from app.generation.base import LLM
from app.generation.prompts import REFUSAL_SENTENCE
from app.retrieval.bm25 import tokenize
from app.utils.stemming import stem
from app.utils.logging import get_logger

log = get_logger(__name__)

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class JudgeInput:
    question: str
    expected_answer: str          # "" for unanswerable questions
    expected_refusal: bool = False
    gold: list[GoldItem] = field(default_factory=list)
    answer: str = ""
    context_chunks: list[tuple[str, float, float, str]] = field(default_factory=list)  # (text, start, end, video_id)
    cited_spans: list[tuple[float, float, str]] = field(default_factory=list)          # (start, end, video_id)


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
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


class Judge(ABC):
    name = "base"

    @abstractmethod
    def score(self, ji: JudgeInput) -> JudgeScore: ...


# ------------------------------------------------------------------ heuristic

class HeuristicJudge(Judge):
    """Lexical, deterministic, offline. Weak on paraphrase — a real LLM judge
    should be used for headline numbers when available; this one keeps CI
    honest and reproducible."""
    name = "heuristic"

    def __init__(self, support_threshold: float = 0.6):
        self.support_threshold = support_threshold

    def score(self, ji: JudgeInput) -> JudgeScore:
        refused = REFUSAL_SENTENCE.lower() in ji.answer.lower() or \
            "unrelated" in ji.answer.lower() and len(ji.answer) < 200
        if ji.expected_refusal:
            return JudgeScore(
                answer_correctness=1.0 if refused else 0.0,
                faithfulness=1.0 if refused else 0.0,
                context_relevance=0.0,
                citation_accuracy=1.0 if (refused and not ji.cited_spans) else (0.0 if not refused else 1.0),
                refused=refused, refusal_correct=refused,
                detail="refusal expected",
            )
        if refused:
            return JudgeScore(0.0, 0.0, self._context_relevance(ji), 0.0,
                              refused=True, refusal_correct=None,
                              detail="refused an answerable question")

        correctness = self._keyword_coverage(ji.answer, ji.expected_answer)
        faithfulness = self._faithfulness(ji.answer, ji.context_chunks)
        ctx_rel = self._context_relevance(ji)
        cit_acc = self._citation_accuracy(ji)
        return JudgeScore(correctness, faithfulness, ctx_rel, cit_acc,
                          refused=False, refusal_correct=None)

    # ------------------------------------------------------------------
    @staticmethod
    def _keyword_coverage(answer: str, expected: str) -> float:
        exp = {stem(t) for t in tokenize(expected)}
        if not exp:
            return 0.0
        got = {stem(t) for t in tokenize(answer)}
        return len(exp & got) / len(exp)

    def _faithfulness(self, answer: str,
                      context: list[tuple[str, float, float, str]]) -> float:
        """Fraction of answer sentences supported by some context chunk:
        ≥threshold of the sentence's content tokens appear in that chunk."""
        ctx_tokens = [set(tokenize(t)) for t, *_ in context]
        if not ctx_tokens:
            return 0.0
        sents = [s for s in _SENT_RE.split(answer) if len(tokenize(s)) >= 3]
        if not sents:
            return 0.0
        supported = 0
        for s in sents:
            st = set(tokenize(s))
            if any(len(st & ct) / max(1, len(st)) >= self.support_threshold
                   for ct in ctx_tokens):
                supported += 1
        return supported / len(sents)

    @staticmethod
    def _context_relevance(ji: JudgeInput) -> float:
        if not ji.context_chunks:
            return 0.0
        q = set(tokenize(ji.question))
        rel = 0
        for text, *_ in ji.context_chunks:
            ct = set(tokenize(text))
            if q and len(q & ct) / len(q) >= 0.3:
                rel += 1
        return rel / len(ji.context_chunks)

    @staticmethod
    def _citation_accuracy(ji: JudgeInput) -> float:
        """Mean of (citation precision, citation coverage)."""
        if not ji.gold:
            return 1.0 if not ji.cited_spans else 0.0
        if not ji.cited_spans:
            return 0.0
        good = [s for s in ji.cited_spans
                if chunk_is_relevant(s[0], s[1], ji.gold, s[2])]
        precision = len(good) / len(ji.cited_spans)
        coverage = 1.0 if good else 0.0
        return (precision + coverage) / 2


# ------------------------------------------------------------------ LLM judge

LLM_JUDGE_PROMPT = """\
You are a strict evaluator for a RAG system that answers questions about a
YouTube video from transcript excerpts.

QUESTION: {question}
EXPECTED ANSWER (reference): {expected}
SYSTEM ANSWER: {answer}
RETRIEVED CONTEXT:
{context}
CITED TIMESTAMPS (spans the answer cites): {cited}
GOLD RELEVANT SPANS: {gold}

Score each axis from 0.0 to 1.0:
- answer_correctness: does the answer match the reference's key facts? (0 = wrong/missing, 1 = fully correct)
- faithfulness: is every claim in the answer supported by the retrieved context? (0 = hallucinated, 1 = fully grounded)
- context_relevance: does the retrieved context actually address the question? (0 = irrelevant, 1 = all relevant)
- citation_accuracy: do the cited timestamps fall inside (or very near) the gold spans? (0 = none, 1 = all)

Respond with ONLY a JSON object:
{{"answer_correctness": x, "faithfulness": x, "context_relevance": x, "citation_accuracy": x, "reason": "<= 30 words"}}"""


class LLMJudge(Judge):
    name = "llm"

    def __init__(self, llm: LLM):
        self.llm = llm

    def score(self, ji: JudgeInput) -> JudgeScore:
        refused = REFUSAL_SENTENCE.lower() in ji.answer.lower()
        if ji.expected_refusal:
            return JudgeScore(1.0 if refused else 0.0, 1.0 if refused else 0.0,
                              0.0, 1.0 if refused else 0.0,
                              refused=refused, refusal_correct=refused,
                              detail="refusal expected")
        context = "\n\n".join(
            f"[{s:.0f}s–{e:.0f}s] {t[:800]}" for t, s, e, _ in ji.context_chunks[:6])
        gold = "; ".join(f"{g.start:.0f}-{g.end:.0f}s" for g in ji.gold) or "none"
        cited = "; ".join(f"{s:.0f}-{e:.0f}s" for s, e, _ in ji.cited_spans) or "none"
        prompt = LLM_JUDGE_PROMPT.format(
            question=ji.question, expected=ji.expected_answer or "(none)",
            answer=ji.answer, context=context or "(empty)",
            cited=cited, gold=gold)
        try:
            out = self.llm.complete(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=300)
            m = re.search(r"\{.*\}", out, re.S)
            data = json.loads(m.group(0)) if m else {}
            return JudgeScore(
                answer_correctness=float(data.get("answer_correctness", 0)),
                faithfulness=float(data.get("faithfulness", 0)),
                context_relevance=float(data.get("context_relevance", 0)),
                citation_accuracy=float(data.get("citation_accuracy", 0)),
                refused=refused, refusal_correct=None,
                detail=str(data.get("reason", ""))[:200],
            )
        except Exception as e:
            log.warning("LLM judge failed (%s); falling back to heuristic", e)
            return HeuristicJudge().score(ji)
