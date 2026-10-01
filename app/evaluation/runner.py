"""Evaluation runner: retrieval + generation metrics over a QA dataset.

Produces honest, reproducible numbers:
  * every retrieval score is computed from actual vector-store results
  * every generation score comes from an actual judge over actual answers
  * the full config + per-question detail is written into the report, so
    any number in the README can be re-derived with one command.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.judges import HeuristicJudge, Judge, JudgeInput
from app.evaluation.metrics import (GoldItem, aggregate, chunk_is_relevant,
                                    relevance_vector)
from app.services.container import Container
from app.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class QuestionResult:
    id: str
    question: str
    answerable: bool
    retrieved_spans: list[tuple[float, float]] = field(default_factory=list)   # stage 1
    final_spans: list[tuple[float, float]] = field(default_factory=list)       # stage 2
    answer: str = ""
    cited_spans: list[tuple[float, float, str]] = field(default_factory=list)
    judge: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    dataset: str
    config: dict[str, Any]
    retrieval_stage1: dict[str, float] = field(default_factory=dict)
    retrieval_stage2: dict[str, float] = field(default_factory=dict)
    generation: dict[str, float] = field(default_factory=dict)
    judge_name: str = ""
    n_questions: int = 0
    per_question: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""

    def console_summary(self) -> str:
        lines = [
            f"=== Evaluation report: {self.dataset} ({self.n_questions} questions) ===",
            f"judge: {self.judge_name} | config: {json.dumps(self.config)}",
            "",
            "Retrieval — stage 1 (candidates, pre-rerank):",
        ]
        lines += [f"  {k}: {v}" for k, v in self.retrieval_stage1.items() if k != "n_questions"]
        lines.append("Retrieval — stage 2 (final, post-rerank):")
        lines += [f"  {k}: {v}" for k, v in self.retrieval_stage2.items() if k != "n_questions"]
        lines.append("Generation:")
        lines += [f"  {k}: {v}" for k, v in self.generation.items()]
        return "\n".join(lines)


def run_evaluation(
    container: Container,
    dataset_path: str | Path,
    judge: Judge | None = None,
    top_k: int = 10,
) -> EvalReport:
    dataset_path = Path(dataset_path)
    ds = json.loads(dataset_path.read_text())
    video_source = ds["video_source"]
    video_id = ds.get("video_id")

    # ---- ingest (uses cache when present)
    meta, cached, elapsed = container.videos.process(video_source)
    log.info("eval video %s (cached=%s, %.1fs)", meta.video_id, cached, elapsed)
    corpus_chunks = container.videos.get_chunks(meta.video_id)

    # ---- judge selection
    if judge is None:
        judge = HeuristicJudge()

    per_q_results: list[QuestionResult] = []
    stage1_rel: list[tuple[list[int], int]] = []
    stage2_rel: list[tuple[list[int], int]] = []
    judge_scores: list[dict[str, float]] = []

    for q in ds["questions"]:
        qr = QuestionResult(id=q["id"], question=q["question"],
                            answerable=q.get("answerable", True))
        gold = [GoldItem(video_id=meta.video_id, start=g["start"], end=g["end"])
                for g in q.get("gold_spans", [])]

        # -------- retrieval (answerable questions only)
        if qr.answerable and gold:
            n_rel_corpus = sum(
                chunk_is_relevant(c.start_time, c.end_time, gold, meta.video_id)
                for c in corpus_chunks)
            res = container.retriever.retrieve(q["question"], video_id=meta.video_id,
                                               top_k=top_k)
            qr.retrieved_spans = [(sc.chunk.start_time, sc.chunk.end_time)
                                  for sc in res.fused]
            qr.final_spans = [(sc.chunk.start_time, sc.chunk.end_time)
                              for sc in res.final]
            spans1 = [(s, e, meta.video_id) for s, e in qr.retrieved_spans]
            spans2 = [(s, e, meta.video_id) for s, e in qr.final_spans]
            stage1_rel.append((relevance_vector(spans1, gold), n_rel_corpus))
            stage2_rel.append((relevance_vector(spans2, gold), n_rel_corpus))

        # -------- generation (all questions, incl. negatives)
        resp = container.engine.answer(q["question"], video_id=meta.video_id)
        qr.answer = resp.answer
        qr.cited_spans = [(s.start_time, s.end_time, s.video_id)
                          for i, s in enumerate(resp.sources) if i in resp.cited_indices]
        ji = JudgeInput(
            question=q["question"],
            expected_answer=q.get("expected_answer", ""),
            expected_refusal=not qr.answerable,
            gold=gold,
            answer=resp.answer,
            context_chunks=[(s.text, s.start_time, s.end_time, s.video_id)
                            for s in resp.sources],
            cited_spans=qr.cited_spans,
        )
        sc = judge.score(ji)
        qr.judge = sc.as_dict()
        judge_scores.append(sc.as_dict())
        per_q_results.append(qr)
        log.info("  %s → correct=%.2f faithful=%.2f cit=%.2f", q["id"],
                 sc.answer_correctness, sc.faithfulness, sc.citation_accuracy)

    # -------- aggregate
    ks = (1, 3, 5, 10)
    m1 = aggregate(stage1_rel, ks)
    m2 = aggregate(stage2_rel, ks)
    gen_keys = ("answer_correctness", "faithfulness", "context_relevance",
                "citation_accuracy")
    gen = {k: round(sum(s[k] for s in judge_scores) / len(judge_scores), 4)
           for k in gen_keys} if judge_scores else {}
    refusals = [s for s in judge_scores if s.get("refusal_correct") is not None]
    if refusals:
        gen["refusal_accuracy"] = round(
            sum(1 for s in refusals if s["refusal_correct"]) / len(refusals), 4)

    report = EvalReport(
        dataset=ds.get("name", dataset_path.name),
        config={
            "embedder": container.embedder.model_name,
            "retrieval_mode": container.retriever.mode,
            "reranker": container.reranker.name if container.reranker else "off",
            "llm": f"{container.llm.name}:{container.llm.model}",
            "top_k": top_k,
            "top_n": container.retriever.top_n,
        },
        retrieval_stage1=m1.summary(),
        retrieval_stage2=m2.summary(),
        generation=gen,
        judge_name=judge.name,
        n_questions=len(ds["questions"]),
        per_question=[asdict(qr) for qr in per_q_results],
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return report


def save_report(report: EvalReport, out_dir: str | Path, tag: str = "") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"report_{tag + '_' if tag else ''}{stamp}"
    jpath = out_dir / f"{base}.json"
    mpath = out_dir / f"{base}.md"
    jpath.write_text(json.dumps(asdict(report), indent=2))

    md = [f"# Evaluation report — {report.dataset}", "",
          f"*Generated: {report.generated_at} · Judge: `{report.judge_name}`*", "",
          "## Configuration", "", "```json",
          json.dumps(report.config, indent=2), "```", "",
          f"## Retrieval ({report.n_questions} questions)", "",
          "| Metric | Stage 1 (candidates) | Stage 2 (post-rerank) |",
          "|---|---|---|"]
    for k in report.retrieval_stage2:
        if k == "n_questions":
            continue
        md.append(f"| {k} | {report.retrieval_stage1.get(k, '—')} | {report.retrieval_stage2[k]} |")
    md += ["", "## Generation", "", "| Metric | Score |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in report.generation.items()]
    md += ["", "## Per-question detail", ""]
    for qr in report.per_question:
        j = qr["judge"]
        md.append(f"### `{qr['id']}` — {qr['question']}")
        md.append("")
        md.append(f"- answerable: {qr['answerable']}")
        if qr["final_spans"]:
            spans = ", ".join(f"{s:.0f}–{e:.0f}s" for s, e in qr["final_spans"][:4])
            md.append(f"- final retrieved spans: {spans}")
        md.append(f"- correctness {j.get('answer_correctness')} · faithfulness "
                  f"{j.get('faithfulness')} · context-rel {j.get('context_relevance')} · "
                  f"citation-acc {j.get('citation_accuracy')}")
        md.append(f"- answer: {qr['answer'][:300]}")
        md.append("")
    mpath.write_text("\n".join(md), encoding="utf-8")
    return jpath, mpath
