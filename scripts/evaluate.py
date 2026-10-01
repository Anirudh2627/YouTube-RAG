#!/usr/bin/env python3
"""Run the RAG evaluation harness.

Examples:
    # production-ish config (BGE embeddings + hybrid + cross-encoder rerank),
    # offline heuristic judge:
    python scripts/evaluate.py --embedding bge

    # fast CI config:
    python scripts/evaluate.py --embedding hashing --no-rerank --tag ci

    # LLM-as-a-judge (needs LLM_API_KEY):
    python scripts/evaluate.py --embedding bge --judge llm --tag llmjudge
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings                       # noqa: E402
from app.evaluation.runner import run_evaluation, save_report  # noqa: E402
from app.services.container import build_container     # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate retrieval + generation")
    p.add_argument("--dataset", default="data/eval/demo_dataset.json")
    p.add_argument("--embedding", choices=["bge", "hashing"], default="bge")
    p.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--judge", choices=["heuristic", "llm"], default="heuristic")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--top-n", type=int, default=4)
    p.add_argument("--tag", default="")
    p.add_argument("--store", choices=["chroma", "memory"], default="memory")
    args = p.parse_args()

    settings = Settings(
        embedding_provider="sentence-transformers" if args.embedding == "bge" else "hashing",
        embedding_dim=384 if args.embedding == "bge" else 256,
        vectorstore_backend=args.store,
        chroma_persist=args.store == "chroma",
        retrieval_mode=args.mode,
        reranker_enabled=not args.no_rerank,
        llm_provider="groq" if (args.judge == "llm" and Settings().llm_api_key) else "mock",
    )
    container = build_container(settings)

    report = run_evaluation(container, args.dataset,
                            eval_llm_judge=(args.judge == "llm"),
                            top_k=args.top_k)
    jpath, mpath = save_report(report, "data/eval/reports", tag=args.tag)

    print()
    print(report.console_summary())
    print(f"\nreport saved: {jpath}\n              {mpath}")


if __name__ == "__main__":
    main()
