#!/usr/bin/env python3
"""Run the RAG evaluation harness.

Examples:

    # Production-ish configuration:
    # BGE embeddings + hybrid retrieval + cross-encoder reranking
    python scripts/evaluate.py --embedding bge

    # Fast CI configuration:
    python scripts/evaluate.py --embedding hashing --no-rerank --tag ci
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.evaluation.runner import run_evaluation, save_report  # noqa: E402
from app.services.container import build_container  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval + generation"
    )

    parser.add_argument(
        "--dataset",
        default="data/eval/demo_dataset.json",
    )

    parser.add_argument(
        "--embedding",
        choices=["bge", "hashing"],
        default="bge",
    )

    parser.add_argument(
        "--mode",
        choices=["vector", "bm25", "hybrid"],
        default="hybrid",
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--tag",
        default="",
    )

    parser.add_argument(
        "--store",
        choices=["chroma", "memory"],
        default="memory",
    )

    args = parser.parse_args()

    settings = Settings(
        embedding_provider=(
            "sentence-transformers"
            if args.embedding == "bge"
            else "hashing"
        ),
        embedding_dim=(
            384
            if args.embedding == "bge"
            else 256
        ),
        vectorstore_backend=args.store,
        chroma_persist=args.store == "chroma",
        retrieval_mode=args.mode,
        reranker_enabled=not args.no_rerank,
    )

    container = build_container(settings)

    report = run_evaluation(
        container,
        args.dataset,
        top_k=args.top_k,
    )

    json_path, markdown_path = save_report(
        report,
        "data/eval/reports",
        tag=args.tag,
    )

    print()
    print(report.console_summary())

    print(
        f"\nreport saved: {json_path}\n"
        f"              {markdown_path}"
    )


if __name__ == "__main__":
    main()
