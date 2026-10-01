#!/usr/bin/env python3
"""Ingest a video (or fixture) into the RAG store from the command line.

Examples:
    python scripts/ingest.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    python scripts/ingest.py fixture:data/fixtures/demo_lecture.json
    python scripts/ingest.py <url> --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from app.config import Settings  # noqa: E402
from app.services.container import build_container  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a YouTube video"
    )

    parser.add_argument(
        "url",
        help="YouTube URL, bare video id, or fixture:<path>",
    )

    parser.add_argument(
        "--language",
        default="en",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore cache",
    )

    parser.add_argument(
        "--embedding",
        choices=["bge", "hashing"],
        default=None,
    )

    args = parser.parse_args()

    overrides = {}

    if args.embedding == "hashing":
        overrides.update(
            embedding_provider="hashing",
            embedding_dim=256,
        )

    container = build_container(
        Settings(**overrides)
    )

    meta, cached, elapsed = container.videos.process(
        args.url,
        language=args.language,
        force=args.force,
    )

    print(
        f"\n{'(cached) ' if cached else ''}"
        f"ingested in {elapsed:.1f}s"
    )

    print(f"  id:       {meta.video_id}")
    print(f"  title:    {meta.title or '—'}")
    print(f"  channel:  {meta.channel or '—'}")
    print(
        f"  duration: "
        f"{meta.duration_s and int(meta.duration_s // 60)}m"
    )
    print(
        f"  segments: {meta.n_segments} → "
        f"chunks: {meta.n_chunks}"
    )
    print(f"  url:      {meta.url}")


if __name__ == "__main__":
    main()