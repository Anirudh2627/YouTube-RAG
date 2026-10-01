#!/usr/bin/env python3
"""Interactive CLI chat against the RAG pipeline (no server needed).

    python scripts/chat.py fixture:data/fixtures/demo_lecture.json
    python scripts/chat.py <video_id>            # already ingested
    python scripts/chat.py --all                 # cross-video mode

Commands inside the REPL:  /debug on|off · /video <id>|all · /quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.container import build_container     # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with a video")
    p.add_argument("source", nargs="?", default=None,
                   help="fixture:<path>, YouTube URL, or ingested video_id")
    p.add_argument("--all", action="store_true", help="cross-video retrieval")
    args = p.parse_args()

    c = build_container()
    video_id = None if args.all else "AUTO"

    if args.source:
        if args.source.startswith("fixture:") or "youtu" in args.source:
            meta, cached, _ = c.videos.process(args.source)
            video_id = None if args.all else meta.video_id
            print(f"[ingested {meta.video_id} — {meta.title[:60]}{' (cached)' if cached else ''}]")
        else:
            video_id = None if args.all else args.source

    debug = False
    conv_id = None
    print("\nAsk questions about the video. /debug on · /video <id>|all · /quit\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q.startswith("/debug"):
            debug = q.endswith("on")
            print(f"[debug {'on' if debug else 'off'}]")
            continue
        if q.startswith("/video"):
            arg = q.split(maxsplit=1)[1] if " " in q else "all"
            video_id = None if arg == "all" else arg
            print(f"[scope: {arg}]")
            continue

        resp = c.engine.answer(q, video_id=video_id,
                               conversation_id=conv_id, debug=debug)
        conv_id = resp.conversation_id
        print(f"\nassistant> {resp.answer_markdown}\n")
        if debug and resp.debug:
            d = resp.debug
            print(f"  [rewrite] {d.rewritten_query or '(none)'}")
            print(f"  [mode]    {d.retrieval_mode}")
            for sc in d.reranked:
                print(f"  [ctx] {sc.chunk.label} vec={sc.vector_score} "
                      f"bm25={sc.bm25_score} fused={sc.fused_score} "
                      f"rerank={sc.rerank_score}")
            print(f"  [timings] {d.timings_ms}\n")


if __name__ == "__main__":
    main()
