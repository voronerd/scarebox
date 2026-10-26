"""`python -m scarebox` -- CLI entrypoint. Spec §8.1: `scarebox run`."""

from __future__ import annotations

import argparse
import asyncio
import logging

from scarebox.app import main as app_main
from scarebox.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scarebox")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline end to end")
    run.add_argument("--config", required=True, help="path to a scarebox YAML config")
    run.add_argument(
        "--mode",
        default="live",
        choices=["live", "library"],
        help="pipeline mode -- 'live' (03-03 default) runs the full judge/quip/censor/voice "
        "orchestrator with a library fallback; 'library' is the pre-baked-line-only subset",
    )
    run.add_argument(
        "--tts-placeholder",
        action="store_true",
        help="synthesise a 0.5s beep instead of a rendered library line (bench only, no "
        "voiced lines until 02-03)",
    )
    run.add_argument(
        "--max-events", type=int, default=None, help="stop after this many events (testing)"
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.command == "run":
        cfg = load_config(args.config)
        asyncio.run(
            app_main(
                cfg,
                mode=args.mode,
                tts_placeholder=args.tts_placeholder,
                max_events=args.max_events,
            )
        )


if __name__ == "__main__":
    main()
