#!/usr/bin/env python3
"""Replay check: FileSource -> Perceiver at real-time pace, on real footage. Spec §4.3;
plan 02-01 Task 3. Saves a few annotated "peek" frames (trigger line + boxes, for eyeballing
the line placement) and the best-crop examples per event, measures detect+track latency and
events/min, and writes the evidence JSON the plan's `<verify>` step checks.

Usage (inside the app container, on gbox1):
    python3 scripts/replay_perceive.py --config config/bench.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import cv2

from scarebox.config import load_config
from scarebox.ingest import make_source
from scarebox.perceive import Perceiver


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * p))
    return sorted_values[idx]


async def _run(cfg, out_dir: Path, peek_frames: int):
    peek_dir = out_dir / "peek"
    crops_dir = out_dir / "crops"
    peek_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    perceiver = Perceiver(cfg)
    source = make_source(cfg)

    detect_track_ms: list[float] = []
    all_events: list[dict] = []
    fire_counts: dict[int, int] = {}
    frame_count = 0
    last_t_capture = 0.0
    first_t_capture = None

    async for frame in source.frames():
        frame_count += 1
        if first_t_capture is None:
            first_t_capture = frame.t_capture
        t0 = time.perf_counter()
        events = perceiver.process_frame(frame.image, frame.t_capture)
        detect_track_ms.append((time.perf_counter() - t0) * 1000.0)
        last_t_capture = frame.t_capture

        if frame_count <= peek_frames:
            annotated = perceiver.debug_draw(frame.image)
            cv2.imwrite(str(peek_dir / f"peek-{frame_count:02d}.jpg"), annotated)

        for ev in events:
            for track_id in ev.track_ids:
                fire_counts[track_id] = fire_counts.get(track_id, 0) + 1
            for i, crop in enumerate(ev.crops):
                cv2.imwrite(str(crops_dir / f"event-{ev.id}-{i}.jpg"), crop)
            all_events.append(
                {
                    "id": ev.id,
                    "t_trigger": ev.t_trigger,
                    "group_size": ev.group_size,
                    "track_ids": ev.track_ids,
                    "n_crops": len(ev.crops),
                }
            )

    # t_capture is a monotonic clock reading, so duration is last minus first, not last alone.
    video_s = (last_t_capture - first_t_capture) if first_t_capture is not None else 0.0
    return frame_count, video_s, detect_track_ms, all_events, fire_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/bench.yaml")
    parser.add_argument("--out", default="runs/perceive")
    parser.add_argument(
        "--evidence",
        default="runs/evidence/perceive-replay.json",
    )
    parser.add_argument("--peek-frames", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_count, video_s, detect_track_ms, all_events, fire_counts = asyncio.run(
        _run(cfg, out_dir, args.peek_frames)
    )

    events_per_min = (len(all_events) / (video_s / 60.0)) if video_s else 0.0
    double_fires = sum(1 for count in fire_counts.values() if count > 1)
    sorted_ms = sorted(detect_track_ms)

    report = {
        "frames": frame_count,
        "video_s": video_s,
        "events": len(all_events),
        "events_per_min": events_per_min,
        "detect_track_p50_ms": _percentile(sorted_ms, 0.50),
        "detect_track_p95_ms": _percentile(sorted_ms, 0.95),
        "double_fires": double_fires,
        "trigger_line": list(cfg.perceive.trigger_line),
        "event_log": all_events,
    }

    evidence_path = Path(args.evidence)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
