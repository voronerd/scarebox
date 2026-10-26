#!/usr/bin/env python3
"""Measure the live-path components' latency with the VLM (and guard) actually busy, like
race day. Plan 03-02 Task 2(a).

Reads `scripts/judge_v2.py`'s output (`judge-v2-verdicts.jsonl`) for the judge-v2 timing
numbers (no second judge pass -- those calls already happened), then measures, against the
SAME live servers:
  - `quip`: `scarebox.quip.Quip.quip()` (full path: VLM call + censor, same as production)
    over 20 verdicts sampled from the judge-v2 output.
  - `judge_and_quip`: the single-call fallback over 20 crops (same sample's files).
  - `voice_render`: `scarebox.voice.Voice.render_to_array()` over the quip lines that
    survived (padded from approved library lines if fewer than 20 survived, so a low quip
    survival rate doesn't starve the voice sample) -- run while the VLM/guard containers are
    still up and warm, matching the shared-GPU race-day condition.

Writes one JSON file with p50/p95 ms (and tokens, for judge/quip) per component.

Usage (inside the app container, vlm+guard up):
    python3 scripts/live_components.py --config config/bench.yaml \
        --judge-v2 runs/evidence/judge-v2-verdicts.jsonl \
        --crops-dir testdata/crops \
        --out runs/evidence/live-components.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from scarebox.censor import Censor  # noqa: E402
from scarebox.config import load_config  # noqa: E402
from scarebox.events import Event, Verdict  # noqa: E402
from scarebox.quip import Quip  # noqa: E402
from scarebox.voice import Voice  # noqa: E402

_SEED = 20261015  # fixed so re-runs sample the same 20 verdicts


def pctl(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _load_bgr(path: Path) -> np.ndarray:
    """PIL loads RGB; `judge._encode_data_url` / `Event.crops` expect BGR ndarrays (same
    convention the real camera/perceive pipeline produces via OpenCV)."""
    rgb = np.array(Image.open(path).convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _verdict_from_row(row: dict) -> Verdict:
    return Verdict(
        costume=row.get("costume", ""),
        category=row.get("category", "mystery"),
        confidence=float(row.get("confidence") or 0.0),
        is_costume=bool(row.get("is_costume", True)),
        group_size=int(row.get("group_size") or 1),
        details=row.get("details", ""),
    )


def judge_v2_stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if "category" in r]
    times = [r["judge_ms"] for r in ok]
    tokens = [r["completion_tokens"] for r in ok if r.get("completion_tokens") is not None]
    return {
        "n": len(ok),
        "n_failed": len(rows) - len(ok),
        "p50_ms": pctl(times, 0.5),
        "p95_ms": pctl(times, 0.95),
        "p50_tokens": pctl(tokens, 0.5),
        "p95_tokens": pctl(tokens, 0.95),
    }


async def measure_quip(cfg, censor: Censor, sample: list[dict]) -> tuple[dict, list[str]]:
    quip = Quip(cfg, censor)
    times: list[float] = []
    lines: list[str] = []
    try:
        for row in sample:
            verdict = _verdict_from_row(row)
            t0 = time.monotonic()
            line = await quip.quip(verdict, verdict.category)
            times.append((time.monotonic() - t0) * 1000)
            if line:
                lines.append(line)
    finally:
        await quip.aclose()
    stats = {
        "n": len(times),
        "n_survived": len(lines),
        "p50_ms": pctl(times, 0.5),
        "p95_ms": pctl(times, 0.95),
    }
    return stats, lines


async def measure_judge_and_quip(cfg, censor: Censor, crops_dir: Path, files: list[str]) -> dict:
    quip = Quip(cfg, censor)
    times: list[float] = []
    n_survived = 0
    try:
        for fname in files:
            crop_path = crops_dir / fname
            if not crop_path.exists():
                continue
            event = Event(crops=[_load_bgr(crop_path)])
            t0 = time.monotonic()
            _verdict, line = await quip.judge_and_quip(event)
            times.append((time.monotonic() - t0) * 1000)
            if line:
                n_survived += 1
    finally:
        await quip.aclose()
    return {
        "n": len(times),
        "n_survived": n_survived,
        "p50_ms": pctl(times, 0.5),
        "p95_ms": pctl(times, 0.95),
    }


def _pad_lines(lines: list[str], manifest_path: str, n_needed: int) -> list[str]:
    """Pads out to `n_needed` lines using approved library lines (never hardcoded filler --
    spec section 6 applies here too) if quip didn't produce enough survivors on its own."""
    if len(lines) >= n_needed:
        return lines[:n_needed]
    try:
        manifest = json.loads(Path(manifest_path).read_text())
        pool = [m["text"] for m in manifest if m.get("status") == "approved"]
    except Exception:  # noqa: BLE001 -- padding is best-effort
        pool = []
    extra = list(lines)
    i = 0
    while len(extra) < n_needed and pool:
        extra.append(pool[i % len(pool)])
        i += 1
    return extra[:n_needed] if extra else lines


def measure_voice(cfg, lines: list[str]) -> dict:
    voice = Voice(cfg)  # warms up on construction, same as production
    times: list[float] = []
    for line in lines:
        t0 = time.monotonic()
        voice.render_to_array(line)
        times.append((time.monotonic() - t0) * 1000)
    return {"n": len(times), "p50_ms": pctl(times, 0.5), "p95_ms": pctl(times, 0.95)}


async def amain(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    censor = Censor(
        cfg.censor.blocklist,
        cfg.censor.guard_endpoint,
        cfg.censor.guard_model,
        guard_timeout_s=cfg.censor.guard_timeout_s,
    )

    jv2_path = Path(args.judge_v2)
    jv2_rows = [json.loads(line) for line in jv2_path.read_text().splitlines() if line.strip()]
    jv2_ok = [r for r in jv2_rows if "category" in r]
    rng = random.Random(_SEED)
    sample = rng.sample(jv2_ok, min(20, len(jv2_ok)))
    print(f"sampled {len(sample)}/{len(jv2_ok)} judge-v2 verdicts for quip/judge_and_quip")

    quip_stats, quip_lines = await measure_quip(cfg, censor, sample)
    print(f"quip: n={quip_stats['n']} survived={quip_stats['n_survived']} "
          f"p50={quip_stats['p50_ms']} p95={quip_stats['p95_ms']}")

    files = [r["file"] for r in sample]
    jq_stats = await measure_judge_and_quip(cfg, censor, Path(args.crops_dir), files)
    print(f"judge_and_quip: n={jq_stats['n']} survived={jq_stats['n_survived']} "
          f"p50={jq_stats['p50_ms']} p95={jq_stats['p95_ms']}")

    voice_lines = _pad_lines(quip_lines, cfg.library.manifest, 20)
    voice_stats = measure_voice(cfg, voice_lines)
    print(f"voice_render: n={voice_stats['n']} p50={voice_stats['p50_ms']} p95={voice_stats['p95_ms']}")

    await censor.aclose()

    return {
        "judge_v2": judge_v2_stats(jv2_rows),
        "quip": quip_stats,
        "judge_and_quip": jq_stats,
        "voice_render": voice_stats,
        "sampled_from": str(jv2_path),
        "seed": _SEED,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/bench.yaml")
    p.add_argument("--judge-v2", required=True, help="judge_v2.py output jsonl")
    p.add_argument("--crops-dir", default="testdata/crops")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = asyncio.run(amain(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
