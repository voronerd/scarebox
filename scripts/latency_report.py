#!/usr/bin/env python3
"""latency_report.py: summarise a `runs/events.jsonl` replay run (spec §4.4/§4.5, 03-03 plan
Task 2). Usage:

    scripts/latency_report.py [runs/events.jsonl]

Prints, as plain text:
  - per-stage and total latency (p50/p95/max, ms) from each event's `ms` dict
  - path mix (live / library / dropped) and live share
  - late count (from the orchestrator's separate `{"id", "late": true}` log lines)
  - category mix
  - censor reject split by reason (deterministic / guard-risk / guard-timeout / parse),
    read from `reject_reason` -- counts `quip-none` separately since it's not a CENSOR
    rejection and shouldn't be folded into the ADR 0001 ">30% rejected" trigger

No third-party deps beyond `numpy` (already a project dependency, used for percentiles).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_CENSOR_BUCKETS = ("deterministic", "guard-risk", "guard-timeout", "parse")


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    arr = np.array(values, dtype=float)
    return {
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "max": round(float(arr.max()), 1),
        "n": len(values),
    }


def load_records(path: Path) -> tuple[list[dict], int]:
    """Returns (event_records, late_count). A `{"late": true}` line (no `path` key) is
    counted separately, not folded into the event records list."""
    records: list[dict] = []
    late = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("late") is True:
            late += 1
            continue
        if "path" in obj:
            records.append(obj)
    return records, late


def report(records: list[dict], late: int) -> str:
    out: list[str] = []
    n = len(records)
    out.append(f"events: {n}  late: {late}")

    # --- path mix / live share ---
    path_counts: dict[str, int] = {}
    for r in records:
        path_counts[r.get("path", "unknown")] = path_counts.get(r.get("path", "unknown"), 0) + 1
    out.append("")
    out.append("path mix:")
    for path in ("live", "library", "dropped"):
        count = path_counts.get(path, 0)
        pct = (count / n * 100) if n else 0.0
        out.append(f"  {path:8s} {count:5d}  ({pct:5.1f}%)")
    other = set(path_counts) - {"live", "library", "dropped"}
    for path in sorted(other):
        out.append(f"  {path:8s} {path_counts[path]:5d}")
    live_share = (path_counts.get("live", 0) / n) if n else 0.0
    out.append(f"  live share: {live_share * 100:.1f}%")

    # --- category mix ---
    cat_counts: dict[str, int] = {}
    for r in records:
        cat = r.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    out.append("")
    out.append("category mix:")
    for cat, count in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        out.append(f"  {cat:16s} {count:5d}")

    # --- per-stage / total latency ---
    stage_values: dict[str, list[float]] = {}
    for r in records:
        for stage, ms in (r.get("ms") or {}).items():
            stage_values.setdefault(stage, []).append(float(ms))
    out.append("")
    out.append("latency (ms):")
    out.append(f"  {'stage':20s} {'n':>5s} {'p50':>8s} {'p95':>8s} {'max':>8s}")
    for stage in sorted(stage_values, key=lambda s: (s != "total", s)):
        stats = _percentiles(stage_values[stage])
        out.append(
            f"  {stage:20s} {stats['n']:5d} {stats['p50']:8.1f} {stats['p95']:8.1f} {stats['max']:8.1f}"
        )

    # --- censor reject split ---
    reject_counts: dict[str, int] = {}
    for r in records:
        reason = r.get("reject_reason")
        if reason:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
    censor_total = sum(reject_counts.get(b, 0) for b in _CENSOR_BUCKETS)
    out.append("")
    out.append("censor reject split (ADR 0001's '>30% rejected' trigger reads from this):")
    denom = censor_total if censor_total else 1
    for bucket in _CENSOR_BUCKETS:
        count = reject_counts.get(bucket, 0)
        pct = count / denom * 100
        out.append(f"  {bucket:16s} {count:5d}  ({pct:5.1f}% of censor rejects)")
    quip_none = reject_counts.get("quip-none", 0)
    out.append(f"  {'quip-none':16s} {quip_none:5d}  (not a censor reject -- filter/call failure)")
    out.append(f"  censor rejects total: {censor_total}")

    return "\n".join(out)


def summary(records: list[dict], late: int) -> dict:
    """Machine-readable form of the same numbers `report()` prints -- used by `--json`
    (`scripts/ladder.sh` folds this straight into a `runs/ladder/<step>.json` lever-ladder
    step file, and the live-replay summary's `chosen` block follows the same shape)."""
    n = len(records)
    path_mix: dict[str, int] = {}
    cat_mix: dict[str, int] = {}
    stage_values: dict[str, list[float]] = {}
    reject_counts: dict[str, int] = {}
    for r in records:
        path_mix[r.get("path", "unknown")] = path_mix.get(r.get("path", "unknown"), 0) + 1
        cat_mix[r.get("category", "unknown")] = cat_mix.get(r.get("category", "unknown"), 0) + 1
        for stage, ms in (r.get("ms") or {}).items():
            stage_values.setdefault(stage, []).append(float(ms))
        reason = r.get("reject_reason")
        if reason:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
    censor_total = sum(reject_counts.get(b, 0) for b in _CENSOR_BUCKETS)
    total_stats = _percentiles(stage_values.get("total", []))
    return {
        "n_events": n,
        "late": late,
        "path_mix": path_mix,
        "live_share": round((path_mix.get("live", 0) / n), 4) if n else 0.0,
        "category_mix": cat_mix,
        "ms": {stage: _percentiles(values) for stage, values in stage_values.items()},
        "total_p50_ms": total_stats["p50"],
        "total_p95_ms": total_stats["p95"],
        "total_max_ms": total_stats["max"],
        "censor_reject_split": {b: reject_counts.get(b, 0) for b in _CENSOR_BUCKETS},
        "censor_rejects_total": censor_total,
        "quip_none": reject_counts.get("quip-none", 0),
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv[1:]
    path = Path(args[0]) if args else Path("runs/events.jsonl")
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        raise SystemExit(1)
    records, late = load_records(path)
    if as_json:
        print(json.dumps(summary(records, late), indent=2))
    else:
        print(report(records, late))


if __name__ == "__main__":
    main()
