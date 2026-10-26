#!/usr/bin/env python3
"""Calibrate `judge.min_confidence` against the 01-02 silver crops, through the LIVE VLM
server. Plan 02-02 Task 1 (hub amendment, WWMD W1/W2).

For each of the 123 usable silver-labelled crops (`testdata/crops/labels.jsonl`, excluding
the 12 `category: unusable` rows), makes one judge call and records the raw
(predicted_category, confidence, silver_category). Then, for thresholds 0.0..0.9 (step
0.1), reports:
  - the fraction of crops that would be routed to `mystery` (confidence < threshold)
  - the wrong-category rate among the rest (predicted_category != silver_category)

Picks the LOWEST threshold where wrong-category rate <= 20% AND mystery fraction <= 50%.
If confidence turns out uncorrelated with correctness (wrong-category rate roughly flat
across thresholds), falls back to a per-category rule instead: any category the judge
gets wrong > 50% of the time (at threshold 0.0) is routed straight to `mystery`.

STOP GATE: if the best achievable wrong-category rate on the non-mystery remainder is
above 30%, writes `runs/evidence/DECISION-REQUIRED.md` and exits 1 (WWMD tipping fact)
-- 02-03 should not proceed on a judge this unreliable without a decision.

Usage (inside the app container, on gbox1, with the vLLM server already up):
    python3 scripts/calibrate_judge.py --config config/bench.yaml \\
        --labels testdata/crops/labels.jsonl --crops-dir testdata/crops
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

import httpx

from scarebox.config import load_config
from scarebox.prompts import JUDGE_JSON_SCHEMA, judge_messages, validate_verdict

THRESHOLDS = [round(0.1 * i, 1) for i in range(10)]  # 0.0 .. 0.9
WRONG_RATE_CEILING = 0.20
MYSTERY_FRACTION_CEILING = 0.50
STOP_GATE_WRONG_RATE = 0.30
FLAT_CORRELATION_SPREAD = 0.05  # if wrong-rate range across thresholds is under this, call it flat
PER_CATEGORY_WRONG_CEILING = 0.50

DECISION_DOC = Path("runs/evidence/DECISION-REQUIRED.md")


def _b64_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


async def _judge_raw(client: httpx.AsyncClient, model: str, timeout_s: float, image_url: str) -> dict | None:
    payload = {
        "model": model,
        "messages": judge_messages([image_url]),
        "response_format": {"type": "json_schema", "json_schema": JUDGE_JSON_SCHEMA},
        "temperature": 0.0,
    }
    try:
        resp = await client.post("/chat/completions", json=payload, timeout=timeout_s)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        obj = json.loads(content)
    except Exception as exc:  # noqa: BLE001
        print(f"  judge call failed: {exc}", file=sys.stderr)
        return None
    ok, reason = validate_verdict(obj)
    if not ok:
        print(f"  invalid verdict ({reason}): {content!r}", file=sys.stderr)
        return None
    return obj


async def run(cfg, labels_path: Path, crops_dir: Path) -> list[dict]:
    rows = []
    for line in labels_path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    usable = [r for r in rows if r["category"] != "unusable"]
    print(f"{len(rows)} labelled rows, {len(usable)} usable (excluding 'unusable')")

    results = []
    async with httpx.AsyncClient(base_url=cfg.judge.endpoint) as client:
        for i, row in enumerate(usable, 1):
            crop_path = crops_dir / row["file"]
            if not crop_path.exists():
                print(f"  [{i}/{len(usable)}] MISSING crop: {crop_path}", file=sys.stderr)
                continue
            obj = await _judge_raw(client, cfg.judge.model, cfg.judge.timeout_s, _b64_url(crop_path))
            if obj is None:
                results.append(
                    {"file": row["file"], "silver_category": row["category"], "predicted_category": None, "confidence": None}
                )
                continue
            results.append(
                {
                    "file": row["file"],
                    "silver_category": row["category"],
                    "predicted_category": obj["category"],
                    "confidence": float(obj["confidence"]),
                }
            )
            print(f"  [{i}/{len(usable)}] silver={row['category']:<16} pred={obj['category']:<16} conf={obj['confidence']:.2f}")
    return results


def threshold_table(results: list[dict]) -> list[dict]:
    scored = [r for r in results if r["confidence"] is not None]
    total = len(scored)
    table = []
    for t in THRESHOLDS:
        kept = [r for r in scored if r["confidence"] >= t]
        mystery_n = total - len(kept)
        wrong_n = sum(1 for r in kept if r["predicted_category"] != r["silver_category"])
        wrong_rate = (wrong_n / len(kept)) if kept else 0.0
        mystery_frac = mystery_n / total if total else 0.0
        table.append(
            {
                "threshold": t,
                "mystery_fraction": round(mystery_frac, 4),
                "wrong_category_rate": round(wrong_rate, 4),
                "n_kept": len(kept),
                "n_total": total,
            }
        )
    return table


def per_category_wrong_rate(results: list[dict]) -> dict[str, float]:
    scored = [r for r in results if r["confidence"] is not None]
    by_cat: dict[str, list[bool]] = {}
    for r in scored:
        by_cat.setdefault(r["silver_category"], []).append(
            r["predicted_category"] != r["silver_category"]
        )
    return {cat: sum(vals) / len(vals) for cat, vals in by_cat.items()}


def pick_threshold(table: list[dict]) -> tuple[float | None, bool]:
    """Returns (threshold_or_None, is_flat). `is_flat` means confidence looked uncorrelated
    with correctness, so the caller should use the per-category rule instead."""
    rates = [row["wrong_category_rate"] for row in table]
    is_flat = (max(rates) - min(rates)) < FLAT_CORRELATION_SPREAD
    if is_flat:
        return None, True
    for row in table:
        if row["wrong_category_rate"] <= WRONG_RATE_CEILING and row["mystery_fraction"] <= MYSTERY_FRACTION_CEILING:
            return row["threshold"], False
    return None, False


def write_example_config(threshold: float, config_paths: list[Path]) -> None:
    for path in config_paths:
        if not path.exists():
            continue
        text = path.read_text()
        lines = text.splitlines(keepends=True)
        out = []
        in_judge = False
        for ln in lines:
            if ln.strip() == "judge:":
                in_judge = True
            elif in_judge and ln.strip() and not ln.startswith((" ", "\t")):
                in_judge = False
            if in_judge and ln.lstrip().startswith("min_confidence:"):
                indent = ln[: len(ln) - len(ln.lstrip())]
                comment = ""
                if "#" in ln:
                    comment = "  # " + ln.split("#", 1)[1].strip()
                ln = f"{indent}min_confidence: {threshold}{comment}\n"
            out.append(ln)
        path.write_text("".join(out))
        print(f"wrote judge.min_confidence: {threshold} to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/bench.yaml")
    parser.add_argument("--labels", default="testdata/crops/labels.jsonl")
    parser.add_argument("--crops-dir", default="testdata/crops")
    parser.add_argument(
        "--out", default="runs/calibrate_judge.json", help="raw per-crop results + table"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = asyncio.run(run(cfg, Path(args.labels), Path(args.crops_dir)))
    table = threshold_table(results)

    print("\nthreshold | mystery_fraction | wrong_category_rate | n_kept/n_total")
    for row in table:
        print(
            f"{row['threshold']:.1f}      | {row['mystery_fraction']:.3f}            | "
            f"{row['wrong_category_rate']:.3f}               | {row['n_kept']}/{row['n_total']}"
        )

    chosen, is_flat = pick_threshold(table)
    per_cat = per_category_wrong_rate(results)
    bad_categories = [c for c, rate in per_cat.items() if rate > PER_CATEGORY_WRONG_CEILING]

    best_wrong_rate = min(row["wrong_category_rate"] for row in table)

    summary = {
        "n_usable": len([r for r in results if r["confidence"] is not None]),
        "table": table,
        "per_category_wrong_rate": {k: round(v, 4) for k, v in per_cat.items()},
        "flat_confidence_correlation": is_flat,
        "chosen_threshold": chosen,
        "bad_categories_routed_to_mystery": bad_categories if is_flat else None,
        "best_wrong_category_rate": round(best_wrong_rate, 4),
        "silver_categories_covered": sorted(per_cat.keys()),
    }

    Path("runs").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")

    if is_flat:
        print(
            "\nconfidence is UNCORRELATED with correctness (wrong-category rate flat across "
            "thresholds). Falling back to per-category routing: "
            f"{bad_categories or '(none over 50%)'}"
        )
    elif chosen is not None:
        print(f"\nchosen threshold: {chosen}")
        write_example_config(
            chosen,
            [Path("config/scarebox.example.yaml"), Path("config/bench.yaml")],
        )
    else:
        print("\nNO threshold in 0.0-0.9 clears the 20% wrong-rate / 50% mystery-fraction bars.")

    if best_wrong_rate > STOP_GATE_WRONG_RATE:
        DECISION_DOC.parent.mkdir(parents=True, exist_ok=True)
        DECISION_DOC.write_text(
            "# DECISION REQUIRED: judge calibration failed the 30% wrong-category STOP gate\n\n"
            f"Best achievable wrong-category rate across all thresholds 0.0-0.9 is "
            f"{best_wrong_rate:.1%}, above the 30% ceiling plan 02-02 Task 1 sets before 02-03 "
            "can proceed (WWMD tipping fact).\n\n"
            f"Full table and per-category breakdown: `{args.out}`.\n\n"
            "Silver set covers 14 of 25 library categories; this number may not generalise to "
            "the other 11 the judge has never been measured against.\n"
        )
        print(f"\nSTOP GATE TRIPPED: wrote {DECISION_DOC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
