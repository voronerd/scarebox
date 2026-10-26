#!/usr/bin/env python3
"""Re-run the judge (terse c/k/f/i/n/d schema, single-crop calls) over all 123 usable silver
crops, and build blind contact sheets from the result. Plan 03-02 Task 2(a).

For each row in `testdata/crops/labels.jsonl` with `category != "unusable"`, makes ONE judge
call against the live VLM server (same `scarebox.prompts.judge_messages`/`JUDGE_JSON_SCHEMA`
the real `scarebox.judge.Judge` uses) and records (file, category, costume, details,
confidence, is_costume, group_size, completion_tokens, judge_ms) as one JSON object per line.
Unlike `scripts/calibrate_judge.py` (which only tracked category/confidence for the threshold
sweep), this keeps every terse-schema field plus latency and token count -- the inputs
`scripts/live_components.py` and `scripts/wrong_verdict_pass.py` both read back.

Then builds blind contact sheets (crop + judge category/costume, NO silver label) at
`<sheets-dir>/sheet_NN.jpg`, 6 crops per sheet, numbered #1..#N in file order, plus
`<sheets-dir>/index.json` mapping `"n" -> file`. Same layout as the 02-02 re-grade sheets
(scratchpad `regrade/sheets.py`) so the hub's independent blind grader can use the same eye.

Usage (inside the app container, vlm already up):
    python3 scripts/judge_v2.py --config config/bench.yaml \
        --labels testdata/crops/labels.jsonl --crops-dir testdata/crops \
        --out runs/evidence/judge-v2-verdicts.jsonl \
        --sheets-dir runs/regrade-v2
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

from scarebox.config import load_config  # noqa: E402
from scarebox.prompts import (  # noqa: E402
    JUDGE_JSON_SCHEMA,
    LEAN_JUDGE_JSON_SCHEMA,
    judge_messages,
    lean_judge_messages,
    validate_lean_verdict,
    validate_verdict,
)


def _b64_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def pctl(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


async def _judge_one(
    client: httpx.AsyncClient, model: str, timeout_s: float, image_url: str, schema: str = "full"
) -> tuple[dict | None, float, int | None, str]:
    """Returns (obj_or_None, elapsed_ms, completion_tokens_or_None, error_reason).

    `schema="lean"` (03-03 Part B) uses the same `judge.schema: lean` wire format
    (`c`/`k`/`d` only) that `scarebox.judge.Judge.judge()` sends when `cfg.judge.schema ==
    "lean"` -- this script has to match it exactly, since the point of the v3 re-grade is
    to score the schema that's actually shipping, not the old six-key one."""
    lean = schema == "lean"
    messages = lean_judge_messages([image_url]) if lean else judge_messages([image_url])
    json_schema = LEAN_JUDGE_JSON_SCHEMA if lean else JUDGE_JSON_SCHEMA
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": json_schema},
        "temperature": 0.0,
    }
    t0 = time.monotonic()
    try:
        resp = await client.post("/chat/completions", json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        elapsed_ms = (time.monotonic() - t0) * 1000
        content = data["choices"][0]["message"]["content"]
        obj = json.loads(content)
        tokens = (data.get("usage") or {}).get("completion_tokens")
    except Exception as exc:  # noqa: BLE001 -- bench script, record and move on
        return None, (time.monotonic() - t0) * 1000, None, str(exc)
    ok, reason = validate_lean_verdict(obj) if lean else validate_verdict(obj)
    if not ok:
        return None, elapsed_ms, tokens, reason
    return obj, elapsed_ms, tokens, ""


async def run(cfg, labels_path: Path, crops_dir: Path, schema: str = "full") -> list[dict]:
    rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    usable = [r for r in rows if r["category"] != "unusable"]
    print(f"{len(rows)} labelled rows, {len(usable)} usable (excluding 'unusable')")

    results: list[dict] = []
    times: list[float] = []
    tokens_list: list[int] = []
    async with httpx.AsyncClient(base_url=cfg.judge.endpoint) as client:
        for i, row in enumerate(usable, 1):
            crop_path = crops_dir / row["file"]
            if not crop_path.exists():
                print(f"  [{i}/{len(usable)}] MISSING crop: {crop_path}", file=sys.stderr)
                continue
            obj, ms, tokens, err = await _judge_one(
                client, cfg.judge.model, cfg.judge.timeout_s, _b64_url(crop_path), schema=schema
            )
            rec: dict = {
                "file": row["file"],
                "silver_category": row["category"],
                "judge_ms": round(ms, 1),
                "completion_tokens": tokens,
            }
            if obj is not None:
                if schema == "lean":
                    # Mirror scarebox.judge.Judge.judge()'s lean-path derivation exactly:
                    # confidence fixed at 1.0, is_costume derived from category, group_size
                    # from the event -- there's no live Event here, so the silver label's own
                    # group_size (same field perceive would have counted) stands in for it.
                    rec.update(
                        category=obj["k"],
                        costume=obj["c"],
                        confidence=1.0,
                        is_costume=obj["k"] != "no_costume",
                        group_size=row.get("group_size", 1),
                        details=obj["d"],
                    )
                else:
                    rec.update(
                        category=obj["k"],
                        costume=obj["c"],
                        confidence=obj["f"],
                        is_costume=obj["i"],
                        group_size=obj["n"],
                        details=obj["d"],
                    )
                times.append(ms)
                if tokens is not None:
                    tokens_list.append(tokens)
            else:
                rec["error"] = err
            results.append(rec)
            tag = rec.get("category", f"FAIL:{err}")
            print(f"  [{i}/{len(usable)}] {row['file']:<30} {ms:.0f}ms tok={tokens} -> {tag}")

    print(f"\n{len(results)} judged ({len(times)} ok, {len(results) - len(times)} failed)")
    if times:
        print(f"judge_ms   p50={pctl(times, 0.5):.0f} p95={pctl(times, 0.95):.0f}")
    if tokens_list:
        print(f"tokens     p50={pctl(tokens_list, 0.5):.0f} p95={pctl(tokens_list, 0.95):.0f}")
    return results


def build_sheets(crops_dir: Path, results: list[dict], sheets_dir: Path) -> int:
    """Blind contact sheets: crop + judge category/costume only, NO silver label. 6 per
    sheet, numbered in file order starting at #1. Writes `<sheets_dir>/index.json`
    (n -> crop file) alongside the jpgs."""
    from PIL import Image, ImageDraw

    sheets_dir.mkdir(parents=True, exist_ok=True)
    W, H, TH = 420, 420, 64
    cols = 3
    index: dict[str, str] = {}
    n_sheets = 0
    for s in range(0, len(results), 6):
        chunk = results[s : s + 6]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * W, rows * (H + TH)), "white")
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk):
            n = s + k + 1
            index[str(n)] = r["file"]
            crop_path = crops_dir / r["file"]
            if crop_path.exists():
                im = Image.open(crop_path).convert("RGB")
                im.thumbnail((W, H))
                x, y = (k % cols) * W, (k // cols) * (H + TH)
                sheet.paste(im, (x, y))
            else:
                x, y = (k % cols) * W, (k // cols) * (H + TH)
            d.text((x + 4, y + H + 4), f"#{n} {r['file'][:40]}", fill="black")
            cat = r.get("category", "FAIL")
            costume = str(r.get("costume", ""))[:38]
            d.text((x + 4, y + H + 22), f"judge: {cat} / {costume}", fill="black")
        n_sheets += 1
        sheet.save(sheets_dir / f"sheet_{n_sheets:02d}.jpg", quality=88)
    (sheets_dir / "index.json").write_text(json.dumps(index, indent=2))
    return n_sheets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/bench.yaml")
    p.add_argument("--labels", default="testdata/crops/labels.jsonl")
    p.add_argument("--crops-dir", default="testdata/crops")
    p.add_argument("--out", required=True, help="output jsonl path")
    p.add_argument("--sheets-dir", default="", help="if set, build blind contact sheets here")
    p.add_argument(
        "--schema",
        default="full",
        choices=("full", "lean"),
        help="judge wire schema to score -- 'lean' matches judge.schema: lean (03-03 Part B v3 re-grade)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    results = asyncio.run(run(cfg, Path(args.labels), Path(args.crops_dir), schema=args.schema))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(results)} verdicts -> {out_path}")

    if args.sheets_dir:
        n_sheets = build_sheets(Path(args.crops_dir), results, Path(args.sheets_dir))
        print(f"wrote {n_sheets} contact sheets -> {args.sheets_dir}/")


if __name__ == "__main__":
    main()
