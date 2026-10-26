#!/usr/bin/env python3
"""Feed KNOWN-WRONG judge verdicts, plus a random sample of v2 verdicts, through quip +
censor, and record exactly what survives and why. Plan 03-02 Task 2(b) (hub resume,
2026-10-15).

`quip`/`censor` police LENGTH, costume-NOUN-presence and CONTENT SAFETY -- they do not, and
cannot, police whether the judge's costume call was actually correct. This script measures
that gap honestly: feed it a verdict you already know is wrong (the 8 crops an independent
blind grader marked "bad" in 02-02's image-level re-grade) and count how many still produce
a spoken line. Expect most of them to survive -- that is the expected, documented behaviour,
not a bug this script is trying to catch.

Inputs:
  - The 8 "bad" rows from `judge-image-regrade.json` (`grade == "bad"`), whose `n` is
    mapped POSITIONALLY to `judge-linefit-regrade.json`'s `pairs[n-1]` (per the hub's
    resume instructions) to recover the crop filename and the (wrong) judge category/costume
    that regrade pass captured.
  - 20 verdicts sampled at random from `judge_v2.py`'s output (`judge-v2-verdicts.jsonl`),
    with no filtering by correctness -- this is the "ordinary" baseline the 8 known-bad ones
    are compared against.

For each input, runs quip's own request/filter pipeline (same schema, same few-shots, same
`Censor.check()` the production `Quip.quip()` uses) but keeps the per-candidate diagnostics
`Quip.quip()` itself discards, so the output records WHY each candidate was kept or dropped,
not just the final line.

Usage (inside the app container, vlm+guard up):
    python3 scripts/wrong_verdict_pass.py --config config/bench.yaml \
        --image-regrade runs/evidence/judge-image-regrade.json \
        --linefit-regrade runs/evidence/judge-linefit-regrade.json \
        --judge-v2 runs/evidence/judge-v2-verdicts.jsonl \
        --out runs/evidence/wrong-verdict-pass.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scarebox.censor import Censor, Pass  # noqa: E402
from scarebox.config import load_config  # noqa: E402
from scarebox.prompts import QUIP_JSON_SCHEMA, costume_noun_in_line, line_word_count, quip_messages, validate_quip  # noqa: E402
from scarebox.quip import Quip, _few_shot_lines  # noqa: E402

_SEED = 20261015


async def _quip_with_diagnostics(
    quip: Quip, costume: str, category: str, details: str, is_costume: bool
) -> dict:
    """Mirrors `Quip.quip()`'s request + filter pipeline exactly, but returns per-candidate
    diagnostics instead of just the winner -- `Quip.quip()` itself has no reason to keep
    that detail once it has picked (or failed to pick) a survivor."""
    verdict_dict = {"costume": costume, "category": category, "details": details, "is_costume": is_costume}
    few_shots = _few_shot_lines(quip._manifest, category)
    payload = {
        "model": quip.cfg.model,
        "messages": quip_messages(verdict_dict, few_shots=few_shots),
        "response_format": {"type": "json_schema", "json_schema": QUIP_JSON_SCHEMA},
        "temperature": 0.9,
        "max_tokens": 50,
    }
    try:
        resp = await quip._client.post("/chat/completions", json=payload, timeout=quip.cfg.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        obj = json.loads(content)
    except Exception as exc:  # noqa: BLE001 -- bench script, record and move on
        return {"candidates": [], "chosen": None, "reason": f"quip_call_failed:{exc}"}

    ok, reason = validate_quip(obj)
    if not ok:
        return {"candidates": [], "chosen": None, "reason": f"invalid_quip:{reason}"}

    noun_target = costume if costume and costume.strip().lower() != "none" else category
    per_candidate = []
    survivors = []
    for line in obj["lines"]:
        if is_costume and not costume_noun_in_line(noun_target, line):
            per_candidate.append({"line": line, "rejected": "costume_noun_missing"})
            continue
        if line_word_count(line) > quip.cfg.max_words:
            per_candidate.append({"line": line, "rejected": "too_long"})
            continue
        result = await quip._censor.check(line, category)
        if isinstance(result, Pass):
            per_candidate.append({"line": line, "rejected": None})
            survivors.append(line)
        else:
            per_candidate.append({"line": line, "rejected": f"censor:{result.reason}"})

    chosen = min(survivors, key=line_word_count) if survivors else None
    top_reason = "survived: shortest clean candidate" if chosen else "all candidates rejected"
    return {"candidates": per_candidate, "chosen": chosen, "reason": top_reason}


def _bad_inputs(image_regrade_path: Path, linefit_regrade_path: Path) -> list[dict]:
    """The 8 blind-graded 'bad' crops, mapped `n -> judge-linefit-regrade.json pairs[n-1]`
    (positional, per the hub's resume instructions) to recover the crop file + the judge's
    (wrong) category/costume call."""
    image_regrade = json.loads(image_regrade_path.read_text())
    bad_ns = [p["n"] for p in image_regrade["pairs"] if p["grade"] == "bad"]
    linefit = json.loads(linefit_regrade_path.read_text())
    linefit_pairs = linefit["pairs"]

    inputs = []
    for n in bad_ns:
        row = linefit_pairs[n - 1]
        category = row["judge_category"]
        inputs.append(
            {
                "source": "bad_verdict",
                "n": n,
                "file": row["crop"],
                "silver_category": row.get("silver_category"),
                "judge_category": category,
                "judge_costume": row.get("judge_costume", ""),
                "costume": row.get("judge_costume", ""),
                "category": category,
                "details": "",
                "is_costume": category != "no_costume",
            }
        )
    return inputs


def _v2_sample(judge_v2_path: Path, n: int) -> list[dict]:
    rows = [json.loads(line) for line in judge_v2_path.read_text().splitlines() if line.strip()]
    ok = [r for r in rows if "category" in r]
    rng = random.Random(_SEED)
    sample = rng.sample(ok, min(n, len(ok)))
    return [
        {
            "source": "v2_sample",
            "n": None,
            "file": r["file"],
            "silver_category": r.get("silver_category"),
            "judge_category": r["category"],
            "judge_costume": r.get("costume", ""),
            "costume": r.get("costume", ""),
            "category": r["category"],
            "details": r.get("details", ""),
            "is_costume": bool(r.get("is_costume", True)),
        }
        for r in sample
    ]


async def amain(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    censor = Censor(
        cfg.censor.blocklist,
        cfg.censor.guard_endpoint,
        cfg.censor.guard_model,
        guard_timeout_s=cfg.censor.guard_timeout_s,
    )
    quip = Quip(cfg, censor)

    bad_inputs = _bad_inputs(Path(args.image_regrade), Path(args.linefit_regrade))
    v2_inputs = _v2_sample(Path(args.judge_v2), 20)
    print(f"{len(bad_inputs)} known-bad verdicts, {len(v2_inputs)} random v2-sample verdicts")

    records = []
    try:
        for inp in bad_inputs + v2_inputs:
            diag = await _quip_with_diagnostics(
                quip, inp["costume"], inp["category"], inp["details"], inp["is_costume"]
            )
            records.append({**inp, **diag})
            print(
                f"  [{inp['source']} n={inp['n']}] {inp['file']:<30} "
                f"judge={inp['judge_category']}/{inp['judge_costume'][:24]!r} "
                f"-> {diag['chosen']!r}"
            )
    finally:
        await quip.aclose()
        await censor.aclose()

    bad_records = [r for r in records if r["source"] == "bad_verdict"]
    v2_records = [r for r in records if r["source"] == "v2_sample"]
    wrong_costume_survivors = sum(1 for r in bad_records if r["chosen"] is not None)

    summary = {
        "n_bad_verdicts": len(bad_records),
        "n_bad_survived": wrong_costume_survivors,
        "n_v2_sample": len(v2_records),
        "n_v2_survived": sum(1 for r in v2_records if r["chosen"] is not None),
        "note": (
            "quip/censor police length, costume-noun-presence and content safety -- NOT "
            "costume accuracy. A verdict already known to have the wrong costume surviving "
            "to a spoken line is EXPECTED, not a defect found by this script."
        ),
        "records": records,
    }
    print(
        f"\nwrong-costume lines that survived censor: {wrong_costume_survivors}/{len(bad_records)}"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/bench.yaml")
    p.add_argument("--image-regrade", required=True)
    p.add_argument("--linefit-regrade", required=True)
    p.add_argument("--judge-v2", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = asyncio.run(amain(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
