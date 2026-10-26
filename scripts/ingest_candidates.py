#!/usr/bin/env python3
"""ingest_candidates.py: merge writer outputs (library/candidates/gen/*.json) into the manifest
as `pending` lines, after dedupe and the full live censor (blocklist + Granite Guardian).

Run on gbox1 inside the app image (the guard on :8101 must be up):
  python scripts/ingest_candidates.py --config config/bench.yaml [--dry-run]

Each gen file is `{"source": str, "packs": {pack_key: [lines]}, "categories": {cat: [lines]}}`.
Keeps at most --tier1-cap / --tier2-cap lines per pack and --category-cap new lines per
category, round-robin across sources so every writer is represented. A pack line must name its
costume (a display/alias word appears in it). Writes runs/ingest-report.json with every drop
and its reason; never touches existing manifest lines.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from scarebox.bake import MAX_WORDS, read_manifest, write_manifest_atomic
from scarebox.censor import Censor, Reject
from scarebox.config import load_config
from scarebox.library import normalise_costume

MIN_WORDS = 4


_TYPOGRAPHY = {"\u2014": ", ", "\u2013": ", ", "\u2026": "...", "\u2018": "'", "\u2019": "'",
               "\u201c": "", "\u201d": "", "\u00e9": "e", "\u00e8": "e", "\u00e0": "a"}


def clean(text: str) -> str:
    """Straighten writer typography the censor rejects as non-printable (em dashes, curly
    quotes, accented letters). Meaning is unchanged; ', ' gives the TTS the same pause a dash
    would."""
    text = str(text)
    for k, v in _TYPOGRAPHY.items():
        text = text.replace(k, v)
    text = " ".join(text.split()).strip().strip('"')
    return re.sub(r"\s+([,!?.])", r"\1", text).replace(",,", ",")


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def is_dup(text: str, seen: list[set[str]], threshold: float = 0.7) -> bool:
    t = tokens(text)
    for s in seen:
        if t and s and len(t & s) / len(t | s) >= threshold:
            return True
    return False


def names_costume(text: str, pack: dict) -> bool:
    words = set(normalise_costume(text).split())
    for alias in [pack.get("display", ""), *pack.get("aliases", [])]:
        alias_words = [w for w in normalise_costume(alias).split() if len(w) >= 3]
        if alias_words and all(w in words for w in alias_words):
            return True
    # single distinctive word from the display name ("Elsa", "banana") is enough
    return any(w in words for w in normalise_costume(pack.get("display", "")).split() if len(w) >= 4)


def round_robin(by_source: dict[str, list[str]]) -> list[tuple[str, str]]:
    queues = {s: list(lines) for s, lines in by_source.items()}
    for q in queues.values():
        random.shuffle(q)
    out = []
    while any(queues.values()):
        for s in sorted(queues):
            if queues[s]:
                out.append((s, queues[s].pop(0)))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/bench.yaml")
    ap.add_argument("--gen-dir", default="library/candidates/gen")
    ap.add_argument("--packs", default="library/costume_packs.json")
    ap.add_argument("--tier1-cap", type=int, default=10)
    ap.add_argument("--tier2-cap", type=int, default=6)
    ap.add_argument("--category-cap", type=int, default=30)
    ap.add_argument("--guard-timeout", type=float, default=5.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    random.seed(20261016)

    cfg = load_config(args.config)
    manifest_path = Path(cfg.library.manifest)
    manifest = read_manifest(manifest_path)
    packs = {p["key"]: p for p in json.loads(Path(args.packs).read_text())}

    # target -> source -> lines. target is ("pack", key) or ("cat", category)
    pool: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(Path(args.gen_dir).glob("*.json")):
        data = json.loads(f.read_text())
        src = data.get("source", f.stem)
        for key, lines in data.get("packs", {}).items():
            pool[("pack", key)][src].extend(lines)
        for cat, lines in data.get("categories", {}).items():
            pool[("cat", cat)][src].extend(lines)

    censor = Censor(
        cfg.censor.blocklist,
        cfg.censor.guard_endpoint,
        cfg.censor.guard_model,
        guard_timeout_s=args.guard_timeout,
    )
    sem = asyncio.Semaphore(args.concurrency)

    async def check(text: str, category: str):
        async with sem:
            res = await censor.check(text, category)
            if isinstance(res, Reject) and "timeout" in res.reason:
                res = await censor.check(text, category)  # one retry: offline, no deadline
            return res

    seen = [tokens(line["text"]) for line in manifest]
    existing_ids = {line["id"] for line in manifest}
    report = {"kept": 0, "dropped": [], "per_target": {}}
    new_lines: list[dict] = []

    for (kind, key), by_source in sorted(pool.items()):
        if kind == "pack":
            pack = packs.get(key)
            if pack is None:
                report["dropped"].append({"target": key, "reason": "unknown pack"})
                continue
            category, cap = pack["category"], (args.tier1_cap if pack["tier"] == 1 else args.tier2_cap)
        else:
            category, cap = key, args.category_cap
        candidates = []
        for src, text in round_robin(by_source):
            text = clean(text)
            n = len(text.split())
            reason = None
            if not MIN_WORDS <= n <= MAX_WORDS:
                reason = f"{n} words"
            elif kind == "pack" and not names_costume(text, pack):
                reason = "doesn't name the costume"
            elif is_dup(text, seen):
                reason = "near-duplicate"
            if reason:
                report["dropped"].append({"target": key, "source": src, "text": text, "reason": reason})
                continue
            seen.append(tokens(text))
            candidates.append((src, text))
        results = await asyncio.gather(*(check(t, category) for _, t in candidates))
        kept = 0
        for (src, text), res in zip(candidates, results):
            if isinstance(res, Reject):
                report["dropped"].append({"target": key, "source": src, "text": text, "reason": f"censor:{res.reason}"})
                continue
            if kept >= cap:
                report["dropped"].append({"target": key, "source": src, "text": text, "reason": "over cap"})
                continue
            kept += 1
            prefix = f"pack-{key}" if kind == "pack" else f"{category}-x"
            i = 1
            while f"{prefix}-{i:02d}" in existing_ids:
                i += 1
            line_id = f"{prefix}-{i:02d}"
            existing_ids.add(line_id)
            entry = {"id": line_id, "category": category, "text": text, "wav": None,
                     "status": "pending", "reviewed_at": None, "source": src}
            if kind == "pack":
                entry["pack"] = key
            new_lines.append(entry)
        report["per_target"][f"{kind}:{key}"] = kept
        print(f"{kind}:{key} kept {kept}/{len(candidates)}", flush=True)

    await censor.aclose()
    report["kept"] = len(new_lines)
    reasons = defaultdict(int)
    for d in report["dropped"]:
        reasons[d["reason"].split(":")[0] if d["reason"].startswith("censor") else d["reason"]] += 1
    report["drop_reasons"] = dict(reasons)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/ingest-report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"kept": len(new_lines), "drop_reasons": report["drop_reasons"]}))
    if not args.dry_run:
        write_manifest_atomic(manifest_path, manifest + new_lines)


if __name__ == "__main__":
    asyncio.run(main())
