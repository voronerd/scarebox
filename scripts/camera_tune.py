#!/usr/bin/env python3
"""Tune the camera's sub-stream (ExtraFormat[0]) for low latency.

Runs on a workstation on the camera's network, against CAMERA_HOST/USER/PASSWORD
from .env. Touches only frame rate, GOP, codec and bit-rate-control keys on the
sub-stream. Never touches resolution (Width/Height/resolution) or the main stream
(MainFormat) -- those feed Martin's home NVR at 640x480 and must not move.

Usage:
    scripts/camera_tune.py                       # tune (writes the orig snapshot first)
    scripts/camera_tune.py --dry-run              # show the planned changes, apply nothing
    scripts/camera_tune.py --rollback             # restore from the orig snapshot
    scripts/camera_tune.py --rollback --dry-run   # show the restore plan only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scarebox.camera import Camera  # noqa: E402

EVIDENCE_DIR = REPO_ROOT / "runs" / "evidence"
ORIG_FILE = EVIDENCE_DIR / "camera-encode-orig.txt"

# Sub-stream (ExtraFormat[0]) targets. Keys are the bare (no "table." prefix)
# form setConfig expects; getConfig returns the "table."-prefixed form.
TARGETS = {
    "Encode[0].ExtraFormat[0].Video.Compression": "H.264",
    "Encode[0].ExtraFormat[0].Video.FPS": "15",
    "Encode[0].ExtraFormat[0].Video.GOP": "15",
    "Encode[0].ExtraFormat[0].Video.BitRateControl": "CBR",
}
ENABLE_KEY = "Encode[0].ExtraFormat[0].VideoEnable"

# Never let this script touch resolution or the main stream, even in rollback.
PROTECTED_SUBSTRINGS = ("width", "height", "resolution", "mainformat")


def _is_protected(bare_key: str) -> bool:
    lower = bare_key.lower()
    return any(s in lower for s in PROTECTED_SUBSTRINGS)


assert not any(_is_protected(k) for k in TARGETS), "tune targets must exclude resolution/MainFormat"
assert not _is_protected(ENABLE_KEY), "enable key must exclude resolution/MainFormat"


def _load_env() -> dict[str, str]:
    env_path = REPO_ROOT / ".env"
    env: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _camera() -> Camera:
    env = _load_env()
    return Camera(env["CAMERA_HOST"], env["CAMERA_USER"], env["CAMERA_PASSWORD"], timeout=5.0)


def _strip_table_prefix(key: str) -> str:
    return key[len("table."):] if key.startswith("table.") else key


def _subformat_keys(cfg: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in cfg.items() if "ExtraFormat[0]" in k}


def _safe_set(cam: Camera, bare_key: str, value: str) -> None:
    if _is_protected(bare_key):
        raise RuntimeError(f"refusing to set protected key: {bare_key}")
    cam.set_config_key(bare_key, value)


def write_orig_snapshot(cfg: dict[str, str]) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    lines = sorted(f"{k}={v}" for k, v in _subformat_keys(cfg).items())
    ORIG_FILE.write_text("\n".join(lines) + "\n")


def read_orig_snapshot() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in ORIG_FILE.read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key] = value
    return result


def tune(dry_run: bool) -> int:
    cam = _camera()
    cfg = cam.get_config("Encode")
    sub = _subformat_keys(cfg)
    if not sub:
        print("ERROR: getConfig&name=Encode returned no ExtraFormat[0] keys", file=sys.stderr)
        return 1

    write_orig_snapshot(cfg)
    print(f"wrote {ORIG_FILE.relative_to(REPO_ROOT)} ({len(sub)} keys)")

    plan: list[tuple[str, str, str]] = []
    for bare_key, target_value in TARGETS.items():
        table_key = f"table.{bare_key}"
        if table_key not in sub:
            print(f"SKIP (key not present): {bare_key}")
            continue
        current = sub[table_key]
        if current == target_value:
            print(f"SKIP (already {target_value}): {bare_key}")
            continue
        plan.append((bare_key, current, target_value))

    enable_table_key = f"table.{ENABLE_KEY}"
    if enable_table_key in sub and sub[enable_table_key].strip().lower() != "true":
        plan.append((ENABLE_KEY, sub[enable_table_key], "true"))

    if dry_run:
        for bare_key, current, target in plan:
            print(f"DRY-RUN would set {bare_key}: {current} -> {target}")
        return 0

    for bare_key, current, target in plan:
        print(f"setting {bare_key}: {current} -> {target}")
        _safe_set(cam, bare_key, target)

    new_cfg = cam.get_config("Encode")
    new_sub = _subformat_keys(new_cfg)
    failures = []
    for bare_key, target_value in TARGETS.items():
        table_key = f"table.{bare_key}"
        if table_key in new_sub and new_sub[table_key] != target_value:
            failures.append((bare_key, new_sub[table_key], target_value))
    enable_now = new_sub.get(enable_table_key, "true")
    if enable_now.strip().lower() != "true":
        failures.append((ENABLE_KEY, enable_now, "true"))

    if failures:
        for bare_key, got, want in failures:
            print(f"ASSERT FAILED: {bare_key} is {got!r}, wanted {want!r}", file=sys.stderr)
        return 1

    print("tune OK: re-read confirms new sub-stream values")
    return 0


def rollback(dry_run: bool) -> int:
    if not ORIG_FILE.exists():
        print(f"ERROR: {ORIG_FILE} does not exist; nothing to roll back to", file=sys.stderr)
        return 1

    orig = read_orig_snapshot()
    cam = None if dry_run else _camera()
    current_cfg = None if dry_run else cam.get_config("Encode")

    restored = 0
    for table_key, orig_value in sorted(orig.items()):
        bare_key = _strip_table_prefix(table_key)
        if _is_protected(bare_key):
            continue
        if dry_run:
            print(f"DRY-RUN would restore {bare_key} -> {orig_value}")
            continue
        current_value = current_cfg.get(table_key)
        if current_value == orig_value:
            continue
        print(f"restoring {bare_key}: {current_value} -> {orig_value}")
        _safe_set(cam, bare_key, orig_value)
        restored += 1

    if not dry_run:
        print(f"rollback OK: restored {restored} key(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback", action="store_true", help="restore from the orig snapshot")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, apply nothing")
    args = parser.parse_args()

    if args.rollback:
        return rollback(args.dry_run)
    return tune(args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
