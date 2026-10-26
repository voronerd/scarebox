#!/usr/bin/env bash
# ladder.sh: runs ONE step of the 03-03 lever ladder on gbox1 and scores it.
#
# Usage (repo root, gbox1, `vlm`+`guard` already up healthy):
#   scripts/ladder.sh <step-name> [dotted.config.path=value ...]
#
# Example (the ladder's own order, spec sect:4.5 + the 03-03 hub amendment):
#   scripts/ladder.sh 01-crop1 perceive.best_crop_frames=1
#   scripts/ladder.sh 02-lean-schema judge.schema=lean perceive.best_crop_frames=1
#   scripts/ladder.sh 03-stream-voice voice.stream_segments=true judge.schema=lean perceive.best_crop_frames=1
#   scripts/ladder.sh 04-single-call quip.single_call=true judge.schema=lean perceive.best_crop_frames=1
#   scripts/ladder.sh 05-crop384 judge.crop_max_px=384 quip.single_call=true judge.schema=lean perceive.best_crop_frames=1
#
# Each override is `dotted.path=value` against `config/bench.yaml` (`section.key=value`,
# value parsed as YAML -- so `true`/`false`/numbers come through typed, not as strings).
# Writes the merged config to `runs/ladder/<step>.yaml`, runs the full stack
# (`vlm`+`guard` assumed already healthy -- this script never touches them) over the
# 5-minute replay via the `app` image with `--mode live` (WavSink: the app image has no
# `alsa-utils`, see app.py's `_make_sink`, so it always falls back to WAV files under
# `runs/wav` regardless of `/dev/snd` being mounted), clears any stale `runs/events.jsonl`
# from a PREVIOUS step first (so each step's report is that step alone, not cumulative),
# then runs `scripts/latency_report.py --json` into `runs/ladder/<step>.json`.
#
# Deny-list compliant: no push/rsync/systemctl/pkill/kill -9/rm -rf outside this repo's own
# runs/ directory (and even there, only `events.jsonl`/`wav/*` from THIS tool's own prior
# runs -- never library/manifest.json).
set -euo pipefail

STEP="${1:?usage: ladder.sh <step-name> [dotted.path=value ...]}"
shift

cd "$(dirname "$0")/.."
mkdir -p runs/ladder

OVERLAY_YAML="runs/ladder/${STEP}.yaml"
REPORT_JSON="runs/ladder/${STEP}.json"

python3 - "config/bench.yaml" "$OVERLAY_YAML" "$@" <<'PYEOF'
import sys
import yaml

base_path, out_path, *overrides = sys.argv[1:]
with open(base_path) as f:
    cfg = yaml.safe_load(f)

for item in overrides:
    dotted, _, raw_value = item.partition("=")
    value = yaml.safe_load(raw_value)  # typed: true/false/int/float/string all parse right
    node = cfg
    parts = dotted.split(".")
    for key in parts[:-1]:
        node = node[key]
    if parts[-1] not in node:
        raise SystemExit(f"ladder.sh: unknown config key {dotted!r}")
    node[parts[-1]] = value

with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"wrote {out_path}")
PYEOF

# Each step's report must be THIS step alone -- clear any events.jsonl left by a previous
# step (never library/manifest.json; never anything outside runs/).
rm -f runs/events.jsonl

echo "== ladder step ${STEP}: $* =="
docker compose -p scarebox -f deploy/compose.yaml --env-file deploy/images.env run --rm app \
    python -m scarebox run --config "$OVERLAY_YAML" --mode live

# Keep this step's raw events (the next step deletes runs/events.jsonl), and score with a python
# that has numpy: the host's bare python3 doesn't (first ladder run lost steps 02-03 this way).
cp runs/events.jsonl "runs/ladder/${STEP}.events.jsonl"
"${PY:-$HOME/.venv/bin/python}" scripts/latency_report.py --json runs/events.jsonl | tee "$REPORT_JSON"
echo "== step ${STEP} report: ${REPORT_JSON} =="
