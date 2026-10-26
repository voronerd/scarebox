#!/usr/bin/env bash
# One VLM bake-off cycle on the GPU host: up -> wait ready -> bench -> down (always, via trap).
# Usage (repo root, on the GPU host):
#   scripts/bench_cycle.sh <model-dir-name> <served-name> [sizes=512,384] [extra vllm args...]
# Writes runs/vlm/<served-name>.jsonl and runs/vlm/<served-name>.cycle.log (load time, memory).
set -uo pipefail
MODEL_DIR="$HOME/models/$1"; MODEL_NAME="$2"; SIZES="${3:-512,384}"; shift 3 2>/dev/null || shift $#
export MODEL_DIR MODEL_NAME VLLM_EXTRA_ARGS="${*:-}"
COMPOSE=(docker compose -p scarebox-bench -f deploy/bench/compose.vlm.yaml --env-file deploy/images.env)
LOG="runs/vlm/${MODEL_NAME}.cycle.log"
mkdir -p runs/vlm
trap '"${COMPOSE[@]}" down >>"$LOG" 2>&1; echo "down $(date -Is)" >>"$LOG"' EXIT

avail() { free -g | awk 'NR==2{print $7}'; }
echo "== $MODEL_NAME start $(date -Is) avail_before=$(avail)GiB" >"$LOG"
[ -f "$MODEL_DIR/config.json" ] || { echo "missing weights $MODEL_DIR" >>"$LOG"; exit 1; }
[ "$(avail)" -ge 40 ] || { echo "not enough memory" >>"$LOG"; exit 1; }
"${COMPOSE[@]}" up -d >>"$LOG" 2>&1
t0=$(date +%s)
for _ in $(seq 1 90); do
    curl -sf http://127.0.0.1:8100/v1/models >/dev/null 2>&1 && break
    docker ps --format '{{.Names}}' | grep -q scarebox-bench-vlm || { echo "container exited" >>"$LOG"; docker logs scarebox-bench-vlm 2>&1 | tail -30 >>"$LOG"; exit 1; }
    sleep 10
done
curl -sf http://127.0.0.1:8100/v1/models >/dev/null || { echo "NOT READY after 15 min" >>"$LOG"; exit 1; }
echo "ready_s=$(( $(date +%s) - t0 )) avail_loaded=$(avail)GiB" >>"$LOG"
~/.venv/bin/python scripts/bench_vlm.py --endpoint http://127.0.0.1:8100/v1 --model "$MODEL_NAME" \
    --crops testdata/crops --labels testdata/crops/labels.jsonl --sizes "$SIZES" \
    --out "runs/vlm/${MODEL_NAME}.jsonl" >>"$LOG" 2>&1
echo "records=$(wc -l <"runs/vlm/${MODEL_NAME}.jsonl") avail_end=$(avail)GiB end $(date -Is)" >>"$LOG"
