#!/usr/bin/env bash
# soak.sh (04-03 Task 1): brings the full stack up on `config/soak.yaml` (bench.yaml +
# `source.loop: true` so replay.mp4 never runs out + `watchdog.debug_hooks: true` so the
# hang injection below has somewhere to fire), samples memory/GPU/restarts/heartbeat every
# 5s into runs/soak.csv, rotates runs/wav/ so disk can't fill, injects an app kill / vlm
# kill / hang at 60/90/105 of a 120-point scale (SOAK_MINUTES scales all three), tears the
# stack down, and writes runs/evidence/soak.json.
#
# Usage (nohup-friendly, no tty needed):
#   SOAK_MINUTES=120 nohup bash scripts/soak.sh > runs/soak.log 2>&1 < /dev/null &
#
# `SOAK_MINUTES` (default 120) scales every timing point:
#   app kill  at SOAK_MINUTES * 0.50   (60/120 of a 2h soak)
#   vlm kill  at SOAK_MINUTES * 0.75   (90/120)
#   hang      at SOAK_MINUTES * 0.875  (105/120, hub amendment -- kills aren't hangs)
#   stop      at SOAK_MINUTES          (120/120)
#   wav rotate every SOAK_MINUTES / 12 minutes (10/120 of a 2h soak)
#
# Memory floor: if `free -m`'s available dips below 16 GiB at any sample, the soak aborts
# immediately (stack torn down, evidence written with aborted_early=true) rather than
# risking an OOM on gbox1 (project rule: never let scarebox work push gbox1 into memory trouble).
set -u
set -o pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$ROOT" ]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT" || exit 1

SOAK_MINUTES="${SOAK_MINUTES:-120}"
SAMPLE_S=5
MEM_FLOOR_MB=16384 # 16 GiB, binary

TOTAL_S=$(awk "BEGIN{print int($SOAK_MINUTES*60)}")
APP_KILL_S=$(awk "BEGIN{print int($SOAK_MINUTES*60*0.50)}")
VLM_KILL_S=$(awk "BEGIN{print int($SOAK_MINUTES*60*0.75)}")
HANG_S=$(awk "BEGIN{print int($SOAK_MINUTES*60*0.875)}")
ROTATE_S=$(awk "BEGIN{print int($SOAK_MINUTES*60/12)}")
[ "$ROTATE_S" -lt 10 ] && ROTATE_S=10

RUNS="$ROOT/runs"
mkdir -p "$RUNS/wav"
CSV="$RUNS/soak.csv"
APPLOG="$RUNS/soak-app.log"
EVIDENCE_DIR="$ROOT/runs/evidence"
mkdir -p "$EVIDENCE_DIR"
EVIDENCE="$EVIDENCE_DIR/soak.json"

rm -f "$RUNS/SOAK_DONE" "$RUNS/debug_block_mainloop"
: > "$APPLOG"
echo "timestamp,mem_avail_mb,gpu_power_w,gpu_temp_c,gpu_util_pct,restart_app,restart_vlm,restart_guard,heartbeat_age_s,events_count" > "$CSV"

log() { echo "[soak] $(date -Iseconds) $*"; }

# app.py opens events.jsonl in append mode and never truncates it -- a leftover file from
# an earlier bench/aim session would inflate this run's event/path-mix/latency numbers with
# unrelated history (dry run 1, 04-03: 186 stale events counted before a single new frame
# had been processed). Archive it out of the way so THIS run starts from a clean file.
if [ -f "$RUNS/events.jsonl" ]; then
    mv "$RUNS/events.jsonl" "$RUNS/events.jsonl.pre-soak-$(date +%s)"
    log "archived pre-existing events.jsonl before starting"
fi

mem_avail_mb() { free -m | awk '/^Mem:/{print $7}'; }
gpu_line() { nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu --format=csv,noheader 2>/dev/null | head -1; }
restart_count() { docker inspect -f '{{.RestartCount}}' "$1" 2>/dev/null; }
heartbeat_age() {
    if [ -f "$RUNS/heartbeat" ]; then
        echo $(( $(date +%s) - $(stat -c %Y "$RUNS/heartbeat") ))
    else
        echo -1
    fi
}
events_count() {
    if [ -f "$RUNS/events.jsonl" ]; then
        grep -c '"path"' "$RUNS/events.jsonl" 2>/dev/null || true
    else
        echo 0
    fi
}
judge_fail_count() { grep -c 'judge\.fail' "$APPLOG" 2>/dev/null || true; }
is_healthy() { [ "$(docker inspect -f '{{.State.Health.Status}}' "$1" 2>/dev/null)" = "healthy" ]; }

log "starting soak: SOAK_MINUTES=$SOAK_MINUTES total_s=$TOTAL_S app_kill_s=$APP_KILL_S vlm_kill_s=$VLM_KILL_S hang_s=$HANG_S rotate_s=$ROTATE_S"

log "bringing stack up (config/soak.yaml, user scarebox)"
SCAREBOX_CONFIG=config/soak.yaml docker compose -p scarebox -f deploy/compose.yaml --env-file deploy/images.env up -d
if [ $? -ne 0 ]; then
    log "FATAL: docker compose up failed"
    exit 1
fi

# --env-file is required here too: compose parses (and ${IMAGE_VLLM}-expands) the WHOLE
# file for any subcommand, including `logs`, even though logs only needs the `app` service's
# own definition -- omitting it fails the whole invocation ("invalid compose project") and
# this tailer dies silently in the background, the dry run's first real bug (04-03).
docker compose -p scarebox -f deploy/compose.yaml --env-file deploy/images.env logs -f --no-color app >> "$APPLOG" 2>&1 &
LOGS_PID=$!

START_EPOCH=$(date +%s)
LAST_ROTATE_EPOCH=$START_EPOCH

APP_KILLED=0; APP_KILL_EPOCH=0; APP_RECOVERED_EPOCH=0
VLM_KILLED=0; VLM_KILL_EPOCH=0; VLM_RECOVERED_EPOCH=0
VLM_EVENTS_AT_KILL=0; VLM_JFAIL_AT_KILL=0; VLM_EVENTS_AT_RECOVER=0; VLM_JFAIL_AT_RECOVER=0
HANG_INJECTED=0; HANG_EPOCH=0; HANG_RESTART_BEFORE=0
HANG_EXIT_DETECTED=0; HANG_EXIT_EPOCH=0; HANG_EXIT_CODE=""
HANG_RECOVERED_EPOCH=0

ABORTED=0
ABORT_REASON=""

MIN_MEM_MB=999999999
GPU_POWER_SUM=0.0
GPU_POWER_N=0
GPU_POWER_MAX=0.0
SAMPLE_N=0

while :; do
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START_EPOCH ))
    [ "$ELAPSED" -ge "$TOTAL_S" ] && break

    MEM=$(mem_avail_mb)
    GPU=$(gpu_line)
    GPU_POWER=$(echo "$GPU" | awk -F',' '{gsub(/[^0-9.]/,"",$1); print $1}')
    GPU_TEMP=$(echo "$GPU" | awk -F',' '{gsub(/[^0-9.]/,"",$2); print $2}')
    GPU_UTIL=$(echo "$GPU" | awk -F',' '{gsub(/[^0-9.]/,"",$3); print $3}')
    RC_APP=$(restart_count scarebox-app); RC_APP="${RC_APP:--1}"
    RC_VLM=$(restart_count scarebox-vlm); RC_VLM="${RC_VLM:--1}"
    RC_GUARD=$(restart_count scarebox-guard); RC_GUARD="${RC_GUARD:--1}"
    HB_AGE=$(heartbeat_age)
    EVC=$(events_count)
    SAMPLE_N=$((SAMPLE_N+1))

    echo "$(date -Iseconds),$MEM,$GPU_POWER,$GPU_TEMP,$GPU_UTIL,$RC_APP,$RC_VLM,$RC_GUARD,$HB_AGE,$EVC" >> "$CSV"

    if [ -n "$MEM" ] && [ "$MEM" -lt "$MIN_MEM_MB" ] 2>/dev/null; then MIN_MEM_MB=$MEM; fi
    if [ -n "$GPU_POWER" ]; then
        GPU_POWER_SUM=$(awk "BEGIN{print $GPU_POWER_SUM+$GPU_POWER}")
        GPU_POWER_N=$((GPU_POWER_N+1))
        GPU_POWER_MAX=$(awk "BEGIN{print ($GPU_POWER>$GPU_POWER_MAX)?$GPU_POWER:$GPU_POWER_MAX}")
    fi

    if [ -n "$MEM" ] && [ "$MEM" -lt "$MEM_FLOOR_MB" ] 2>/dev/null; then
        log "ABORT: available memory ${MEM}MB below floor ${MEM_FLOOR_MB}MB at elapsed_s=$ELAPSED"
        ABORTED=1
        ABORT_REASON="mem_avail_mb=$MEM below floor=$MEM_FLOOR_MB at elapsed_s=$ELAPSED"
        break
    fi

    if [ $(( NOW - LAST_ROTATE_EPOCH )) -ge "$ROTATE_S" ]; then
        ROTATE_MIN=$(awk "BEGIN{print $ROTATE_S/60}")
        find "$RUNS/wav" -name 'stage-*.wav' -mmin "+$ROTATE_MIN" -delete 2>/dev/null
        LAST_ROTATE_EPOCH=$NOW
        log "wav rotation: purged files older than ${ROTATE_S}s"
    fi

    # --- app kill (60/120) ---
    if [ "$APP_KILLED" -eq 0 ] && [ "$ELAPSED" -ge "$APP_KILL_S" ]; then
        log "injecting APP KILL at elapsed_s=$ELAPSED"
        # Confirmed on gbox1 (dry run 1, 04-03): `docker kill` marks a container the SAME
        # way `docker stop` does for restart-policy purposes -- `unless-stopped` then
        # NEVER auto-restarts it (documented Docker behaviour, not scarebox-specific; the
        # watchdog's own os._exit(1) self-exit is a different, unaffected path -- proven by
        # a throwaway busybox container crash-looping fine under the same policy). A real
        # unexpected death (OOM, segfault) restarts itself; an operator-issued `docker kill`
        # does not, so the kill-injection needs an explicit follow-up `docker start` to
        # actually exercise app_recover_s instead of leaving the container dead for good.
        docker kill scarebox-app >/dev/null 2>&1
        docker start scarebox-app >/dev/null 2>&1
        # `docker compose logs -f` does not auto-resubscribe once the container it's
        # following has stopped and been restarted (even the same container ID) -- dry
        # run 2 (04-03) lost everything after this point (vlm-kill/hang log evidence) to
        # a tailer that quietly stopped updating. Re-attach a fresh one.
        kill "$LOGS_PID" >/dev/null 2>&1
        docker compose -p scarebox -f deploy/compose.yaml --env-file deploy/images.env logs -f --no-color app >> "$APPLOG" 2>&1 &
        LOGS_PID=$!
        APP_KILLED=1; APP_KILL_EPOCH=$NOW
    fi
    if [ "$APP_KILLED" -eq 1 ] && [ "$APP_RECOVERED_EPOCH" -eq 0 ] && is_healthy scarebox-app; then
        APP_RECOVERED_EPOCH=$NOW
        log "app recovered $(( APP_RECOVERED_EPOCH - APP_KILL_EPOCH ))s after kill"
    fi

    # --- vlm kill (90/120) ---
    if [ "$VLM_KILLED" -eq 0 ] && [ "$ELAPSED" -ge "$VLM_KILL_S" ]; then
        log "injecting VLM KILL at elapsed_s=$ELAPSED"
        VLM_EVENTS_AT_KILL=$EVC
        VLM_JFAIL_AT_KILL=$(judge_fail_count)
        docker kill scarebox-vlm >/dev/null 2>&1
        docker start scarebox-vlm >/dev/null 2>&1  # see the app-kill comment above
        VLM_KILLED=1; VLM_KILL_EPOCH=$NOW
    fi
    if [ "$VLM_KILLED" -eq 1 ] && [ "$VLM_RECOVERED_EPOCH" -eq 0 ] && is_healthy scarebox-vlm; then
        VLM_RECOVERED_EPOCH=$NOW
        VLM_EVENTS_AT_RECOVER=$EVC
        VLM_JFAIL_AT_RECOVER=$(judge_fail_count)
        log "vlm recovered $(( VLM_RECOVERED_EPOCH - VLM_KILL_EPOCH ))s after kill"
    fi

    # --- hang (105/120): docker pause freezes _watchdog's own task too, so this uses the
    # software hook instead (scarebox/app.py + watchdog.debug_hooks) -- touching the file
    # makes the per-frame loop sleep 35s once, which the real _watchdog (tick_timeout_s,
    # default 20s) should trip on well before the sleep ends.
    if [ "$HANG_INJECTED" -eq 0 ] && [ "$ELAPSED" -ge "$HANG_S" ]; then
        log "injecting HANG (debug hook) at elapsed_s=$ELAPSED"
        HANG_RESTART_BEFORE=$RC_APP
        touch "$RUNS/debug_block_mainloop"
        HANG_INJECTED=1; HANG_EPOCH=$NOW
    fi
    if [ "$HANG_INJECTED" -eq 1 ] && [ "$HANG_EXIT_DETECTED" -eq 0 ] && [ "$RC_APP" -gt "$HANG_RESTART_BEFORE" ] 2>/dev/null; then
        HANG_EXIT_DETECTED=1; HANG_EXIT_EPOCH=$NOW
        HANG_EXIT_CODE=$(docker inspect -f '{{.State.ExitCode}}' scarebox-app 2>/dev/null)
        log "hang self-exit detected (restart $HANG_RESTART_BEFORE -> $RC_APP, exit_code=$HANG_EXIT_CODE) at elapsed_s=$ELAPSED"
        # Same logs-f re-attach as the app-kill injection -- this self-exit also stops and
        # restarts the container, which also kills the follow stream.
        kill "$LOGS_PID" >/dev/null 2>&1
        docker compose -p scarebox -f deploy/compose.yaml --env-file deploy/images.env logs -f --no-color app >> "$APPLOG" 2>&1 &
        LOGS_PID=$!
    fi
    if [ "$HANG_EXIT_DETECTED" -eq 1 ] && [ "$HANG_RECOVERED_EPOCH" -eq 0 ] && is_healthy scarebox-app; then
        HANG_RECOVERED_EPOCH=$NOW
        log "app recovered $(( HANG_RECOVERED_EPOCH - HANG_EXIT_EPOCH ))s after hang self-exit"
    fi

    sleep "$SAMPLE_S"
done

DURATION_S=$(( $(date +%s) - START_EPOCH ))
log "soak loop ended at elapsed_s=$DURATION_S aborted=$ABORTED"

# --- final latency report (while the app container may still be up) ---
LATENCY_JSON="{}"
if docker ps --format '{{.Names}}' | grep -qx scarebox-app; then
    LATENCY_JSON=$(docker exec --user 1001:1001 -w /app scarebox-app python scripts/latency_report.py runs/events.jsonl --json 2>/dev/null)
    [ -z "$LATENCY_JSON" ] && LATENCY_JSON="{}"
fi

kill "$LOGS_PID" >/dev/null 2>&1
wait "$LOGS_PID" 2>/dev/null

log "stack down"
# Same --env-file requirement as the up/logs invocations above (dry run 1, 04-03: omitting
# it here silently failed teardown -- scarebox-guard was still running 15 minutes later).
docker compose -p scarebox -f deploy/compose.yaml --env-file deploy/images.env down >/dev/null 2>&1

WATCHDOG_TRIPPED=false
grep -q 'app\.watchdog_trip' "$APPLOG" 2>/dev/null && WATCHDOG_TRIPPED=true

EXCEPTIONS_N=$(grep -c 'Traceback (most recent call last)' "$APPLOG" 2>/dev/null || true)
EXCEPTIONS_SAMPLE=$(grep -oE '[A-Za-z_.]+(Error|Exception):.*' "$APPLOG" 2>/dev/null | sort -u | head -20)

if [ "$GPU_POWER_N" -gt 0 ]; then
    GPU_POWER_MEAN=$(awk "BEGIN{printf \"%.2f\", $GPU_POWER_SUM/$GPU_POWER_N}")
else
    GPU_POWER_MEAN=0
fi
if [ "$MIN_MEM_MB" -eq 999999999 ]; then MIN_MEM_MB=0; fi
MIN_MEM_GIB=$(awk "BEGIN{printf \"%.2f\", $MIN_MEM_MB/1024}")
DURATION_MIN=$(awk "BEGIN{printf \"%.2f\", $DURATION_S/60}")

if [ "$APP_KILLED" -eq 1 ] && [ "$APP_RECOVERED_EPOCH" -gt 0 ]; then
    APP_RECOVER_S=$(( APP_RECOVERED_EPOCH - APP_KILL_EPOCH ))
else
    APP_RECOVER_S=-1
fi

VLM_JFAIL_DURING_OUTAGE=0
if [ "$VLM_RECOVERED_EPOCH" -gt 0 ]; then
    VLM_JFAIL_DURING_OUTAGE=$(( VLM_JFAIL_AT_RECOVER - VLM_JFAIL_AT_KILL ))
fi
VLM_RECOVER_S=-1
[ "$VLM_RECOVERED_EPOCH" -gt 0 ] && VLM_RECOVER_S=$(( VLM_RECOVERED_EPOCH - VLM_KILL_EPOCH ))
# Did any event resume on the live path (vs. judge-fail fallback) after vlm came back?
VLM_LIVE_RESUMED=false
if [ "$VLM_RECOVERED_EPOCH" -gt 0 ] && [ -f "$RUNS/events.jsonl" ]; then
    TAIL_N=$(( $(events_count) - VLM_EVENTS_AT_RECOVER ))
    if [ "$TAIL_N" -gt 0 ] && tail -n "$TAIL_N" "$RUNS/events.jsonl" 2>/dev/null | grep -q '"path": *"live"'; then
        VLM_LIVE_RESUMED=true
    fi
fi

HANG_RECOVER_S=-1
[ "$HANG_EXIT_DETECTED" -eq 1 ] && [ "$HANG_RECOVERED_EPOCH" -gt 0 ] && HANG_RECOVER_S=$(( HANG_RECOVERED_EPOCH - HANG_EXIT_EPOCH ))

export SOAK_MINUTES DURATION_MIN DURATION_S ABORTED ABORT_REASON MIN_MEM_GIB MIN_MEM_MB \
    GPU_POWER_MEAN GPU_POWER_MAX APP_KILLED APP_KILL_S APP_RECOVER_S VLM_KILLED VLM_KILL_S \
    VLM_RECOVER_S VLM_JFAIL_DURING_OUTAGE VLM_LIVE_RESUMED HANG_INJECTED HANG_S \
    HANG_EXIT_DETECTED HANG_EXIT_CODE HANG_RECOVER_S WATCHDOG_TRIPPED EXCEPTIONS_N \
    EXCEPTIONS_SAMPLE LATENCY_JSON SAMPLE_N

python3 - "$EVIDENCE" <<'PYEOF'
import json
import os
import sys

out_path = sys.argv[1]
try:
    latency = json.loads(os.environ.get("LATENCY_JSON", "{}") or "{}")
except json.JSONDecodeError:
    latency = {}

def _f(name, default=0.0):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

def _i(name, default=0):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

def _b(name):
    # Bash passes some flags as "true"/"false" (WATCHDOG_TRIPPED, VLM_LIVE_RESUMED) and
    # others as the native 0/1 ints the shell loop already compares with -eq (APP_KILLED,
    # VLM_KILLED, HANG_INJECTED, HANG_EXIT_DETECTED, ABORTED) -- both forms must count as
    # true here (dry run 1, 04-03: comparing only against "true" left every injection
    # reporting `injected: false` even though the log showed it firing).
    return os.environ.get(name, "0") in ("1", "true", "True")

evidence = {
    "soak_minutes_configured": _f("SOAK_MINUTES"),
    "duration_min": _f("DURATION_MIN"),
    "duration_s": _i("DURATION_S"),
    "aborted_early": _b("ABORTED"),
    "abort_reason": os.environ.get("ABORT_REASON") or None,
    "sample_count": _i("SAMPLE_N"),
    "events": latency.get("n_events", 0),
    "path_mix": latency.get("path_mix", {}),
    "live_share": latency.get("live_share", 0.0),
    "total_p95_ms": latency.get("total_p95_ms", 0.0),
    "total_p50_ms": latency.get("total_p50_ms", 0.0),
    "total_max_ms": latency.get("total_max_ms", 0.0),
    "min_mem_avail_gib": _f("MIN_MEM_GIB"),
    "min_mem_avail_mb": _i("MIN_MEM_MB"),
    "mean_gpu_power_w": _f("GPU_POWER_MEAN"),
    "max_gpu_power_w": _f("GPU_POWER_MAX"),
    "app_recover_s": _i("APP_RECOVER_S"),
    "app_kill": {
        "injected": _b("APP_KILLED"),
        "scheduled_at_s": _i("APP_KILL_S"),
        "recover_s": _i("APP_RECOVER_S"),
    },
    "vlm_kill": {
        "injected": _b("VLM_KILLED"),
        "scheduled_at_s": _i("VLM_KILL_S"),
        "recover_s": _i("VLM_RECOVER_S"),
        "judge_fail_during_outage": _i("VLM_JFAIL_DURING_OUTAGE"),
        "library_only_during_outage": _i("VLM_JFAIL_DURING_OUTAGE") > 0,
        "live_resumed_after_recovery": _b("VLM_LIVE_RESUMED"),
    },
    "hang": {
        "injected": _b("HANG_INJECTED"),
        "scheduled_at_s": _i("HANG_S"),
        "method": "debug_hook (docker pause freezes the watchdog task too; "
                  "runs/debug_block_mainloop + watchdog.debug_hooks blocks the real "
                  "per-frame loop for 35s instead)",
        "watchdog_tripped": _b("WATCHDOG_TRIPPED"),
        "self_exited": _b("HANG_EXIT_DETECTED"),
        "exit_code": os.environ.get("HANG_EXIT_CODE") or None,
        "recover_s": _i("HANG_RECOVER_S"),
    },
    "exceptions_in_app_log": {
        "count": _i("EXCEPTIONS_N"),
        "sample": [l for l in os.environ.get("EXCEPTIONS_SAMPLE", "").splitlines() if l],
    },
}

with open(out_path, "w") as f:
    json.dump(evidence, f, indent=2)
    f.write("\n")
PYEOF

log "evidence written to $EVIDENCE"
date -Iseconds > "$RUNS/SOAK_DONE"
log "done"
