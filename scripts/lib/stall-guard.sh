#!/bin/bash
# stall-guard.sh — sourceable bash library, bash + coreutils + sha256sum ONLY (no jq).
#
# PURPOSE: detect a stalled batch job by COMPLETED WORK UNITS, not by log activity.
#
# ORIGIN INCIDENT (2026-07-12, extraction-models baseline): dense xlsx chunks hit a
# deterministic output-cap runaway. The harness retried each identical request 3x at
# ~9 minutes a pop. The log stayed busy with RETRY lines the whole time — an
# activity-based guard (watching a log's modification-time / growth / elapsed time) slept through
# it. ~10 wall-hours burned before Martin eyeballed it: "if it's making the same error
# we're not learning much." A guard that keys on log activity cannot tell a busy retry
# loop from real work. sg_progress is deliberately the ONLY thing that resets the
# stall clock, and it is called ONLY when a caller-defined unit of work actually
# completes (a file written, a doc done, a chunk index advanced) — never on retries,
# never on log lines, never on elapsed time alone.
#
# USAGE:
#   source ~/scripts/lib/stall-guard.sh
#   sg_init "my-batch-job" 900          # 900s = 15min stall threshold
#   for chunk in "${chunks[@]}"; do
#       if process "$chunk"; then
#           sg_progress "$chunk"        # ONLY call this on real completed work
#       fi
#       sg_check || { echo "bailing, job stalled" >&2; break; }
#   done
#
#   # deterministic-failure short-circuit around a retry loop:
#   while ! do_thing; do
#       sg_fail_fast $? "$stderr_tail" || break   # nonzero == stop retrying
#   done
#
# STATE: kept in a caller-supplied dir via ${SG_STATE_DIR:-}. If unset when sg_init
# runs, sg_init creates one with mktemp -d and exports SG_STATE_DIR so the other
# functions find it. Re-running sg_init resets all state (fresh job).
#
# CONTRACT: safe to source under `set -u`. Every happy path ends in an explicit
# `return 0` so a `set -e` caller never trips on an internal command's exit status.
# Missing/absent state is a LOUD failure (return 2), never a silent 0 — a guard that
# can go quiet on a broken setup is worse than no guard at all.

sg_init() {
    local job="$1"
    local threshold="${2:-900}"
    if [ -z "${SG_STATE_DIR:-}" ]; then
        SG_STATE_DIR=$(mktemp -d)
        export SG_STATE_DIR
    fi
    mkdir -p "$SG_STATE_DIR"
    local now
    now=$(date +%s)
    printf '%s\n' "$job" > "$SG_STATE_DIR/job"
    printf '%s\n' "$threshold" > "$SG_STATE_DIR/threshold"
    printf '%s\n' "0" > "$SG_STATE_DIR/units"
    printf '%s\n' "$now" > "$SG_STATE_DIR/last_progress"
    printf '%s\n' "$now" > "$SG_STATE_DIR/start"
    : > "$SG_STATE_DIR/progress.log"
    rm -f "$SG_STATE_DIR/fingerprint"
    return 0
}

sg_progress() {
    local unit_id="${1:-unit}"
    if [ -z "${SG_STATE_DIR:-}" ] || [ ! -d "$SG_STATE_DIR" ]; then
        echo "sg_progress: no stall-guard state (run sg_init first)" >&2
        return 2
    fi
    local now units
    now=$(date +%s)
    printf '%s %s\n' "$now" "$unit_id" >> "$SG_STATE_DIR/progress.log"
    units=$(cat "$SG_STATE_DIR/units" 2>/dev/null || echo 0)
    units=$((units + 1))
    printf '%s\n' "$units" > "$SG_STATE_DIR/units"
    printf '%s\n' "$now" > "$SG_STATE_DIR/last_progress"
    return 0
}

sg_check() {
    if [ -z "${SG_STATE_DIR:-}" ] || [ ! -d "$SG_STATE_DIR" ] || [ ! -f "$SG_STATE_DIR/last_progress" ]; then
        echo "sg_check: no stall-guard state (run sg_init first)" >&2
        return 2
    fi
    local last threshold now elapsed job
    last=$(cat "$SG_STATE_DIR/last_progress")
    threshold=$(cat "$SG_STATE_DIR/threshold" 2>/dev/null || echo 900)
    now=$(date +%s)
    elapsed=$((now - last))
    if [ "$elapsed" -le "$threshold" ]; then
        return 0
    fi
    job=$(cat "$SG_STATE_DIR/job" 2>/dev/null || echo "unknown")
    echo "stall-guard: job '$job' stalled — no completed work units in ${elapsed}s (threshold ${threshold}s)" >&2
    return 3
}

sg_fail_fast() {
    local exit_code="$1"
    local tail_input="$2"
    if [ -z "${SG_STATE_DIR:-}" ] || [ ! -d "$SG_STATE_DIR" ]; then
        echo "sg_fail_fast: no stall-guard state (run sg_init first)" >&2
        return 2
    fi
    local tail_text fp prev
    if [ -f "$tail_input" ]; then
        tail_text=$(tail -n 3 "$tail_input")
    else
        tail_text=$(printf '%s\n' "$tail_input" | tail -n 3)
    fi
    fp=$(printf '%s\n%s' "$exit_code" "$tail_text" | sha256sum | cut -d' ' -f1)
    if [ -f "$SG_STATE_DIR/fingerprint" ]; then
        prev=$(cat "$SG_STATE_DIR/fingerprint")
        if [ "$prev" = "$fp" ]; then
            echo "stall-guard: deterministic failure, not retrying: $fp" >&2
            return 4
        fi
    fi
    printf '%s\n' "$fp" > "$SG_STATE_DIR/fingerprint"
    return 0
}
