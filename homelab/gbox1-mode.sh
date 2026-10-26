#!/usr/bin/env bash
# Author's homelab only: switch gbox1 between its normal fleet role and scarebox mode.
#
#   homelab/gbox1-mode.sh scarebox [--dry-run]   unload the big reason-l engine (whichever is loaded)
#   homelab/gbox1-mode.sh normal   [--dry-run]   stop scarebox containers, reload the engine the sanctioned way
#   homelab/gbox1-mode.sh status                 read-only: what's loaded, memory available
#
# The big engine changed on 2026-10-22 (fleet ADR 0087): Qwen3.8-Flash-Next (container gbox1-q38-vllm,
# loaded with ENGINE=q38 GUARD=0) replaced B1 (gbox1-b1-vllm), which stays on disk as the rollback.
# "scarebox" replicates the pre-stop stage of /srv/gbox1/compose/b1-cold-reload.sh for BOTH engines
# (stop guard + cache-drop units, remove launch marker, compose down, assert gone). While it is down
# the model-router sends reason-l to Gemini and the gbox1 watchdogs page: expected.
# "normal" runs b1-cold-reload.sh itself, the only sanctioned way to load an engine, and checks its
# exit. It reloads ENGINE (default q38, per ADR 0087); ENGINE=b1 restores the rollback. The memory guard
# stays OFF for both engines (Martin, 2026-10-23; ADR 0087 addendum) -- never pass GUARD=1 unasked.
# Runs from the author's workstation as martin (passwordless sudo on gbox1). Never run it from
# an unattended agent session.
set -uo pipefail

HOST="${GBOX1_HOST:-gbox1}"
MODE="${1:-status}"
DRY=0; [[ "${2:-}" == "--dry-run" ]] && DRY=1
ENGINE="${ENGINE:-q38}"
case "$ENGINE" in
    q38) GUARD="${GUARD:-0}"; NEED_GIB=100 ;;  # ~101 GiB resident (steady ~20 GiB free of 121)
    b1)  GUARD="${GUARD:-0}"; NEED_GIB=90 ;;   # 75 GiB + reserve; guard off for B1 too (Martin, 2026-10-23)
    *)   echo "ENGINE must be q38 or b1 (got '$ENGINE')" >&2; exit 2 ;;
esac
ENGINES="gbox1-q38-vllm:b6-q38-flash-vllm.yaml gbox1-b1-vllm:b1-qwen35-122b-vllm-brain.yaml"

run() {  # run a remote command, or print it in dry-run mode
    if (( DRY )); then echo "DRY-RUN would run on $HOST: $*"; else ssh "$HOST" "$@"; fi
}

status() {
    ssh "$HOST" 'echo "-- containers:"; docker ps --format "  {{.Names}}  {{.Status}}";
        echo "-- guard unit: $(systemctl is-active gbox1-r3-guard)";
        echo "-- shim /v1/models: $(curl -s -o /dev/null -w %{http_code} --max-time 5 http://127.0.0.1:8080/v1/models)";
        echo "-- scarebox boot unit: $(systemctl is-enabled scarebox.service 2>&1 | head -1)";
        free -g | awk "NR==2{print \"-- memory available: \"\$7\" GiB of \"\$2}"'
}

case "$MODE" in
status)
    status ;;
scarebox)
    echo "== gbox1 -> scarebox mode (unload the big engine, start scarebox, boot into it)"
    status
    run 'set -e
        if systemctl is-active --quiet gbox1-r3-guard; then sudo systemctl stop gbox1-r3-guard; fi
        if systemctl is-active --quiet gbox1-r3-cachedrop; then
            sudo systemctl stop gbox1-r3-cachedrop || true; sudo systemctl reset-failed gbox1-r3-cachedrop || true; fi
        sudo rm -f /run/gbox1-r3-launch.marker
        for pair in '"$ENGINES"'; do c=${pair%%:*}; f=${pair#*:}
            if docker inspect "$c" >/dev/null 2>&1; then
                (cd /srv/gbox1/compose && docker compose -f "$f" down); fi
            ! docker inspect "$c" >/dev/null 2>&1 || { echo "$c still present"; exit 1; }
        done
        # Field cold boot: the boot unit brings the stack up on every power-on while in this mode, and
        # the system-wide WirePlumber rule keeps every login session (incl. the GDM greeter) off the dongle.
        H=/home/scarebox/scarebox/deploy/host
        sudo install -m 644 "$H/scarebox.service" /etc/systemd/system/scarebox.service
        sudo install -d /etc/wireplumber/main.lua.d
        sudo install -m 644 "$H/51-scarebox-ignore-dongle.lua" /etc/wireplumber/main.lua.d/51-scarebox-ignore-dongle.lua
        sudo systemctl daemon-reload
        # enable + restart, not enable --now: the oneshot stays "active" (RemainAfterExit) from
        # an earlier boot, and --now on an active unit runs nothing (2026-10-24: only the app,
        # already up, was running).
        sudo systemctl enable scarebox.service
        sudo systemctl restart scarebox.service
        systemctl is-enabled scarebox.service'
    (( DRY )) || { echo "== after:"; status; }
    ;;
normal)
    echo "== gbox1 -> normal mode (stop scarebox, reload ENGINE=$ENGINE GUARD=$GUARD)"
    status
    if ssh "$HOST" "docker inspect gbox1-$ENGINE-vllm >/dev/null 2>&1"' && [ "$(curl -s -o /dev/null -w %{http_code} --max-time 5 http://127.0.0.1:8080/v1/models)" = 200 ]'; then
        echo "$ENGINE already loaded and serving: nothing to do"; exit 0
    fi
    run 'if [ -f /etc/systemd/system/scarebox.service ]; then sudo systemctl disable --now scarebox.service >/dev/null 2>&1 || true; fi
        for p in scarebox scarebox-bench; do docker compose -p $p stop >/dev/null 2>&1 || true; done
        docker ps --format "{{.Names}}" | grep -i scarebox && { echo "scarebox containers still running"; exit 1; } || true
        avail=$(free -g | awk "NR==2{print \$7}")
        [ "$avail" -ge '"$NEED_GIB"' ] || { echo "only ${avail} GiB available; '"$ENGINE"' needs ~'"$NEED_GIB"'. Stop other loads first."; exit 1; }
        ENGINE='"$ENGINE"' GUARD='"$GUARD"' /srv/gbox1/compose/b1-cold-reload.sh'
    rc=$?
    (( DRY )) || { echo "== after (b1-cold-reload exit $rc):"; status; }
    exit $rc
    ;;
*)
    echo "usage: $0 status | scarebox [--dry-run] | normal [--dry-run]" >&2; exit 2 ;;
esac
