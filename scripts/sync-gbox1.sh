#!/usr/bin/env bash
# Sync this repo's committed HEAD to the execution host (default: ssh alias gbox1-scarebox).
# Uses git bundle + scp + fast-forward merge: no git push, no remote rsync.
# Untracked files on the remote (models/, .env, runs/, testdata/) are never touched.
# Usage: scripts/sync-gbox1.sh [--env]   (--env also copies .env, mode 600)
set -euo pipefail

HOST="${SCAREBOX_HOST:-gbox1-scarebox}"
REMOTE_DIR="${SCAREBOX_REMOTE_DIR:-scarebox}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
BUNDLE="$(mktemp --suffix=.bundle)"
trap 'rm -f "$BUNDLE"' EXIT

if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]]; then
    echo "sync-gbox1: uncommitted changes to tracked files; only committed HEAD is synced" >&2
fi

git -C "$ROOT" bundle create "$BUNDLE" "$BRANCH" >/dev/null 2>&1
scp -q "$BUNDLE" "$HOST:/tmp/scarebox-sync.bundle"

# Files whose live copy on the GPU host wins over the repo: the line library is reviewed in
# place by the vet page (statuses, edits, rendered wav paths), so a sync must never clobber it.
# Pull it back with `scarebox bake pull` and commit to update the repo copy.
LIVE_FILES="library/manifest.json"

ssh "$HOST" "set -e
if [ ! -d $REMOTE_DIR/.git ]; then
    git clone -q -b $BRANCH /tmp/scarebox-sync.bundle $REMOTE_DIR
else
    cd $REMOTE_DIR
    for f in $LIVE_FILES; do
        if [ -f \"\$f\" ] && ! git diff --quiet -- \"\$f\"; then
            cp \"\$f\" \"/tmp/live.\$(echo \$f | tr / _)\"
            git checkout -q -- \"\$f\"
        fi
    done
    git fetch -q /tmp/scarebox-sync.bundle $BRANCH
    git merge -q --ff-only FETCH_HEAD
    for f in $LIVE_FILES; do
        saved=\"/tmp/live.\$(echo \$f | tr / _)\"
        if [ -f \"\$saved\" ]; then mv \"\$saved\" \"\$f\"; echo \"sync-gbox1: kept live \$f\"; fi
    done
    cd - >/dev/null
fi
rm -f /tmp/scarebox-sync.bundle
git -C $REMOTE_DIR log --oneline -1"

if [[ "${1:-}" == "--env" ]]; then
    scp -q "$ROOT/.env" "$HOST:$REMOTE_DIR/.env"
    ssh "$HOST" "chmod 600 $REMOTE_DIR/.env"
    echo "sync-gbox1: .env copied (600)"
fi
