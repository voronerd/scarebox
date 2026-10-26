#!/usr/bin/env bash
# Export the public tree of scarebox into DEST, then run the leak gate on it.
#
# Usage: scripts/export_public.sh DEST
#
# Writes every git-tracked file under the allowlist below (working-tree
# content, so uncommitted edits to tracked files are exported) into DEST, then
# runs scripts/leak_gate.py on DEST and exits with the gate's status:
#   0  clean: DEST is safe to publish
#   1  the gate found leaks (it prints them as path:line: rule)
#   2  usage or input error (nothing exported, or the gate could not run)
#
# DEST must not exist, be empty, or hold only a .git entry (so the export can
# fill a fresh clone of the public repo); anything else is refused untouched.
#
# The gate gets the repo's .env (--env) and private deny list
# scripts/leak_gate.deny (--deny-file) when they exist. Both are git-ignored,
# so in a linked git worktree they are looked up in the worktree first and
# then in the main working tree. Their contents are never read here.
set -euo pipefail

# The repo is the one this script lives in, whatever the caller's environment.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR

ALLOWLIST=(
    scarebox
    tests
    config
    deploy
    homelab
    scripts
    docs
    library/manifest.json
    library/costume_packs.json
    library/voice/reference.wav
    library/voice/reference_excited.wav
    library/voice/reference_excited.txt
    library/voice/reference_calm.wav
    library/voice/reference_calm.txt
    library/voice/SOURCES.md
    library/sfx
    README.md
    AGENTS.md
    THIRD_PARTY.md
    LICENSE
    pyproject.toml
    .gitignore
    .env.example
)

die() { echo "export_public: $*" >&2; exit 2; }

[ "$#" -eq 1 ] && [ -n "$1" ] || die "usage: scripts/export_public.sh DEST"
dest=$1

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
top=$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null) \
    || die "$script_dir is not inside a git working tree"
common=$(git -C "$top" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) \
    || die "cannot find the git directory of $top"
main_top=$(dirname "$common")

# --- DEST: absent, empty, or only .git ------------------------------------
if [ -e "$dest" ] || [ -L "$dest" ]; then
    [ -d "$dest" ] && [ ! -L "$dest" ] || die "DEST $dest exists and is not a directory"
    while IFS= read -r -d '' entry; do
        [ "$(basename "$entry")" = ".git" ] \
            || die "DEST $dest is not empty (holds $(basename "$entry")); refusing to export into it"
    done < <(find "$dest" -mindepth 1 -maxdepth 1 -print0)
else
    mkdir -p "$dest" || die "cannot create DEST $dest"
fi
dest_abs=$(cd "$dest" && pwd -P)

# --- copy tracked files under the allowlist -------------------------------
# A tracked file deleted from the working tree is skipped (it would not ship).
list=$(mktemp)
trap 'rm -f "$list"' EXIT
git -C "$top" ls-files -z --cached -- "${ALLOWLIST[@]}" \
    | while IFS= read -r -d '' path; do
        if [ -e "$top/$path" ] || [ -L "$top/$path" ]; then
            printf '%s\0' "$path"
        fi
    done >"$list"

[ -s "$list" ] || die "no tracked files under the allowlist in $top"

tar -C "$top" --null --no-recursion -T "$list" -cf - \
    | tar -C "$dest_abs" -xf - \
    || die "copying files into $dest_abs failed"

count=$(tr -cd '\0' <"$list" | wc -c)
echo "export_public: wrote $count files to $dest_abs" >&2

# --- leak gate ------------------------------------------------------------
find_private() {  # $1 = path relative to the repo top; prints the first that exists
    local cand
    for cand in "$top/$1" "$main_top/$1"; do
        if [ -f "$cand" ]; then
            printf '%s\n' "$cand"
            return 0
        fi
    done
    return 1
}

gate_args=("$dest_abs")
if env_file=$(find_private .env); then
    gate_args+=(--env "$env_file")
else
    echo "export_public: no .env found; gate runs without env values" >&2
fi
if deny_file=$(find_private scripts/leak_gate.deny); then
    gate_args+=(--deny-file "$deny_file")
else
    echo "export_public: no scripts/leak_gate.deny found; gate runs without the deny list" >&2
fi

python=${PYTHON:-python3}
set +e
"$python" "$top/scripts/leak_gate.py" "${gate_args[@]}"
status=$?
set -e
case "$status" in
    0) echo "export_public: leak gate clean" >&2 ;;
    1) echo "export_public: leak gate FOUND LEAKS in $dest_abs; do not publish" >&2 ;;
    *) echo "export_public: leak gate failed to run (status $status)" >&2; status=2 ;;
esac
exit "$status"
