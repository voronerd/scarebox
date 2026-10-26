#!/usr/bin/env bash
# Fetches public Halloween costume-run clips for scarebox test data.
#
# Runs on gbox1 only (never on the CPU-only dev machine -- no video processing there).
# Usage:
#   scripts/fetch_testdata.sh fetch                  -- search, download <=12 clips, cut thumbnails
#   scripts/fetch_testdata.sh replay <clip_id> <start_seconds>  -- cut a 5-min replay.mp4
#
# "fetch" needs: ~/.venv with yt-dlp, imageio-ffmpeg, Pillow (installed here if missing).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HOME/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
CLIPS_DIR="$ROOT/testdata/clips"
THUMBS_DIR="$ROOT/testdata/thumbs"
mode="${1:-fetch}"

for pkg_mod in "yt_dlp:yt-dlp" "imageio_ffmpeg:imageio-ffmpeg" "PIL:Pillow"; do
    mod="${pkg_mod%%:*}"
    pkg="${pkg_mod##*:}"
    "$PY" -c "import $mod" 2>/dev/null || "$PIP" install -q -U "$pkg"
done

if [ "$mode" = "replay" ]; then
    clip_id="${2:?usage: fetch_testdata.sh replay <clip_id> <start_seconds>}"
    start="${3:?usage: fetch_testdata.sh replay <clip_id> <start_seconds>}"
    src="$CLIPS_DIR/${clip_id}.mp4"
    [ -f "$src" ] || { echo "fetch_testdata.sh: no such clip $src" >&2; exit 1; }
    FFMPEG="$("$PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
    "$FFMPEG" -y -ss "$start" -i "$src" -t 300 -c copy "$ROOT/testdata/replay.mp4" \
        2> "$ROOT/testdata/replay-cut.log" \
        || "$FFMPEG" -y -ss "$start" -i "$src" -t 300 -c:v libx264 -c:a aac "$ROOT/testdata/replay.mp4"
    echo "fetch_testdata.sh: wrote $ROOT/testdata/replay.mp4 (source=$clip_id start=${start}s)"
    exit 0
fi

mkdir -p "$CLIPS_DIR" "$THUMBS_DIR"
SOURCES="$CLIPS_DIR/SOURCES.tsv"
printf 'id\ttitle\turl\tduration_s\n' > "$SOURCES"

export SCAREBOX_ROOT="$ROOT"
"$PY" <<'PYEOF'
import json
import os
import subprocess
import sys

import imageio_ffmpeg
import yt_dlp

ROOT = os.environ["SCAREBOX_ROOT"]
CLIPS_DIR = os.path.join(ROOT, "testdata", "clips")
THUMBS_DIR = os.path.join(ROOT, "testdata", "thumbs")
SOURCES = os.path.join(CLIPS_DIR, "SOURCES.tsv")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

QUERIES = [
    "halloween costume fun run",
    "costume 5k finish line",
    "halloween parade kids costumes",
    "zombie run costume race",
    "marathon finish line costumes",
    "inflatable costume race",
]

MAX_CLIPS = 12
MAX_PER_QUERY = 2
MIN_DUR, MAX_DUR = 60, 900  # 1 to 15 minutes

flat_opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}
dl_opts = {
    "quiet": True,
    "no_warnings": True,
    "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]",
    "merge_output_format": "mp4",
    "ffmpeg_location": FFMPEG,
}

seen_ids = set()
rows = []
total = 0

with open(SOURCES, "a") as src_f:
    for q in QUERIES:
        if total >= MAX_CLIPS:
            break
        try:
            with yt_dlp.YoutubeDL(flat_opts) as ydl:
                info = ydl.extract_info(f"ytsearch5:{q}", download=False)
        except Exception as e:
            print(f"SEARCH FAILED for {q!r}: {e}", file=sys.stderr)
            continue

        kept_this_query = 0
        for e in info.get("entries") or []:
            if total >= MAX_CLIPS or kept_this_query >= MAX_PER_QUERY:
                break
            vid = e.get("id")
            dur = e.get("duration")
            if not vid or vid in seen_ids or not dur:
                continue
            if not (MIN_DUR <= dur <= MAX_DUR):
                continue
            url = e.get("url") or f"https://www.youtube.com/watch?v={vid}"
            title = (e.get("title") or "").replace("\t", " ").replace("\n", " ")
            out_path = os.path.join(CLIPS_DIR, f"{vid}.mp4")
            opts = dict(dl_opts)
            opts["outtmpl"] = out_path
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])
            except Exception as ex:
                print(f"DOWNLOAD FAILED for {vid} ({title}): {ex}", file=sys.stderr)
                continue
            if not os.path.exists(out_path):
                print(f"DOWNLOAD MISSING for {vid} ({title}) -- yt-dlp reported success but no file", file=sys.stderr)
                continue
            seen_ids.add(vid)
            kept_this_query += 1
            total += 1
            src_f.write(f"{vid}\t{title}\t{url}\t{dur}\n")
            src_f.flush()
            print(f"kept {vid} dur={dur}s title={title!r}", file=sys.stderr)

print(f"TOTAL downloaded: {total}", file=sys.stderr)

# Six evenly spaced thumbnails per clip, for picking the replay source by eye.
for vid in sorted(seen_ids):
    clip = os.path.join(CLIPS_DIR, f"{vid}.mp4")
    probe = subprocess.run(
        [FFMPEG, "-i", clip],
        capture_output=True, text=True,
    )
    dur = None
    for line in probe.stderr.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            dur = int(h) * 3600 + int(m) * 60 + float(s)
            break
    if not dur:
        continue
    for i in range(6):
        t = max(1.0, dur * (i + 1) / 7.0)
        out = os.path.join(THUMBS_DIR, f"{vid}_{i}.jpg")
        subprocess.run(
            [FFMPEG, "-y", "-ss", f"{t:.1f}", "-i", clip, "-frames:v", "1", "-q:v", "3", out],
            capture_output=True, text=True,
        )
print("thumbnails written", file=sys.stderr)
PYEOF
echo "fetch_testdata.sh: done. clips in $CLIPS_DIR, thumbnails in $THUMBS_DIR"
