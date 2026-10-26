#!/usr/bin/env python3
"""track_shots.py: one best picture of every person tracked in a capture run, plus contact
sheets for grading (2026-10-19, timeline tuning).

Ground truth for "who should have got a line" has to be per person over the whole video,
including people who never triggered. The capture log keeps only boxes per frame (the video
is the dataset), so this decodes the replay video once and, for each track, crops the frame
where that person's box was biggest (padded 15%).

Run in the app image on gbox1, from the repo root:
  python scripts/track_shots.py runs/capture/<run> testdata/hallway30.mp4 [--min-frames 8]
Writes <run>/tracks/<id>.jpg, <run>/sheets/sheet_NN.jpg (4x3 numbered tiles) and
<run>/tracks.json ({id: {first_f, last_f, n, best_f, best_h, box}}).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import av
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("video")
ap.add_argument("--min-frames", type=int, default=8, help="ignore flicker tracks shorter than this")
ap.add_argument("--min-height", type=float, default=0.2, help="best box height as a share of the frame")
ap.add_argument("--frame-h", type=int, default=1080)
args = ap.parse_args()
run = Path(args.run)

tracks: dict[int, dict] = defaultdict(lambda: {"first_f": None, "last_f": None, "n": 0, "best_h": 0.0})
for line in (run / "tracks.jsonl").open():
    row = json.loads(line)
    f = row["f"]
    for tid, x1, y1, x2, y2 in row["b"]:
        t = tracks[tid]
        if t["first_f"] is None:
            t["first_f"], t["first_box"] = f, [x1, y1, x2, y2]
        t["last_f"], t["last_box"] = f, [x1, y1, x2, y2]
        t["n"] += 1
        if y2 - y1 > t["best_h"]:
            t["best_h"], t["best_f"], t["box"] = y2 - y1, f, [x1, y1, x2, y2]

# Stitch tracker ID switches: the tracker sometimes loses someone for a moment and gives them
# a new ID, which would count one child as two "people". Track B continues track A if it
# starts within 20 frames of A ending, near A's last position, at a similar size.
def _centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


parent = {tid: tid for tid in tracks}


def root(t):
    while parent[t] != t:
        parent[t] = parent[parent[t]]
        t = parent[t]
    return t


order = sorted(tracks, key=lambda k: tracks[k]["first_f"])
for b in order:
    tb = tracks[b]
    best = None
    for a in order:
        ta = tracks[a]
        if a == b or not (0 < tb["first_f"] - ta["last_f"] <= 20):
            continue
        (ax, ay), (bx, by) = _centre(ta["last_box"]), _centre(tb["first_box"])
        ha, hb = ta["last_box"][3] - ta["last_box"][1], tb["first_box"][3] - tb["first_box"][1]
        d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        if d < 0.6 * max(ha, hb) and 0.7 < hb / max(ha, 1) < 1.4 and (best is None or d < best[0]):
            best = (d, a)
    if best is not None:
        parent[root(b)] = root(best[1])

people: dict[int, dict] = {}
for tid, t in tracks.items():
    r = root(tid)
    p = people.setdefault(r, {"first_f": t["first_f"], "last_f": t["last_f"], "n": 0, "best_h": 0.0, "ids": []})
    p["ids"].append(tid)
    p["first_f"], p["last_f"], p["n"] = min(p["first_f"], t["first_f"]), max(p["last_f"], t["last_f"]), p["n"] + t["n"]
    if t["best_h"] > p["best_h"]:
        p["best_h"], p["best_f"], p["box"] = t["best_h"], t["best_f"], t["box"]
keep = {
    pid: p for pid, p in people.items()
    if p["n"] >= args.min_frames and p["best_h"] >= args.min_height * args.frame_h
}
by_frame: dict[int, list[int]] = defaultdict(list)
for tid, t in keep.items():
    by_frame[t["best_f"]].append(tid)

out = run / "tracks"
out.mkdir(exist_ok=True)
container = av.open(args.video)
for seq, frame in enumerate(container.decode(video=0), start=1):  # FileSource seq is 1-based
    if seq not in by_frame:
        continue
    img = frame.to_ndarray(format="bgr24")
    h, w = img.shape[:2]
    for tid in by_frame.pop(seq):
        x1, y1, x2, y2 = keep[tid]["box"]
        px, py = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        crop = img[max(int(y1 - py), 0):min(int(y2 + py), h), max(int(x1 - px), 0):min(int(x2 + px), w)]
        if crop.size:
            s = 384 / max(crop.shape[:2])
            if s < 1:
                crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"{tid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not by_frame:
        break

(run / "tracks.json").write_text(json.dumps({str(k): v for k, v in sorted(keep.items())}, indent=0))

# Contact sheets: 4 x 3 tiles of 240 x 320, each numbered with its track id.
sheets = run / "sheets"
sheets.mkdir(exist_ok=True)
ids = sorted(int(p.stem) for p in out.glob("*.jpg"))
index = {}
for n in range(0, len(ids), 12):
    sheet = np.full((3 * 340, 4 * 250, 3), 255, np.uint8)
    for k, tid in enumerate(ids[n:n + 12]):
        tile = cv2.imread(str(out / f"{tid}.jpg"))
        s = min(240 / tile.shape[1], 300 / tile.shape[0])
        tile = cv2.resize(tile, (max(1, int(tile.shape[1] * s)), max(1, int(tile.shape[0] * s))))
        r, c = divmod(k, 4)
        y0, x0 = r * 340 + 30, c * 250 + 5
        sheet[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]] = tile
        cv2.putText(sheet, f"#{tid}", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    name = f"sheet_{n // 12:02d}.jpg"
    cv2.imwrite(str(sheets / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    index[name] = ids[n:n + 12]
(sheets / "index.json").write_text(json.dumps(index))
print(json.dumps({"tracks_total": len(tracks), "people_after_stitching": len(people), "people_kept": len(keep),
                  "shots": len(ids), "sheets": len(index)}))
