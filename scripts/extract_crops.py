#!/usr/bin/env python3
"""Extract candidate person crops from scarebox test clips (gbox1 only).

This is a throwaway labelling tool, not the production detector (that's a later phase's
job). Uses torchvision's fasterrcnn_resnet50_fpn_v2 (BSD, COCO weights) for the person class.

Two stages, because gbox1's host venv has no CUDA torch (PyPI aarch64 torch is CPU-only) and
the NVIDIA PyTorch container has no ffmpeg/opencv:

  ~/.venv/bin/python scripts/extract_crops.py frames   # host venv: ffmpeg frame dump
  docker run ... IMAGE_TORCH ... python3 scripts/extract_crops.py detect   # container: torchvision

"frames" needs imageio-ffmpeg + Pillow. "detect" needs torch + torchvision + Pillow.
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = ROOT / "testdata" / "clips"
FRAMES_DIR = ROOT / "testdata" / "_frames"
CROPS_DIR = ROOT / "testdata" / "crops"
SAMPLE_EVERY_S = 2.0
MIN_PERSON_HEIGHT_FRAC = 0.25
PAD_FRAC = 0.10
SCORE_THRESH = 0.8


def stage_frames(_args):
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    if not clips:
        print("no clips found in testdata/clips", file=sys.stderr)
        sys.exit(1)
    for clip in clips:
        out_dir = FRAMES_DIR / clip.stem
        out_dir.mkdir(exist_ok=True)
        if any(out_dir.glob("*.jpg")):
            print(f"skip (frames exist): {clip.stem}", file=sys.stderr)
            continue
        fps = 1.0 / SAMPLE_EVERY_S
        cmd = [
            ffmpeg, "-y", "-i", str(clip),
            "-vf", f"fps={fps}",
            "-qscale:v", "3",
            str(out_dir / "%06d.jpg"),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        n = len(list(out_dir.glob("*.jpg")))
        if n == 0:
            print(f"WARNING: {clip.stem} produced 0 frames: {r.stderr[-500:]}", file=sys.stderr)
        else:
            print(f"{clip.stem}: {n} frames", file=sys.stderr)


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def pad_box(box, w, h, pad=PAD_FRAC):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return (
        max(0.0, x0 - bw * pad),
        max(0.0, y0 - bh * pad),
        min(float(w), x1 + bw * pad),
        min(float(h), y1 + bh * pad),
    )


def stage_detect(_args):
    import torch
    from PIL import Image
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", file=sys.stderr)
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=SCORE_THRESH)
    model = model.to(device).eval()
    preprocess = weights.transforms()
    person_label = weights.meta["categories"].index("person")

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    clip_dirs = sorted(d for d in FRAMES_DIR.iterdir() if d.is_dir())
    if not clip_dirs:
        print("no frame dirs found; run the 'frames' stage first", file=sys.stderr)
        sys.exit(1)

    candidates = []
    for clip_dir in clip_dirs:
        clip = clip_dir.name
        frame_files = sorted(clip_dir.glob("*.jpg"))
        for i, fp in enumerate(frame_files):
            t = i * SAMPLE_EVERY_S
            img = Image.open(fp).convert("RGB")
            w, h = img.size
            with torch.no_grad():
                out = model([preprocess(img).to(device)])[0]
            boxes = out["boxes"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            people = []
            for box, label, score in zip(boxes, labels, scores):
                if label != person_label or score < SCORE_THRESH:
                    continue
                x0, y0, x1, y1 = box
                if (y1 - y0) < MIN_PERSON_HEIGHT_FRAC * h:
                    continue
                people.append((float(x0), float(y0), float(x1), float(y1), float(score)))
            if people:
                candidates.append({"clip": clip, "t": t, "frame": str(fp), "w": w, "h": h, "people": people})
        print(f"{clip}: {len(frame_files)} frames scanned", file=sys.stderr)

    print(f"{len(candidates)} frames with qualifying people", file=sys.stderr)
    select_crops(candidates)


def select_crops(candidates, target=120, max_per_clip=25, group_target=15):
    from PIL import Image

    by_clip = defaultdict(list)
    for c in candidates:
        by_clip[c["clip"]].append(c)
    for v in by_clip.values():
        v.sort(key=lambda c: c["t"])

    clip_order = sorted(by_clip)
    cursors = {c: 0 for c in clip_order}
    per_clip_count = defaultdict(int)
    recents = defaultdict(list)  # clip -> [(t, box)]
    manifest = []
    n_single = 0
    n_group = 0

    progressed = True
    while progressed and (n_single < target or n_group < group_target):
        progressed = False
        for clip in clip_order:
            frames = by_clip[clip]
            idx = cursors[clip]
            if idx >= len(frames):
                continue
            cand = frames[idx]
            cursors[clip] = idx + 1
            progressed = True
            t = cand["t"]
            people = cand["people"]

            if len(people) >= 2 and n_group < group_target:
                xs0 = min(p[0] for p in people)
                ys0 = min(p[1] for p in people)
                xs1 = max(p[2] for p in people)
                ys1 = max(p[3] for p in people)
                box = pad_box((xs0, ys0, xs1, ys1), cand["w"], cand["h"])
                img = Image.open(cand["frame"]).convert("RGB")
                crop = img.crop(tuple(map(int, box)))
                name = f"{clip}_{int(t)}_group.jpg"
                crop.save(CROPS_DIR / name, quality=90)
                manifest.append({"file": name, "clip": clip, "t": t, "kind": "group", "n_people": len(people)})
                n_group += 1

            if per_clip_count[clip] >= max_per_clip or n_single >= target:
                continue
            for pidx, p in enumerate(people):
                if per_clip_count[clip] >= max_per_clip or n_single >= target:
                    break
                box = (p[0], p[1], p[2], p[3])
                if any(abs(t - rt) <= 4.0 and iou(box, rbox) > 0.5 for rt, rbox in recents[clip]):
                    continue
                pbox = pad_box(box, cand["w"], cand["h"])
                img = Image.open(cand["frame"]).convert("RGB")
                crop = img.crop(tuple(map(int, pbox)))
                name = f"{clip}_{int(t)}_{pidx}.jpg"
                crop.save(CROPS_DIR / name, quality=90)
                manifest.append({"file": name, "clip": clip, "t": t, "kind": "single", "score": p[4]})
                recents[clip].append((t, box))
                per_clip_count[clip] += 1
                n_single += 1

    with open(CROPS_DIR / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(
        f"single crops: {n_single}, group crops: {n_group}, per-clip: {dict(per_clip_count)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["frames", "detect"])
    ns = ap.parse_args()
    if ns.stage == "frames":
        stage_frames(ns)
    else:
        stage_detect(ns)
