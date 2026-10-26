#!/usr/bin/env python3
"""Measure camera lag: trigger a known motion event (a PTZ nudge) and time
how long it takes a decoded RTSP frame to show the change.

Runs on gbox1 against CAMERA_HOST/USER/PASSWORD from .env, over the
sub-stream RTSP URL (never the main stream). Method: "ptz-onset" -- includes
the camera's own motor start lag (about 50-150 ms), so the reported numbers
are an upper bound on pure camera-glass-to-decoded-frame lag, not a clean
isolate of it.

Usage:
    scripts/camera_latency.py [--trials 20] [--out runs/evidence/camera-latency.json]
    scripts/camera_latency.py --diagnose   # try subtype=0 only, to tell RTSP-open
                                            # failures from sub-stream-specific ones;
                                            # never changes any camera config.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scarebox.camera import Camera  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "runs" / "evidence" / "camera-latency.json"

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)


def _load_env() -> dict[str, str]:
    env_path = REPO_ROOT / ".env"
    env: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


class LatestFrameReader:
    """Keeps only the newest decoded frame; a slow consumer never builds lag."""

    def __init__(self, url: str) -> None:
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open RTSP stream: {url}")
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._t_recv: float = 0.0
        self._frame_count = 0
        self._t_start = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            t_recv = time.monotonic()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._t_recv = t_recv
                self._frame_count += 1

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self._frame, self._t_recv

    def fps_observed(self) -> float:
        elapsed = time.monotonic() - self._t_start
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 255.0
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def diagnose(env: dict[str, str]) -> int:
    cam = Camera(env["CAMERA_HOST"], env["CAMERA_USER"], env["CAMERA_PASSWORD"], timeout=5.0)
    for subtype, label in ((1, "sub-stream (subtype=1)"), (0, "main stream (subtype=0, diagnostic only)")):
        url = cam.rtsp_url(subtype=subtype)
        print(f"trying {label}...")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        ok = cap.isOpened()
        frame_ok = False
        if ok:
            frame_ok, _ = cap.read()
        cap.release()
        print(f"  opened={ok} first_frame_read={frame_ok}")
    print("Diagnostic only: no camera config was changed.")
    return 0


def measure(env: dict[str, str], trials: int, out_path: Path) -> int:
    cam = Camera(env["CAMERA_HOST"], env["CAMERA_USER"], env["CAMERA_PASSWORD"], timeout=5.0)
    url = cam.rtsp_url(subtype=1)  # sub-stream only

    start_pos = cam.ptz_position()  # raises rather than silently recording (0, 0, 0)
    print(f"start PTZ position (pan, tilt, zoom): {start_pos}")

    reader = LatestFrameReader(url)
    onset_delays_ms: list[float] = []
    directions = ["Left", "Right"]

    try:
        time.sleep(1.0)  # let the reader thread fill its first frame
        for i in range(trials):
            time.sleep(2.0)  # still period
            baseline, _ = reader.latest()
            if baseline is None:
                print(f"trial {i}: no frame available, skipping", file=sys.stderr)
                continue

            # Still-noise level: how much a static scene differs frame to frame.
            noise_samples = []
            for _ in range(5):
                time.sleep(0.05)
                frame, _ = reader.latest()
                if frame is not None:
                    noise_samples.append(_mean_abs_diff(baseline, frame))
            still_noise = max(statistics.median(noise_samples), 0.5) if noise_samples else 0.5
            threshold = 3.0 * still_noise

            direction = directions[i % 2]
            t_send = time.monotonic()
            send_thread = threading.Thread(
                target=cam.ptz_nudge, kwargs={"direction": direction, "speed": 4, "duration_s": 0.3}
            )
            send_thread.start()

            onset_ms: float | None = None
            deadline = t_send + 3.0
            while time.monotonic() < deadline:
                frame, t_recv = reader.latest()
                if frame is not None and _mean_abs_diff(baseline, frame) > threshold:
                    onset_ms = (t_recv - t_send) * 1000.0
                    break
                time.sleep(0.01)
            send_thread.join(timeout=2.0)

            if onset_ms is None:
                print(f"trial {i}: no onset detected within 3s, skipping", file=sys.stderr)
                continue
            onset_delays_ms.append(onset_ms)
            print(f"trial {i} ({direction}): onset {onset_ms:.0f} ms")
    finally:
        fps_observed = reader.fps_observed()
        reader.close()
        print("returning to start PTZ position...")
        try:
            cam.ptz_goto_abs(pan=start_pos[0], tilt=start_pos[1], zoom=start_pos[2])
        except Exception as exc:  # pragma: no cover - best-effort restore
            print(f"WARNING: could not restore PTZ position: {exc}", file=sys.stderr)

    if len(onset_delays_ms) < 15:
        print(
            f"ERROR: only {len(onset_delays_ms)} usable trials (need >= 15)",
            file=sys.stderr,
        )
        return 1

    onset_delays_ms.sort()
    result = {
        "trials": len(onset_delays_ms),
        "median_ms": statistics.median(onset_delays_ms),
        "p95_ms": onset_delays_ms[min(int(len(onset_delays_ms) * 0.95), len(onset_delays_ms) - 1)],
        "max_ms": max(onset_delays_ms),
        "fps_observed": round(fps_observed, 1),
        "method": "ptz-onset (includes motor start ~50-150ms, so it's an upper bound)",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()

    env = _load_env()
    if args.diagnose:
        return diagnose(env)
    return measure(env, args.trials, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
