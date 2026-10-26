"""ingest: frame sources. Spec §4.3.

`Frame` is the shared unit the rest of the pipeline consumes. `RtspSource` decodes the
camera's live RTSP sub-stream on a daemon reader thread and keeps only the newest frame (a
single-slot overwrite buffer, `LatestFrameBuffer`) so a slow downstream consumer never builds
lag -- `perceive` always sees "now", never a backlog. `FileSource` replays an MP4
(testing/bench) at real-time pace by PTS, or as fast as decode allows with `realtime=False`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import av
import numpy as np

from scarebox.camera import Camera
from scarebox.config import Config
from scarebox.discover import find_camera

logger = logging.getLogger(__name__)

# How often frames() polls LatestFrameBuffer for a new seq. Cheap: the buffer read is a
# lock + attribute check, and 5ms is well under one frame period at 15fps (~67ms).
_POLL_S = 0.005


@dataclasses.dataclass
class Frame:
    image: np.ndarray  # BGR, HxWx3 uint8
    t_capture: float  # time.monotonic() when this frame was decoded
    seq: int


class LatestFrameBuffer:
    """Single-slot overwrite buffer: `put()` always replaces whatever was there, so a
    consumer polling `get_new(after_seq)` sees only the newest frame once it lands, never a
    backlog. This is the "keep only the newest frame" contract from spec §4.3, factored out
    of `RtspSource` so it can be exercised directly in tests without a live RTSP stream.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Frame | None = None

    def put(self, frame: Frame) -> None:
        with self._lock:
            self._frame = frame

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame

    async def get_new(self, after_seq: int, poll_s: float = _POLL_S) -> Frame:
        """Wait until a frame newer than `after_seq` is available, then return it."""
        while True:
            frame = self.latest()
            if frame is not None and frame.seq > after_seq:
                return frame
            await asyncio.sleep(poll_s)


class RtspSource:
    """Decodes `url` (PyAV, `rtsp_transport=tcp`/`fflags=nobuffer`/`flags=low_delay`) on a
    daemon reader thread into a `LatestFrameBuffer`. Reconnects on error with
    `reconnect_backoff` (capped at the last entry), logging `ingest.reconnect` each attempt.
    """

    def __init__(
        self,
        url: str,
        reconnect_backoff: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
    ) -> None:
        self.url = url
        self._reconnect_backoff = reconnect_backoff
        self._buffer = LatestFrameBuffer()
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        backoff_idx = 0
        while not self._stop.is_set():
            try:
                options = {
                    "rtsp_transport": "tcp",
                    "fflags": "nobuffer",
                    "flags": "low_delay",
                }
                with av.open(self.url, options=options, timeout=5.0) as container:
                    backoff_idx = 0  # a clean open resets the backoff
                    stream = container.streams.video[0]
                    for frame in container.decode(stream):
                        if self._stop.is_set():
                            return
                        image = frame.to_ndarray(format="bgr24")
                        self._seq += 1
                        self._buffer.put(
                            Frame(image=image, t_capture=time.monotonic(), seq=self._seq)
                        )
            except Exception as exc:  # noqa: BLE001 -- any decode/connection failure reconnects
                if self._stop.is_set():
                    return
                delay = self._reconnect_backoff[
                    min(backoff_idx, len(self._reconnect_backoff) - 1)
                ]
                backoff_idx += 1
                logger.warning("ingest.reconnect error=%r delay_s=%s", exc, delay)
                time.sleep(delay)

    async def frames(self) -> AsyncIterator[Frame]:
        self.start()
        last_seq = 0
        while True:
            frame = await self._buffer.get_new(last_seq)
            last_seq = frame.seq
            yield frame


class FileSource:
    """Replays `path` (an MP4). `realtime=True` paces yields by PTS so a bench run matches
    the camera's real frame rate; `realtime=False` yields as fast as decode allows.
    `loop=True` restarts from the beginning when the file ends; `False` stops the iterator.
    """

    def __init__(self, path: str | Path, realtime: bool = True, loop: bool = False) -> None:
        self.path = str(path)
        self.realtime = realtime
        self.loop = loop

    async def frames(self) -> AsyncIterator[Frame]:
        seq = 0
        while True:
            start_monotonic = time.monotonic()
            start_pts_s: float | None = None
            with av.open(self.path) as container:
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    image = frame.to_ndarray(format="bgr24")
                    if self.realtime:
                        if frame.pts is not None:
                            pts_s = float(frame.pts * stream.time_base)
                        else:
                            pts_s = 0.0
                        if start_pts_s is None:
                            start_pts_s = pts_s
                        target = start_monotonic + (pts_s - start_pts_s)
                        delay = target - time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
                    seq += 1
                    yield Frame(image=image, t_capture=time.monotonic(), seq=seq)
            if not self.loop:
                return


def find_usb_camera(name: str, dev_dir: str = "/dev", sys_dir: str = "/sys/class/video4linux") -> str | None:
    """Device path of the USB camera whose V4L2 name contains `name` ("" matches any), or
    None. A UVC camera exposes several nodes (video, metadata, sometimes an infrared sensor);
    the main picture is the node with the lowest `index`. Looked up by name every time rather
    than pinned to /dev/videoN, because the number changes when the camera is replugged
    (2026-10-18: the Dell WB7022 came back as video0 after being video2)."""
    best: tuple[int, str] | None = None
    for node in sorted(Path(sys_dir).glob("video*")):
        try:
            label = (node / "name").read_text().strip()
            index = int((node / "index").read_text().strip() or 0)
        except (OSError, ValueError):
            continue
        if name.lower() not in label.lower():
            continue
        if best is None or index < best[0]:
            best = (index, f"{dev_dir.rstrip('/')}/{node.name}")
    return best[1] if best else None


def _open_v4l2(path: str, width: int, height: int, fps: int):
    import cv2  # lazy: only the usb source needs OpenCV's V4L2 backend here

    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    # MJPG is the mode the Dell sustained for a clean 20-min, 30 fps soak (2026-10-19). Asking
    # a webcam for a mode it doesn't support can knock it off the USB bus, so keep this fixed.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest frame, not a queue of stale ones
    return cap


class UsbSource:
    """A USB (UVC) webcam read on a daemon thread into a `LatestFrameBuffer`, same contract as
    `RtspSource`. No network, no camera clock. If the camera vanishes (30 failed reads in a
    row) or isn't plugged in yet, it re-runs `find_usb_camera` with backoff, so a replug mid-
    race recovers on its own. `opener` is injectable for tests."""

    _MAX_FAILED_READS = 30

    def __init__(
        self,
        name: str,
        dev_dir: str = "/dev",
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        reconnect_backoff: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
        opener=_open_v4l2,
        finder=find_usb_camera,
    ) -> None:
        self.name, self.dev_dir = name, dev_dir
        self.width, self.height, self.fps = width, height, fps
        self._reconnect_backoff = reconnect_backoff
        self._opener, self._finder = opener, finder
        self._buffer = LatestFrameBuffer()
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _capture(self, path: str) -> None:
        """Reads frames from `path` until the camera stops answering (or stop())."""
        cap = self._opener(path, self.width, self.height, self.fps)
        try:
            if not cap.isOpened():
                raise OSError(f"cannot open {path}")
            failed = 0
            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok or image is None:
                    failed += 1
                    if failed >= self._MAX_FAILED_READS:
                        raise OSError(f"{path}: {failed} failed reads in a row")
                    time.sleep(0.02)
                    continue
                failed = 0
                self._seq += 1
                self._buffer.put(Frame(image=image, t_capture=time.monotonic(), seq=self._seq))
        finally:
            cap.release()

    def _run(self) -> None:
        backoff_idx = 0
        while not self._stop.is_set():
            seq_before = self._seq
            try:
                path = self._finder(self.name, self.dev_dir)
                if path is None:
                    raise OSError(f"no USB camera matching {self.name!r}")
                logger.info("ingest.usb_open path=%s", path)
                self._capture(path)
            except Exception as exc:  # noqa: BLE001 -- unplugged, wedged, anything: retry
                if self._stop.is_set():
                    return
                if self._seq > seq_before:
                    backoff_idx = 0  # it was streaming before it failed: retry quickly
                delay = self._reconnect_backoff[min(backoff_idx, len(self._reconnect_backoff) - 1)]
                backoff_idx += 1
                logger.warning("ingest.reconnect error=%r delay_s=%s", exc, delay)
                time.sleep(delay)

    async def frames(self) -> AsyncIterator[Frame]:
        self.start()
        last_seq = 0
        while True:
            frame = await self._buffer.get_new(last_seq)
            last_seq = frame.seq
            yield frame


def resolve_camera_host(cfg: Config) -> str:
    """`camera.host` if set, else `discover.find_camera(camera.mac)` (the home-to-race
    subnet changes with zero config, spec §4.3). Factored out of `make_source` (04-02) so
    `app.py`'s aim-page PTZ `Camera` resolves the SAME host as the RTSP stream itself,
    instead of duplicating (and risking drifting from) this lookup."""
    host = cfg.camera.host.strip()
    if host:
        return host
    if not cfg.camera.mac:
        raise ValueError("camera.host is empty and camera.mac is not set")
    return find_camera(cfg.camera.mac)


def make_source(cfg: Config) -> RtspSource | FileSource | UsbSource:
    """Picks a frame source by `cfg.source.kind` (`usb` | `rtsp` | `file`). `usb` (the race
    default since 2026-10-19, Martin's call) is a webcam on gbox1 itself. For `rtsp`, resolves the
    camera's IP via `resolve_camera_host`, then, after connecting, optionally pushes gbox1's
    clock to the camera (`camera.set_clock_on_start`): the camera has no RTC battery and
    resets to 2000-01-31 on power loss (spec §4.3)."""
    if cfg.source.kind == "usb":
        s = cfg.source
        return UsbSource(s.usb_name, dev_dir=s.usb_dev_dir, width=s.width, height=s.height, fps=s.fps)
    if cfg.source.kind == "rtsp":
        host = resolve_camera_host(cfg)
        if cfg.camera.set_clock_on_start:
            with Camera(host, cfg.camera.user, cfg.camera.password) as cam:
                cam.set_time(datetime.now())
        user = quote(cfg.camera.user, safe="")
        password = quote(cfg.camera.password, safe="")
        url = f"rtsp://{user}:{password}@{host}:554{cfg.camera.rtsp_path}"
        return RtspSource(url)
    if cfg.source.kind == "file":
        return FileSource(cfg.source.file, realtime=True, loop=cfg.source.loop)
    raise ValueError(f"unknown source.kind: {cfg.source.kind!r}")
