"""app: `scarebox run` -- wires ingest -> perceive -> (judge -> library -> stage) | the full
live orchestrator (spec §4.4) -> stage, writing one JSON line per event to `runs/events.jsonl`.

`--mode live` (03-03, default): every event runs through `orchestrator.Orchestrator` --
TRIGGERED -> JUDGING -> QUIPPING -> CENSORING (inside quip) -> VOICING -> QUEUED -> PLAYED,
falling back to the vetted library on a deadline miss, an exception, or a censor reject.
`--mode library` (02-xx, kept): the original library-only subset (TRIGGERED -> JUDGING ->
(QUEUED -> PLAYED) | FALLBACK -> QUEUED) -- no quip/censor/voice, useful for a race-day
"library only" decision (the 2026-10-22 go/no-go) without touching the rest of the
stack, and for the `--tts-placeholder` bench runs that predate baked audio.

04-04 (aim control room): a mutable `tuning.Tuning` instance, loaded from `runs/tuning.json`
over `cfg`'s frozen defaults, is pushed into `perceiver`/`stage`/`orchestrator` at startup
and every time the aim page's `/settings` POST changes it (see `_set_settings`/
`_reset_settings` below). A digital crop (no PTZ on the USB webcam) is applied to every
incoming frame, in ONE place (`main()`'s frame loop, right before `perceiver.process_frame`)
-- detection/tracking/the trigger line/judge crops all then work in that same cropped
coordinate space. `EventRing` and `TrackStateMap` are the RAM-only (never written to disk)
state the aim page's live `/stream.mjpg` overlay and `/events` cards read from.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import functools
import json
import logging
import os
import random
import signal
import subprocess
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import cv2
import numpy as np

from scarebox import tuning as tuning_mod
from scarebox.aim import AimDeps, load_trigger_line, run_aim_server
from scarebox.bake import load_env_token
from scarebox.camera import Camera
from scarebox.censor import Censor
from scarebox.config import Config
from scarebox.events import Event
from scarebox.ingest import make_source, resolve_camera_host
from scarebox.judge import Judge
from scarebox.library import Library, resolve_category
from scarebox.orchestrator import Orchestrator
from scarebox.perceive import Perceiver
from scarebox.quip import Quip
from scarebox.selftest import run_selftest
from scarebox.capture import Capture
from scarebox.report_pdf import build_report_pdf
from scarebox.stage import AlsaSink, PlayItem, Stage, TimedSink, WavSink
from scarebox.voice import Voice

logger = logging.getLogger(__name__)

# Spec §4.4: "the deadline is t_trigger + budget_s"; this phase renders nothing itself
# (no voice yet), but keeps the same 300ms render margin the spec's deadline orchestrator
# uses, so judge is cut off with room for `library.pick` + `stage.enqueue` before the clock
# the real 03-03 orchestrator will enforce.
_RENDER_MARGIN_S = 0.3
_STAGE_POLL_S = 0.05
_HEARTBEAT_PATH = Path("runs/heartbeat")
_HEARTBEAT_INTERVAL_S = 5.0
_CAMERA_REANNOUNCE_S = 120.0  # 04-01: camera-down self-test failure re-announces every 2 min
_AIM_SNAPSHOT_MAX_WIDTH = 960  # 04-02: aim.html's snapshot, spec §2
_AIM_SNAPSHOT_JPEG_QUALITY = 70
_AIM_STREAM_MAX_WIDTH = 960  # 04-04: /stream.mjpg -- same cap, ~10 fps instead of 1
_AIM_STREAM_JPEG_QUALITY = 70
_AIM_EVENT_RING_MAXLEN = 60  # the page shows 12; the HCA report wants more (2026-10-19)
_DEBUG_BLOCK_MAINLOOP_PATH = Path("runs/debug_block_mainloop")  # 04-03 soak hang injection
_DEBUG_BLOCK_MAINLOOP_S = 35.0

# 04-04: per-track overlay colours (BGR, cv2 convention) for `/stream.mjpg`'s colour-coded
# boxes -- grey (tracked, no event yet), yellow (crossed the line / being judged), green
# (judged as a costume), red (judged no-costume).
_TRACK_STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "tracked": (150, 150, 150),
    "crossed": (0, 215, 255),
    "costume": (0, 200, 0),
    "no_costume": (0, 0, 220),
}
_DEFAULT_TRACK_STATE = "tracked"


def _placeholder_beep(sr: int = 48_000, dur_s: float = 0.5, freq: float = 880.0) -> np.ndarray:
    """A 0.5s sine beep -- `--tts-placeholder`'s stand-in for a rendered line, since the
    02-02 library manifest ships with every line `pending`/`wav: null` (02-03 bakes and
    vets the real audio)."""
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * freq * t)
    return (tone * 32767).astype(np.int16)


def _make_sink(cfg: Config):
    """04-01: the sink choice is an explicit config key now (`stage.sink`), not an implicit
    `/dev/snd`-exists probe. "wav" (bench/CI) always gets `WavSink`, no device touched at
    all. "alsa" (race default) always gets a REAL `AlsaSink` -- whether that device can
    actually be opened is `selftest.run_selftest`'s job (its `audio` check), and a failure
    there is reported LOUDLY by `main()` before it decides whether to swap sinks. This
    function itself never silently substitutes one sink for another: that silent fallback
    (02-02's bug -- `aplay` missing from the image meant a box that ran perfectly and never
    made a sound) is exactly what 04-01 exists to kill.
    """
    if cfg.stage.sink == "wav":
        return WavSink("runs/wav")
    if cfg.stage.sink == "timed":  # video replays: silent, but busy for each line's length
        return TimedSink()
    if cfg.stage.sink != "alsa":
        raise ValueError(f"unknown stage.sink: {cfg.stage.sink!r}")
    return AlsaSink(cfg.stage.alsa_device)


def _apply_volume(pct: float, cfg: Config) -> dict:
    """`amixer -c <card> sset Speaker <n>%` (04-04 Settings panel) -- only when
    `cfg.stage.alsa_device` is a bare card name (`stage.alsa_pcm_name`'s own distinction: a
    `plughw:...`/`hw:...`/`default` spec is already a specific PCM, not a mixer-controllable
    card, so volume control is skipped -- reported, never failed -- for those). Missing
    `amixer` (the `WavSink` bench container has no ALSA utils at all) is reported the same
    way: the task's own rule is "don't fail the request if amixer is missing, report it"."""
    device = cfg.stage.alsa_device
    if ":" in device or device in ("default", "sysdefault", "null"):
        return {"ok": False, "reason": "alsa_device is not a bare card name, volume skipped"}
    try:
        subprocess.run(
            ["amixer", "-c", device, "sset", "Speaker", f"{int(round(pct))}%"],
            check=True,
            capture_output=True,
            timeout=3.0,
        )
        return {"ok": True}
    except FileNotFoundError:
        return {"ok": False, "reason": "amixer not installed"}
    except Exception as exc:  # noqa: BLE001 -- a bad mixer control must never fail the request
        logger.warning("app.volume_apply_fail reason=%s", exc)
        return {"ok": False, "reason": str(exc)}


class TrackStateMap:
    """track_id -> (state, label) for the aim page's colour-coded `/stream.mjpg` overlay
    (04-04). Read/pruned from the HTTP stream-serving thread(s) (`ThreadingHTTPServer` gives
    `/stream.mjpg` its own worker thread per connection) and written from the asyncio main
    loop's thread -- genuinely concurrent, so every access goes through `self._lock`.

    "dropped when the track disappears" (the spec's own words) is `prune()`: called once per
    rendered stream frame with that frame's CURRENTLY tracked ids, so a track_id's entry
    outlives the track by at most one frame interval (~100ms) after it leaves the scene.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[int, tuple[str, str | None]] = {}

    def set(self, track_id: int, state: str, label: str | None = None) -> None:
        with self._lock:
            self._states[track_id] = (state, label)

    def get(self, track_id: int) -> tuple[str, str | None]:
        with self._lock:
            return self._states.get(track_id, (_DEFAULT_TRACK_STATE, None))

    def prune(self, live_ids) -> None:
        live = set(live_ids)
        with self._lock:
            for tid in list(self._states):
                if tid not in live:
                    del self._states[tid]


class EventRing:
    """RAM-only ring buffer of the last `maxlen` events (04-04) -- the aim page's event
    cards. Holds the EXACT crop JPEG the judge saw; there is no file-writing path anywhere
    in this class, by construction (hard privacy rule, "no recording") -- it is a `deque` of
    plain dicts, nothing more.

    `open()` is called synchronously the instant `perceiver.process_frame` returns a new
    Event (before judging even starts) so the card exists immediately with the crop + the
    track_ids that triggered it; `resolve()` is called once judging/quip/voice/library has
    actually finished, filling in the rest. A card that never resolves (the process died
    mid-judge) just stays half-filled -- `/events` still returns it, `resolved: false`.
    """

    def __init__(self, maxlen: int = _AIM_EVENT_RING_MAXLEN) -> None:
        self._lock = threading.Lock()
        self._buf: collections.deque = collections.deque(maxlen=maxlen)
        self._by_id: dict[str, dict] = {}

    def open(
        self,
        event_id: str,
        t_trigger: float,
        track_ids: list[int],
        crop_jpeg: bytes | None,
        wide_jpeg: bytes | None = None,
        frame_jpeg: bytes | None = None,
    ) -> None:
        card = {
            "id": event_id,
            "t_trigger": t_trigger,
            "wall_time": time.time(),  # for the report's timestamps (t_trigger is monotonic)
            "track_ids": list(track_ids),
            "crop": crop_jpeg,  # stripped out by `cards()` -- never leaves this process as JSON
            # Report images (2026-10-19, Martin: a PDF for HCA "without cropping the people so
            # much"): a wider shot around the person, and the whole frame with their box drawn.
            "wide": wide_jpeg,
            "frame": frame_jpeg,
            "score": None,
            "standout": None,
            "confidence": None,
            "category": None,
            "costume": None,
            "is_costume": None,
            "path": None,
            "pack": None,
            # 04-04 (Martin, costume clarity ask): the costume-PACK match for the judged
            # costume string itself (`library.match_pack`), independent of whether a pack
            # LINE actually got spoken -- `pack` above only reflects the latter.
            "pack_match": None,
            "pack_match_display": None,
            "line": None,
            "blocked_text": None,
            "blocked_reason": None,
            "ms": {},
            "resolved": False,
        }
        with self._lock:
            self._buf.append(card)
            self._by_id[event_id] = card
            live = {c["id"] for c in self._buf}
            for eid in list(self._by_id):
                if eid not in live:
                    del self._by_id[eid]

    def resolve(self, event_id: str, **fields) -> None:
        with self._lock:
            card = self._by_id.get(event_id)
            if card is None:
                return
            card.update(fields)
            card["resolved"] = True

    def discard(self, event_id: str) -> None:
        """Busy mode skips crossings without judging them; a card for each would push the
        interesting events off the phone in a crowd, so they're removed instead."""
        with self._lock:
            card = self._by_id.pop(event_id, None)
            if card is not None:
                self._buf.remove(card)

    _IMAGE_KEYS = ("crop", "wide", "frame")

    def cards(self) -> list[dict]:
        with self._lock:
            return [{k: v for k, v in c.items() if k not in self._IMAGE_KEYS} for c in self._buf]

    def crop(self, event_id: str) -> bytes | None:
        return self.image(event_id, "crop")

    def image(self, event_id: str, kind: str) -> bytes | None:
        if kind not in self._IMAGE_KEYS:
            return None
        with self._lock:
            card = self._by_id.get(event_id)
            return card.get(kind) if card else None


def _report_shots(
    image: np.ndarray, boxes: list, widen: float = 1.8, wide_max_px: int = 720, frame_max_px: int = 640
) -> tuple[bytes | None, bytes | None]:
    """The report's two pictures of an event, from the frame it fired on: a wide shot (the
    people's boxes, widened 1.8x around their centre and clipped to the frame) and the whole
    frame with their boxes drawn. RAM only, like the judge crop. (None, None) if the tracker
    had no box for them this frame."""
    if not boxes:
        return None, None
    h, w = image.shape[:2]
    x1 = min(float(b[0]) for b in boxes)
    y1 = min(float(b[1]) for b in boxes)
    x2 = max(float(b[2]) for b in boxes)
    y2 = max(float(b[3]) for b in boxes)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * widen / 2, (y2 - y1) * widen / 2
    wx1, wy1 = max(int(cx - half_w), 0), max(int(cy - half_h), 0)
    wx2, wy2 = min(int(cx + half_w), w), min(int(cy + half_h), h)
    marked = image.copy()
    for b in boxes:
        cv2.rectangle(marked, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 215, 255), 3)

    def _fit(img: np.ndarray, max_px: int) -> np.ndarray:
        ih, iw = img.shape[:2]
        scale = min(1.0, max_px / max(ih, iw))
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(iw * scale)), max(1, int(ih * scale))), interpolation=cv2.INTER_AREA)
        return img

    wide = marked[wy1:wy2, wx1:wx2]
    wide_jpeg = _encode_jpeg(_fit(wide, wide_max_px), 85) if wide.size else None
    frame_jpeg = _encode_jpeg(_fit(marked, frame_max_px), 80)
    return wide_jpeg, frame_jpeg


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return bytes(buf) if ok else None


def _resize_to_max_width(image: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _make_snapshot_jpeg(
    perceiver: Perceiver,
    last_frame_image: list,
    tuning: tuning_mod.Tuning,
    max_width: int = _AIM_SNAPSHOT_MAX_WIDTH,
    quality: int = _AIM_SNAPSHOT_JPEG_QUALITY,
) -> Callable[[], bytes | None]:
    """04-02: the aim page's `/snapshot.jpg` -- `perceive.debug_draw` overlays (trigger line,
    boxes, track IDs) on the latest raw frame, resized to `max_width`, JPEG-encoded straight
    to bytes. Generated on demand, in memory, every call -- never written to disk (spec §2;
    `cv2.imencode` has no file handle involved at all). 04-04: the frame is cropped to
    `tuning.crop` FIRST (same crop `main()`'s loop feeds `perceiver`), since the trigger line
    and every box `debug_draw` renders are in that cropped coordinate space."""

    def _snapshot() -> bytes | None:
        image = last_frame_image[0]
        if image is None:
            return None
        cropped = tuning_mod.apply_crop_to_frame(image, tuning.crop)
        overlay = perceiver.debug_draw(cropped)
        overlay = _resize_to_max_width(overlay, max_width)
        jpeg = _encode_jpeg(overlay, quality)
        if jpeg is None:
            logger.error("app.aim_snapshot_encode_fail")
        return jpeg

    return _snapshot


def _make_stream_frame_jpeg(
    perceiver: Perceiver,
    last_frame_image: list,
    track_states: TrackStateMap,
    tuning: tuning_mod.Tuning,
    max_width: int = _AIM_STREAM_MAX_WIDTH,
    quality: int = _AIM_STREAM_JPEG_QUALITY,
) -> Callable[[], bytes | None]:
    """04-04: the normal (cropped) `/stream.mjpg` view -- trigger line + one colour-coded box
    per currently-tracked id (grey/yellow/green/red per `track_states`, label only on a
    green/"costume" box: `"<costume> · <category>"`). ~10 fps, resized to `max_width`."""

    def _frame() -> bytes | None:
        image = last_frame_image[0]
        if image is None:
            return None
        overlay = tuning_mod.apply_crop_to_frame(image, tuning.crop).copy()
        h, w = overlay.shape[:2]
        (lx0, ly0), (lx1, ly1) = perceiver.trigger_line
        cv2.line(overlay, (int(lx0 * w), int(ly0 * h)), (int(lx1 * w), int(ly1 * h)), (0, 0, 255), 2)

        boxes = perceiver.current_tracked_boxes()
        track_states.prune(boxes.keys())
        for track_id, bbox in boxes.items():
            state, label = track_states.get(track_id)
            color = _TRACK_STATE_COLORS.get(state, _TRACK_STATE_COLORS[_DEFAULT_TRACK_STATE])
            x1b, y1b, x2b, y2b = (int(v) for v in bbox)
            cv2.rectangle(overlay, (x1b, y1b), (x2b, y2b), color, 2)
            if label:
                cv2.putText(
                    overlay, label, (x1b, max(0, y1b - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                )

        overlay = _resize_to_max_width(overlay, max_width)
        return _encode_jpeg(overlay, quality)

    return _frame


def _make_stream_frame_full_jpeg(
    last_frame_image: list,
    tuning: tuning_mod.Tuning,
    max_width: int = _AIM_STREAM_MAX_WIDTH,
    quality: int = _AIM_STREAM_JPEG_QUALITY,
) -> Callable[[], bytes | None]:
    """04-04: `/stream.mjpg?view=full` -- the Crop-mode editor's view. The UNCROPPED frame
    with the current crop rectangle drawn on it (no trigger line/boxes -- those live in the
    cropped frame's coordinate space, not this one), so dragging a new rectangle can see
    what's currently being cut off."""

    def _frame() -> bytes | None:
        image = last_frame_image[0]
        if image is None:
            return None
        overlay = image.copy()
        h, w = overlay.shape[:2]
        x0, y0, x1, y1 = tuning.crop
        cv2.rectangle(overlay, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (0, 215, 255), 2)
        overlay = _resize_to_max_width(overlay, max_width)
        return _encode_jpeg(overlay, quality)

    return _frame


def _mem_available_mb() -> float | None:
    """`MemAvailable` from `/proc/meminfo` (Linux-only -- this always runs in a Linux
    container) -- no `psutil` dependency in the image for one aim-page status field."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return round(kb / 1024.0, 1)
    except OSError as exc:
        logger.warning("app.mem_available_fail reason=%r", exc)
    return None


async def _heartbeat_writer(stop: asyncio.Event, path: Path = _HEARTBEAT_PATH) -> None:
    """Touches `path` every `_HEARTBEAT_INTERVAL_S` -- the app healthcheck (deploy/compose.yaml)
    checks this file's mtime freshness (<= 15s) since there's no HTTP port to probe here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        path.write_text(str(time.time()))
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=_HEARTBEAT_INTERVAL_S)


# A cold boot takes the vlm ~80 s to load on gbox1 (2026-10-23); 5 min matches the runbook's
# "nothing after 5 minutes" rule.
_BACKEND_WAIT_S = 300.0

def _backend_health_urls(cfg) -> list[str]:
    """The URLs that answer once each model server is ready: the judge and guard (vLLM's
    /v1/models) and, for the Chatterbox engine, the voice server's /health."""
    urls = [cfg.judge.endpoint.rstrip("/") + "/models", cfg.censor.guard_endpoint.rstrip("/") + "/models"]
    if cfg.voice.engine == "chatterbox":
        urls.append(cfg.voice.tts_url.rstrip("/") + "/health")
    return urls


async def _wait_for_backends(
    urls: list[str],
    timeout_s: float,
    keepalive,
    clock=time.monotonic,
    poll_s: float = 3.0,
    probe=None,
) -> bool:
    """Cold boot (2026-10-23 field test): Docker restarts every container at once and ignores
    compose's depends_on, so the app came up ~25 s before the judge and guard and its self-test
    failed, leaving it in library-only mode for the whole run. Wait here until every model
    server answers, or `timeout_s` passes (then the self-test reports the failure as before).
    `keepalive()` runs every poll so the frame watchdog doesn't treat a deliberate wait as a hang.
    Returns True when everything answered."""
    import httpx

    async def _http_ok(client, url: str) -> bool:
        try:
            return (await client.get(url, timeout=3.0)).status_code == 200
        except Exception:  # noqa: BLE001 -- connection refused while a server loads
            return False

    deadline = clock() + timeout_s
    waiting = list(urls)
    async with httpx.AsyncClient() as client:
        check = probe or (lambda url: _http_ok(client, url))
        while True:
            keepalive()
            waiting = [u for u in waiting if not await check(u)]
            if not waiting:
                logger.info("app.backends_ready")
                return True
            if clock() >= deadline:
                logger.error("app.backends_not_ready after_s=%.0f waiting=%s", timeout_s, waiting)
                return False
            logger.info("app.waiting_for_backends waiting=%s", waiting)
            await asyncio.sleep(poll_s)


async def _watchdog(
    stop: asyncio.Event,
    last_frame_at: list[float],
    last_tick_at: list[float],
    frame_timeout_s: float,
    tick_timeout_s: float,
    clock=time.monotonic,
    poll_s: float = 1.0,
    exit_fn=lambda code: os._exit(code),
) -> None:
    """Hub amendment (04-01): Docker's healthcheck marks a container `unhealthy` but NEVER
    restarts it on its own -- this is the thing that actually DOES something about a hang.
    If no camera frame has landed in `frame_timeout_s`, or the frame-consuming loop itself
    hasn't ticked in `tick_timeout_s` (e.g. stuck inside `perceiver.process_frame`, frames
    still arriving upstream but nothing downstream is moving), log loudly and hard-exit so
    the compose `restart: unless-stopped` policy brings a fresh container up.

    `os._exit` (not `sys.exit`/an exception) is deliberate: this process may be wedged
    somewhere that won't unwind cleanly, so skip every finalizer/cleanup path. `exit_fn` is
    injectable so tests can assert the trip fires without killing the test process; a fake
    clock (`clock=`) makes the 30s/20s timers trivially testable without a real sleep.
    """
    while not stop.is_set():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=poll_s)
        if stop.is_set():
            return
        now = clock()
        frame_age = now - last_frame_at[0]
        tick_age = now - last_tick_at[0]
        if frame_age > frame_timeout_s:
            logger.error("app.watchdog_trip reason=no_frame age_s=%.1f", frame_age)
            exit_fn(1)
            return
        if tick_age > tick_timeout_s:
            logger.error("app.watchdog_trip reason=no_tick age_s=%.1f", tick_age)
            exit_fn(1)
            return


async def _camera_retry_announcer(
    stop: asyncio.Event,
    library: Library,
    stage: Stage,
    is_camera_up: Callable[[], Awaitable[bool]],
    interval_s: float = _CAMERA_REANNOUNCE_S,
) -> None:
    """Spec: a camera-down self-test failure keeps retrying (ingest's own reconnect loop
    never gives up) and re-announces `meta-err-camera` every 2 minutes until frames are
    seen again. Runs only when the initial self-test actually failed on `camera`."""
    while not stop.is_set():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        if stop.is_set():
            return
        if await is_camera_up():
            logger.info("app.camera_recovered")
            return
        logger.error("app.camera_still_down reannouncing meta-err-camera")
        line = library.get("meta-err-camera")
        if line is not None:
            stage.enqueue(
                PlayItem(event_id="selftest-camera-retry", t_trigger=time.monotonic(), audio=line.wav, kind="meta")
            )


def _is_costume_judged(record: dict) -> bool:
    """04-04: green ("judged as a costume") vs red ("judged no-costume") for the aim page's
    track overlay/event card. `record["is_costume"]` (added to both event-record shapes
    alongside this feature) is the ground truth when present; the `bool(costume) and
    category != "no_costume"` fallback only fires for a record shape that predates it (there
    shouldn't be one left, but this never hard-crashes the overlay if there is)."""
    if "is_costume" in record:
        return bool(record["is_costume"])
    costume = record.get("costume") or ""
    category = record.get("category") or ""
    return bool(costume) and category != "no_costume"


def _costume_label(costume: str, category: str, is_costume: bool, pack_display: str | None) -> str:
    """The aim page's box label/event-card headline (Martin, 2026-10-19: "show clearly
    WHICH costume it thinks it saw"). A genuine no-costume verdict reads literally "no
    costume" (shown in red by `_TRACK_STATE_COLORS`), never "none · no_costume" -- the raw
    field values are still on the event card (`costume`/`category`/`is_costume`), this is
    just the headline. A pack match gets its human `display` name, flagged "(pack)" so it
    reads as a specific-costume hit, not just a category guess."""
    if not is_costume or not costume:
        return "no costume"
    if pack_display:
        return f"{pack_display} (pack) · {category}"
    return f"{costume} · {category}"


def _on_event_resolved(
    event: Event,
    event_ring: EventRing,
    track_states: TrackStateMap,
    library: Library,
    task: asyncio.Task,
) -> None:
    """`task.add_done_callback` target (one per event, bound via `functools.partial` so each
    callback captures its OWN `event`, not whatever `ev` the main loop's `for` variable holds
    by the time the callback actually fires). Updates the event card (04-04 bullet 2) and the
    live-view track colours/labels (bullet 1) from the SAME record `handle_event`/
    `_handle_event` already built -- no separate judge/censor call, no extra GPU work."""
    if task.cancelled() or task.exception() is not None:
        return
    record = task.result()
    if not isinstance(record, dict):
        return

    if record.get("path") == "skipped":
        # Never judged (busy mode): back to plain grey, not a red "no costume", and no card.
        for track_id in event.track_ids:
            track_states.set(track_id, _DEFAULT_TRACK_STATE, None)
        event_ring.discard(event.id)
        return

    costume = record.get("costume") or ""
    category = record.get("category") or ""
    is_costume = _is_costume_judged(record)
    # Independent of whether a pack LINE actually got picked (`record["pack"]` below) --
    # this is "does the judged costume STRING match a known pack", always worth showing.
    pack_match = library.match_pack(costume, group=category == "group") if (is_costume and costume) else None
    pack_match_display = library.pack_display(pack_match)
    label = _costume_label(costume, category, is_costume, pack_match_display)
    state = "costume" if is_costume else "no_costume"
    spot = record.get("spotlight") or {}
    for track_id in event.track_ids:
        if spot and track_id != spot.get("track"):
            # Spotlight: the line went to one member; the rest go back to plain grey.
            track_states.set(track_id, _DEFAULT_TRACK_STATE, None)
        else:
            track_states.set(track_id, state, label)

    path = record.get("path")
    pack = record.get("pack")
    # "live, pack (say which pack), library category line, or dropped" -- the task's own
    # four-way badge; `path` stays "live"/"library"/"dropped" in `events.jsonl` untouched,
    # this combined string is purely the aim page's display value.
    badge = f"pack:{pack}" if pack else path
    event_ring.resolve(
        event.id,
        category=category,
        costume=costume,
        is_costume=is_costume,
        pack_match=pack_match,
        pack_match_display=pack_match_display,
        path=badge,
        pack=pack,
        line=record.get("line"),
        blocked_text=record.get("blocked_text"),
        blocked_reason=record.get("blocked_reason"),
        ms=record.get("ms", {}),
        score=record.get("score"),
        standout=record.get("standout"),
        confidence=record.get("confidence"),
        group_size=record.get("group_size"),
        spotlight=record.get("spotlight"),
    )


async def _handle_event(
    event: Event,
    judge: Judge,
    library: Library,
    stage: Stage,
    cfg: Config,
    tts_placeholder: bool,
    events_log,
) -> dict:
    t0 = time.monotonic()
    deadline = event.t_trigger + cfg.orchestrator.budget_s - _RENDER_MARGIN_S
    timeout = max(0.0, deadline - time.monotonic())

    verdict = None
    try:
        verdict = await asyncio.wait_for(judge.judge(event), timeout=timeout)
    except Exception as exc:  # noqa: BLE001 -- timeout or any judge-call failure
        logger.warning("app.judge_fail event=%s reason=%s", event.id, exc)
    judge_ms = (time.monotonic() - t0) * 1000.0

    category = verdict.category if verdict else "mystery"
    costume = verdict.costume if verdict else ""
    confidence = verdict.confidence if verdict else 0.0
    # Hub amendment (02-02 Task 3): a group always plays a `group` line, regardless of what
    # the judge guessed -- see scarebox.library.resolve_category.
    category = resolve_category(category, event.group_size)

    line = library.pick(category, costume=costume if verdict and verdict.is_costume else "")
    path = "library" if line is not None else "dropped"
    pack = line.pack if line is not None else None

    if line is not None:
        audio = _placeholder_beep() if tts_placeholder else line.wav
        stage.enqueue(
            PlayItem(event_id=event.id, t_trigger=event.t_trigger, audio=audio, kind="library")
        )

    total_ms = (time.monotonic() - t0) * 1000.0
    record = {
        "id": event.id,
        "t_trigger": event.t_trigger,
        "group_size": event.group_size,
        "category": category,
        "costume": costume,
        "confidence": confidence,
        "is_costume": bool(verdict.is_costume) if verdict is not None else False,
        "path": path,
        "pack": pack,
        "line": line.text if line else None,
        "ms": {"judge": round(judge_ms, 1), "total": round(total_ms, 1)},
    }
    events_log.write(json.dumps(record) + "\n")
    events_log.flush()
    return record


async def _stage_runner(stage: Stage, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            played = await stage.try_play_next()
        except Exception as exc:  # noqa: BLE001 -- the speaker loop must never die (2026-10-19)
            logger.error("app.stage_runner_error reason=%r", exc)
            played = None
        if played is None:
            await asyncio.sleep(_STAGE_POLL_S)


async def _next_frame(source) -> object | None:
    """One frame from a FRESH `source.frames()` async generator -- used by the self-test's
    camera check and the retry announcer, both of which only need "is anything arriving
    right now", not a sustained subscription. Safe to call repeatedly: `RtspSource.frames()`
    starts (idempotently) the SAME background reader thread/buffer every call, so this never
    restarts a live camera connection; `FileSource.frames()` re-decodes from the top of the
    clip each call, which is fine for a liveness probe."""
    try:
        return await source.frames().__anext__()
    except StopAsyncIteration:
        return None


async def _is_camera_up(source, timeout_s: float = 3.0) -> bool:
    try:
        frame = await asyncio.wait_for(_next_frame(source), timeout=timeout_s)
    except Exception:  # noqa: BLE001
        return False
    return frame is not None


async def main(
    cfg: Config,
    mode: str = "live",
    tts_placeholder: bool = False,
    max_events: int | None = None,
) -> None:
    source = make_source(cfg)
    perceiver = Perceiver(cfg)
    # 04-02: a saved aim-page trigger line (from a previous session's `/trigger` POST)
    # takes priority over whatever `config/*.yaml` says -- that's the whole point of
    # letting Martin drag the line on his phone instead of editing YAML at the race.
    saved_trigger_line = load_trigger_line()
    if saved_trigger_line is not None:
        logger.info("app.trigger_line_loaded_from_disk line=%s", saved_trigger_line)
        perceiver.trigger_line = saved_trigger_line
    judge = Judge(cfg)
    # `--tts-placeholder` bench runs predate 02-03's rendered wavs (seed manifest lines all
    # have wav=None): require_wav=False lets pick() return a pending text-only line so there's
    # something to measure, while production (`tts_placeholder=False`) keeps the real
    # on-disk-wav gate.
    library = Library(
        cfg.library.manifest,
        cfg.library.require_approved,
        require_wav=not tts_placeholder,
        packs_path=cfg.library.packs,
    )
    sink = _make_sink(cfg)
    requested_alsa = cfg.stage.sink == "alsa"
    stage = Stage(cfg, sink)

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    # Capture mode (2026-10-19): a per-run tuning log over the video's timeline. Off at the
    # race (no recording); see scarebox/capture.py.
    capture: Capture | None = None
    if cfg.capture.enabled:
        capture = Capture(
            cfg.capture.dir,
            meta={"source": cfg.source.kind, "file": cfg.source.file, "trigger_line": cfg.perceive.trigger_line},
        )
        stage.on_outcome = lambda kind, item, **kw: capture.event(
            kind, id=item.event_id, score=item.score, line_kind=item.kind, t_trigger=item.t_trigger, **kw
        )

    # 04-04: `Tuning` -- config defaults overlaid by a previous session's `runs/tuning.json`
    # -- is the single mutable seam the aim page's Settings panel writes through. Pushed
    # into `perceiver`/`stage` immediately (the `Orchestrator` gets it at construction,
    # below, since it needs to exist first).
    tuning = tuning_mod.load_tuning(cfg, cfg.aim.tuning_file)
    tuning_mod.apply_to_perceiver(tuning, perceiver)
    tuning_mod.apply_to_stage(tuning, stage)
    volume_result = _apply_volume(tuning.volume_pct, cfg)
    if not volume_result.get("ok"):
        logger.info("app.volume_apply_skip reason=%s", volume_result.get("reason"))

    # 04-04: RAM-only event ring buffer + live-view track-colour map -- see both classes'
    # own docstrings for the "never written to disk" / thread-safety rules.
    event_ring = EventRing()
    track_states = TrackStateMap()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stop.set)

    now0 = time.monotonic()
    last_frame_at = [now0]
    last_tick_at = [now0]
    last_frame_image: list = [None]  # 04-02: latest raw BGR frame, for /snapshot.jpg
    selftest_result = None  # 04-02: filled below when mode == "live"; /status reports it

    # 04-02: a second `Camera` for PTZ only -- `make_source`'s own `Camera` (clock-sync) is
    # short-lived and never kept around. Only meaningful against the live camera (`rtsp`);
    # a `file`/`usb` source (bench, or the race-day webcam -- 04-04: no motor) has nothing to
    # nudge, so `_status`'s `ptz_available` and the aim page both gate the PTZ pad off it.
    ptz_available = cfg.source.kind == "rtsp"
    camera_for_ptz: Camera | None = None
    if ptz_available:
        try:
            ptz_host = resolve_camera_host(cfg)
            camera_for_ptz = Camera(ptz_host, cfg.camera.user, cfg.camera.password)
        except Exception as exc:  # noqa: BLE001 -- aim's PTZ is best-effort, never fatal to the run
            logger.error("app.aim_camera_unavailable reason=%r", exc)

    def _ptz_nudge(direction: str, speed: int, duration_s: float) -> None:
        if camera_for_ptz is None:
            raise RuntimeError(
                "no pan/tilt: the USB webcam is aimed by hand (and a file source has no camera), "
                "or the network camera wasn't found at startup"
            )
        camera_for_ptz.ptz_nudge(direction, speed, duration_s)

    def _test_line() -> str | None:
        categories = library.categories()
        if not categories:
            return None
        picked = library.pick(random.choice(categories))
        if picked is None:
            return None
        stage.enqueue(
            PlayItem(event_id="aim-test", t_trigger=time.monotonic(), audio=picked.wav, kind="meta")
        )
        return picked.text

    def _status() -> dict:
        return {
            "selftest": {
                "ok": selftest_result.ok if selftest_result is not None else None,
                "failed_check": selftest_result.failed_check if selftest_result is not None else None,
            },
            "audio_device": cfg.stage.alsa_device if requested_alsa else None,
            "sink": type(stage.sink).__name__,
            "source": cfg.source.kind,
            # Busy mode's automatic crowd meter (stage.crowded): shown as a BUSY chip.
            "crowded": bool(tuning.busy_mode and stage.crowded()),
            # Speaker health: False once a line has failed to play and none has played since.
            "speaker_ok": stage.audio_ok,
            "speaker_error": stage.audio_error,
            "speaker_failures": stage.audio_failures,
            "mem_available_mb": _mem_available_mb(),
            "uptime_s": round(time.monotonic() - now0, 1),
            # 04-04: the aim page hides the pan/tilt pad entirely when this is false (the
            # USB webcam -- race default -- has no motor).
            "ptz_available": ptz_available,
        }

    # -- 04-04 Settings panel -----------------------------------------------------------------

    def _settings_ranges() -> dict:
        ranges = {name: [r.lo, r.hi] for name, r in tuning_mod.RANGES.items()}
        ranges["crop"] = {"min_size": tuning_mod.MIN_CROP_SIZE}
        return ranges

    def _get_settings() -> dict:
        values = tuning.as_dict()
        # Trigger line isn't a `Tuning` field (it keeps its own `/trigger` drag mechanism --
        # see `aim.py`); reported here purely for display, same source `AimDeps.
        # get_trigger_line` already uses.
        values["trigger_line"] = [list(pt) for pt in perceiver.trigger_line]
        return {"values": values, "defaults": tuning_mod.defaults_dict(cfg), "ranges": _settings_ranges()}

    def _apply_tuning_everywhere() -> None:
        tuning_mod.apply_to_stage(tuning, stage)
        tuning_mod.apply_to_perceiver(tuning, perceiver)
        if orchestrator is not None:
            tuning_mod.apply_to_orchestrator(tuning, orchestrator)

    def _set_settings(patch: dict) -> tuple[bool, str | None]:
        ok, error = tuning_mod.validate_and_update(tuning, patch)
        if not ok:
            return ok, error
        _apply_tuning_everywhere()
        if "volume_pct" in patch:
            result = _apply_volume(tuning.volume_pct, cfg)
            if not result.get("ok"):
                logger.info("app.volume_apply_skip reason=%s", result.get("reason"))
        tuning_mod.save_tuning(tuning, cfg.aim.tuning_file)
        return True, None

    def _reset_settings() -> dict:
        # Mutates the EXISTING `tuning` object's fields back to cfg defaults, in place --
        # never rebinds the name to a new object, since `orchestrator.tuning`/every closure
        # here captured THIS object, not the local variable, and a rebind would leave them
        # all reading a now-stale copy.
        defaults = tuning_mod.defaults_from_cfg(cfg)
        for f in dataclasses.fields(tuning):
            setattr(tuning, f.name, getattr(defaults, f.name))
        _apply_tuning_everywhere()
        _apply_volume(tuning.volume_pct, cfg)
        tuning_mod.save_tuning(tuning, cfg.aim.tuning_file)
        return _get_settings()

    aim_deps = AimDeps(
        token=load_env_token(),
        trusted_networks=tuple(cfg.aim.trusted_networks),
        snapshot_jpeg=_make_snapshot_jpeg(perceiver, last_frame_image, tuning),
        stream_frame_jpeg=_make_stream_frame_jpeg(perceiver, last_frame_image, track_states, tuning),
        stream_frame_full_jpeg=_make_stream_frame_full_jpeg(last_frame_image, tuning),
        ptz_nudge=_ptz_nudge,
        get_trigger_line=lambda: perceiver.trigger_line,
        set_trigger_line=lambda line: setattr(perceiver, "trigger_line", line),
        events=event_ring.cards,
        event_crop=event_ring.crop,
        event_image=event_ring.image,
        report_pdf=lambda n: build_report_pdf(event_ring.cards(), event_ring.image, max_events=n),
        test_line=_test_line,
        status=_status,
        get_settings=_get_settings,
        set_settings=_set_settings,
        reset_settings=_reset_settings,
    )
    aim_server = run_aim_server(aim_deps, cfg.aim.listen)
    aim_thread = threading.Thread(target=aim_server.serve_forever, daemon=True)
    aim_thread.start()
    logger.info("app.aim_listening addr=%s", cfg.aim.listen)

    stage_task = asyncio.create_task(_stage_runner(stage, stop))
    heartbeat_task = asyncio.create_task(_heartbeat_writer(stop))
    watchdog_task = asyncio.create_task(
        _watchdog(
            stop,
            last_frame_at,
            last_tick_at,
            cfg.watchdog.frame_timeout_s,
            cfg.watchdog.tick_timeout_s,
        )
    )
    background_tasks = [stage_task, heartbeat_task, watchdog_task]
    pending: set[asyncio.Task] = set()
    n_events = 0

    censor: Censor | None = None
    quip: Quip | None = None
    orchestrator: Orchestrator | None = None
    library_only = False
    if mode == "live":
        # Cold boot: wait until the model servers answer (see _wait_for_backends) BEFORE
        # building anything that calls them -- Voice() warms up with a real synthesis, which
        # crashed the app on every boot until Docker's restarts outlasted the voice server.
        def _keepalive() -> None:
            last_frame_at[0] = last_tick_at[0] = time.monotonic()

        await _wait_for_backends(_backend_health_urls(cfg), _BACKEND_WAIT_S, _keepalive)

        # 03-03 hub amendment: wire the real guard_timeout_s from config, not the censor
        # module's hardcoded default -- 03-01 measured guard p95 253ms with the VLM busy, so
        # a too-tight timeout here would reject nearly every live line as `guard-timeout`.
        censor = Censor(
            cfg.censor.blocklist,
            cfg.censor.guard_endpoint,
            cfg.censor.guard_model,
            guard_timeout_s=cfg.censor.guard_timeout_s,
        )
        quip = Quip(cfg, censor)
        voice = Voice(cfg)
        orchestrator = Orchestrator(judge, quip, voice, library, stage, cfg, tuning=tuning)
        # `_set_settings`/`_reset_settings` above call `_apply_tuning_everywhere()`, which
        # reads this `orchestrator` closure variable -- it now exists, so the wiring is
        # complete from here on (it was already correct at construction time too, via the
        # `tuning=tuning` kwarg just above).

        # 04-01: the self-test is the only status display there is -- no screen at the
        # race. Runs once at startup, before the first real frame is processed.
        result = await run_selftest(
            get_frame=lambda: _next_frame(source),
            judge=judge,
            censor=censor,
            voice=voice,
            sink=stage.sink,
            requested_alsa=requested_alsa,
        )
        selftest_result = result
        if result.failed_check == "audio":
            # Reported LOUDLY, then (and only then) swapped -- never a silent substitution.
            logger.error(
                "app.audio_device_unavailable device=%s falling back to WavSink; "
                "this is a FAILURE, not a quiet degrade",
                cfg.stage.alsa_device,
            )
            stage.sink = WavSink("runs/wav")
        library_only = result.library_only
        if library_only:
            logger.error("app.library_only_mode reason=%s", result.failed_check)

        meta_line = library.get(result.meta_id)
        if meta_line is not None:
            stage.enqueue(
                PlayItem(event_id="selftest", t_trigger=time.monotonic(), audio=meta_line.wav, kind="meta")
            )
        else:
            logger.error("app.selftest_meta_missing id=%s", result.meta_id)

        if result.failed_check == "camera":
            camera_task = asyncio.create_task(
                _camera_retry_announcer(stop, library, stage, lambda: _is_camera_up(source))
            )
            background_tasks.append(camera_task)

    use_orchestrator = orchestrator is not None and not library_only

    try:
        with (runs_dir / "events.jsonl").open("a") as events_log:
            async for frame in source.frames():
                if stop.is_set():
                    break
                last_frame_at[0] = time.monotonic()
                last_frame_image[0] = frame.image
                # 04-03: soak hang injection. `docker pause` freezes this loop AND the
                # watchdog task that's supposed to notice, so proving the self-exit watchdog
                # (`_watchdog`, cfg.watchdog.tick_timeout_s) actually fires under a real hang
                # needs a software one instead. One-shot: the trigger file is removed before
                # sleeping so a post-restart container never re-trips it.
                if cfg.watchdog.debug_hooks and _DEBUG_BLOCK_MAINLOOP_PATH.exists():
                    logger.error(
                        "app.debug_hook_block_mainloop sleeping_s=%.1f", _DEBUG_BLOCK_MAINLOOP_S
                    )
                    _DEBUG_BLOCK_MAINLOOP_PATH.unlink(missing_ok=True)
                    await asyncio.sleep(_DEBUG_BLOCK_MAINLOOP_S)
                # 04-04: the digital crop (no PTZ on the USB webcam) is applied HERE, in the
                # ONE place frames enter the pipeline -- detection/tracking/the trigger line/
                # judge crops downstream all then work in this same cropped coordinate space.
                cropped_image = tuning_mod.apply_crop_to_frame(frame.image, tuning.crop)
                events = perceiver.process_frame(cropped_image, frame.t_capture)
                last_tick_at[0] = time.monotonic()
                if capture is not None:
                    capture.frame(frame.seq, frame.t_capture, perceiver.current_tracked_boxes())
                for ev in events:
                    # 04-04: the event card exists (with its crop) and the track overlay
                    # flips to "crossed" (yellow) the INSTANT the event fires -- judging
                    # takes seconds, the phone shouldn't wait that long to see anything.
                    crop_jpeg = _encode_jpeg(ev.crops[0], _AIM_SNAPSHOT_JPEG_QUALITY) if ev.crops else None
                    live_boxes = perceiver.current_tracked_boxes()
                    ev_boxes = [live_boxes[t] for t in ev.track_ids if t in live_boxes]
                    wide_jpeg, frame_jpeg = _report_shots(cropped_image, ev_boxes)
                    event_ring.open(ev.id, ev.t_trigger, ev.track_ids, crop_jpeg, wide_jpeg, frame_jpeg)
                    if capture is not None:
                        capture.trigger(ev, frame.seq, live_boxes, crop_jpeg)
                    for track_id in ev.track_ids:
                        track_states.set(track_id, "crossed", None)

                    if use_orchestrator:
                        task = asyncio.create_task(orchestrator.handle_event(ev, events_log))
                    else:
                        task = asyncio.create_task(
                            _handle_event(ev, judge, library, stage, cfg, tts_placeholder, events_log)
                        )
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                    task.add_done_callback(
                        functools.partial(_on_event_resolved, ev, event_ring, track_states, library)
                    )
                    if capture is not None:
                        task.add_done_callback(
                            lambda t, _cap=capture: _cap.event("result", **t.result())
                            if not t.cancelled() and t.exception() is None and isinstance(t.result(), dict)
                            else None
                        )
                    n_events += 1
                if max_events is not None and n_events >= max_events:
                    break
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
    finally:
        stop.set()
        if capture is not None:
            capture.close()
        aim_server.shutdown()
        aim_thread.join(timeout=5)
        if camera_for_ptz is not None:
            camera_for_ptz.close()
        for task in background_tasks:
            task.cancel()
        for task in background_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await judge.aclose()
        if quip is not None:
            await quip.aclose()
        if censor is not None:
            await censor.aclose()
