"""selftest: the startup diagnostic. Spec §4.4/§5.2 -- "there's no screen at the race, so
the speaker is the only status display."

`run_selftest()` runs five liveness checks IN ORDER: (a) camera frames arriving (<= 20s),
(b) the judge VLM server answers a canned (synthetic) crop, (c) the censor passes a
known-good line (both the blocklist AND the guard server), (d) voice (Kokoro) renders
something, (e) the configured audio device actually opens. Every check still runs even
after an earlier one fails -- the console/log gets the full picture -- but only the FIRST
failed check decides which single meta line plays (one line, not a list read aloud).

All pass: `meta-ready`. Any failure: `meta-err-<thing>`, naming the first broken check.
Diagnostic lines are pre-rendered WAVs from `library/manifest.json`'s `meta` category, so
they still play even when the thing that's broken is TTS itself (`voice`).

`app.py` reads `SelfTestResult.library_only` to decide whether the rest of THIS run skips
the live orchestrator (brain/censor/voice are all live-path dependencies -- a camera miss
is different: the camera's own reconnect loop keeps trying, so `app.py` just keeps
re-announcing on a timer instead of giving up on live for good).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Awaitable, Callable

import numpy as np

from scarebox.censor import Censor, Pass
from scarebox.events import Event
from scarebox.judge import Judge
from scarebox.stage import AlsaSink
from scarebox.voice import Voice

logger = logging.getLogger(__name__)

CAMERA_TIMEOUT_S = 20.0
# Spec §7-clean, persona-appropriate, used ONLY as a liveness probe for the censor's two
# layers (blocklist + guard model) -- content accuracy isn't the point here.
KNOWN_GOOD_LINE = "Fantastic costume, keep up the great work out there!"

_META_IDS = {
    "camera": "meta-err-camera",
    "brain": "meta-err-brain",
    "censor": "meta-err-censor",
    "voice": "meta-err-voice",
    "audio": "meta-err-audio",
}

GetFrame = Callable[[], Awaitable[object]]


@dataclasses.dataclass
class SelfTestResult:
    ok: bool
    failed_check: str | None  # "camera" | "brain" | "censor" | "voice" | "audio" | None
    meta_id: str

    @property
    def library_only(self) -> bool:
        """brain (judge), censor and voice are all live-path dependencies: any one of them
        failing means the whole run falls back to library-only mode, announced (spec). A
        camera miss does NOT demote to library-only -- there's simply nothing to judge yet,
        and ingest's own reconnect loop keeps trying underneath."""
        return self.failed_check in ("brain", "censor", "voice")


def canned_crop() -> np.ndarray:
    """A synthetic 128x128 BGR crop -- enough for the judge's VLM server to answer
    SOMETHING. This is a liveness/connectivity check (the server is up and answers JSON),
    not an accuracy check (that's the Phase 1 bake-off's job)."""
    return np.full((128, 128, 3), 127, dtype=np.uint8)


async def _check_camera(get_frame: GetFrame, timeout_s: float = CAMERA_TIMEOUT_S) -> bool:
    try:
        frame = await asyncio.wait_for(get_frame(), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 -- timeout, decode failure, anything: no camera
        logger.error("selftest.camera_fail reason=%r", exc)
        return False
    if frame is None:
        logger.error("selftest.camera_fail reason=no_frame")
        return False
    return True


async def _check_judge(judge: Judge) -> bool:
    event = Event(crops=[canned_crop()], bbox_heights=[1.0])
    try:
        verdict = await judge.judge(event)
    except Exception as exc:  # noqa: BLE001
        logger.error("selftest.judge_fail reason=%r", exc)
        return False
    if verdict is None:
        logger.error("selftest.judge_fail reason=no_verdict")
        return False
    return True


async def _check_censor(censor: Censor) -> bool:
    try:
        result = await censor.check(KNOWN_GOOD_LINE)
    except Exception as exc:  # noqa: BLE001
        logger.error("selftest.censor_fail reason=%r", exc)
        return False
    if not isinstance(result, Pass):
        logger.error("selftest.censor_fail reason=%r", result)
        return False
    return True


async def _check_voice(voice: Voice) -> bool:
    try:
        chunks = [chunk async for chunk in voice.speak(KNOWN_GOOD_LINE)]
    except Exception as exc:  # noqa: BLE001
        logger.error("selftest.voice_fail reason=%r", exc)
        return False
    if sum(np.asarray(c).size for c in chunks) == 0:
        logger.error("selftest.voice_fail reason=empty_audio")
        return False
    return True


async def _check_audio(sink: object, requested_alsa: bool) -> bool:
    """Only probed when `stage.sink: alsa` was actually requested -- `sink: wav` (bench)
    explicitly asked for no device at all, so there's nothing to fail (hub amendment:
    WavSink fallback is allowed ONLY when config says `stage.sink: wav` explicitly).
    Plays a near-silent 50ms buffer: the real "does `aplay -D <device>` work" test, not just
    a name/path check -- but quiet enough it isn't itself an audible glitch before the
    self-test has decided what the FIRST real sound should be."""
    if not requested_alsa or not isinstance(sink, AlsaSink):
        return True
    silence = np.zeros(int(0.05 * 48_000), dtype=np.int16)
    try:
        sink.play(silence, 48_000)
    except Exception as exc:  # noqa: BLE001
        logger.error("selftest.audio_fail device=%s reason=%r", sink.device, exc)
        return False
    return True


async def run_selftest(
    *,
    get_frame: GetFrame,
    judge: Judge,
    censor: Censor,
    voice: Voice,
    sink: object,
    requested_alsa: bool,
) -> SelfTestResult:
    checks: list[tuple[str, bool]] = [
        ("camera", await _check_camera(get_frame)),
        ("brain", await _check_judge(judge)),
        ("censor", await _check_censor(censor)),
        ("voice", await _check_voice(voice)),
        ("audio", await _check_audio(sink, requested_alsa)),
    ]

    failed = next((name for name, ok in checks if not ok), None)
    if failed is None:
        logger.info("selftest.pass")
        return SelfTestResult(ok=True, failed_check=None, meta_id="meta-ready")

    for name, ok in checks:
        if not ok:
            logger.error("selftest.check_failed check=%s", name)
    logger.error("selftest.fail reporting=%s", failed)
    return SelfTestResult(ok=False, failed_check=failed, meta_id=_META_IDS[failed])
