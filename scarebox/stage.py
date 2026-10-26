"""stage: owns the audio-out queue and sinks. Spec §4.3.

`Stage(cfg, sink)` holds a small, non-blocking queue of `PlayItem`s and plays one at a
time. It is driven by repeated calls to `try_play_next()` (from a polling loop in
`app.py`, or directly from a test with an injected fake clock) rather than by sleeping
inside the class itself -- this keeps `cooldown_s` trivially testable without a real
clock: `try_play_next()` simply declines to play (returns `None`) until `cooldown_s` has
elapsed since the last play, judged by `clock()`.

- On `enqueue`, if the queue already holds `queue_max` pending items, the OLDEST pending
  item is dropped to make room (spec §4.3: "that runner has gone").
- Before playing, any item whose `t_trigger` is older than `max_line_age_s` is dropped.
- An optional short sting from `library/sfx/` plays before the line, if `cfg.stage.stings`.
- Every play logs `stage.play` with the wait time (now - t_trigger).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48_000


@dataclasses.dataclass
class PlayItem:
    # PCM int16 mono ndarray, a path to a .wav file, or (03-03) an async iterator of 48kHz
    # float32 chunks straight from `voice.Voice.speak()` -- the orchestrator enqueues the
    # live path as soon as the FIRST chunk exists (spec §4.4: "a slow tail doesn't miss the
    # deadline"), handing the iterator itself to stage so the rest of the line's synthesis
    # happens off the deadline clock, inside `try_play_next()`'s own await.
    event_id: str
    t_trigger: float
    audio: object
    sr: int = SAMPLE_RATE
    kind: str = "library"  # live | library | meta
    # How much a line is about THIS runner's costume (SCORE_* below). When several lines are
    # waiting, the speaker plays the best one rather than the oldest.
    score: int = 2

    def __post_init__(self) -> None:
        if self.kind == "meta":  # self-test, error lines, the TEST button: always play
            self.score = max(self.score, SCORE_META)


# Line priority (busy mode, Martin 2026-10-19: in a crowd the box was blurting generic and
# no-costume lines, late, instead of the striking costumes).
SCORE_META = 100     # self-test / error announcements: always play
SCORE_SPECIFIC = 3   # names this exact costume: a costume-pack line, or a live-written line
SCORE_COSTUME = 2    # a pre-recorded line for the costume's category (incl. groups)
SCORE_NO_COSTUME = 1 # plain finish-line goading for someone with no costume
SCORE_GENERIC = 0    # the judge couldn't tell ("mystery" lines)
SCORE_STANDOUT_BONUS = 2  # added for a costume the judge rated 3 (eye-catching)


def _load_pcm(audio: np.ndarray | str, sr: int) -> tuple[np.ndarray, int]:
    if isinstance(audio, np.ndarray):
        return audio, sr
    with wave.open(str(audio), "rb") as wf:
        file_sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(raw, dtype=np.int16)
        return pcm, file_sr


def _float_to_int16(audio: np.ndarray) -> np.ndarray:
    """48kHz float32 [-1, 1] (voice.Voice's output) -> int16 PCM (what every sink expects)."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


async def _drain_chunk_iterator(chunks) -> np.ndarray:
    """Collects every remaining chunk from a live `voice.Voice.speak()` async iterator into
    one int16 PCM array. Runs INSIDE `try_play_next()`, after the item is already queued and
    past the orchestrator's deadline -- a slow tail here never costs live share, only delays
    this one line's actual playback."""
    collected: list[np.ndarray] = []
    async for chunk in chunks:
        collected.append(np.asarray(chunk, dtype=np.float32))
    full = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
    return _float_to_int16(full)


class NullSink:
    """Discards audio. Used when there's no device (CI, or bench without `--wav`)."""

    def play(self, pcm: np.ndarray, sr: int) -> None:
        pass


class TimedSink:
    """Silent, but takes as long as the line would (2026-10-19). For video replays: the
    speaker being busy for each line's real length is half of what the timeline tuning
    measures, and WavSink/NullSink return instantly. `stage.sink: timed`."""

    def play(self, pcm: np.ndarray, sr: int) -> None:
        time.sleep(len(pcm) / float(sr or SAMPLE_RATE))


class WavSink:
    """Bench-only: appends every played item to its own timestamped WAV in `out_dir`."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def play(self, pcm: np.ndarray, sr: int) -> None:
        path = self.out_dir / f"stage-{time.time_ns()}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())


def alsa_pcm_name(device: str) -> str:
    """`stage.alsa_device` names a sound CARD ("scarebox_out", pinned by the udev rule in
    deploy/host/), but `aplay -D` wants a PCM. A bare card name fails with "audio open error:
    No such file or directory" (confirmed on gbox1 2026-10-19 -- every race line would have
    been silent). `plughw` also converts our mono 48 kHz into whatever the dongle takes.
    Anything that's already a PCM spec ("plughw:...", "hw:1,0", "default") passes through."""
    if ":" in device or device in ("default", "sysdefault", "null"):
        return device
    return f"plughw:CARD={device},DEV=0"


class AlsaSink:
    """Plays PCM via the `aplay` CLI (alsa-utils). A subprocess call is simpler and more
    portable than the `alsaaudio` CPython extension, which needs `libasound2-dev` at
    container-image build time."""

    def __init__(self, device: str) -> None:
        self.device = alsa_pcm_name(device)

    def play(self, pcm: np.ndarray, sr: int) -> None:
        subprocess.run(
            ["aplay", "-D", self.device, "-t", "raw", "-f", "S16_LE", "-r", str(sr), "-c", "1"],
            input=np.asarray(pcm, dtype=np.int16).tobytes(),
            check=True,
        )


class Stage:
    def __init__(self, cfg, sink, clock=time.monotonic, sfx_dir: str | Path = "library/sfx") -> None:
        self.cooldown_s: float = cfg.stage.cooldown_s
        self.queue_max: int = cfg.stage.queue_max
        self.stings: bool = cfg.stage.stings
        self.max_line_age_s: float = cfg.orchestrator.max_line_age_s
        self.sink = sink
        self._clock = clock
        self.sfx_dir = Path(sfx_dir)
        self._queue: list[PlayItem] = []
        self._last_play_at: float | None = None
        # Busy-mode thresholds (set from Tuning via tuning.apply_to_stage). With more than
        # one line waiting, lines below `crowd_min_score` are discarded; a lone line needs
        # `quiet_min_score`. The defaults here (0/0) keep the old play-everything behaviour
        # for code that never applies a Tuning (tests, bench).
        self.crowd_min_score: int = 0
        self.quiet_min_score: int = 0
        # Automatic crowd detection (Martin, 2026-10-19): busy mode switches on when at least
        # `crowd_crossings` people crossed the line in the last `crowd_window_s` seconds, not
        # only once lines have already piled up -- and off again when the crowd thins.
        # Speaker health (2026-10-19): a failed play used to raise out of try_play_next and
        # kill the app's stage loop for good -- silence until a restart. Now a failure is
        # recorded here (shown on the phone page) and the next line simply tries again.
        self.audio_ok: bool | None = None  # None until something has been played
        self.audio_error: str | None = None
        self.audio_failures: int = 0
        # Busy-mode gap: when crowded, this replaces cooldown_s (None = no change).
        self.busy_gap_s: float | None = None
        self.crowd_window_s: float = 10.0
        self.crowd_crossings: int = 3
        self._crossings: list[float] = []

    def note_crossing(self, t: float, people: int = 1) -> None:
        """Called by the orchestrator for every crossing, judged or skipped."""
        self._crossings.extend([t] * max(people, 1))

    def crowded(self, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        self._crossings = [t for t in self._crossings if now - t <= self.crowd_window_s]
        return len(self._queue) > 1 or len(self._crossings) >= self.crowd_crossings

    @property
    def queue(self) -> list[PlayItem]:
        return list(self._queue)

    # Optional `(kind, item, **fields)` callback (capture mode, 2026-10-19): told about every
    # line's fate -- "played" (with start/end times) or "dropped" (with the reason) -- so the
    # tuning simulator can replay what the speaker actually did over the video's timeline.
    on_outcome = None

    def _outcome(self, kind: str, item: PlayItem, **fields) -> None:
        if self.on_outcome is not None:
            try:
                self.on_outcome(kind, item, **fields)
            except Exception as exc:  # noqa: BLE001 -- logging must never break playback
                logger.warning("stage.outcome_hook_fail reason=%r", exc)

    def enqueue(self, item: PlayItem) -> None:
        if len(self._queue) >= self.queue_max:
            dropped = self._queue.pop(0)
            logger.info("stage.drop event=%s reason=queue_full", dropped.event_id)
            self._outcome("dropped", dropped, reason="queue_full")
        self._queue.append(item)

    def drop_stale(self, now: float | None = None) -> list[PlayItem]:
        now = self._clock() if now is None else now
        keep: list[PlayItem] = []
        dropped: list[PlayItem] = []
        for item in self._queue:
            if (now - item.t_trigger) > self.max_line_age_s:
                dropped.append(item)
            else:
                keep.append(item)
        self._queue = keep
        for item in dropped:
            logger.info("stage.drop event=%s reason=stale", item.event_id)
            self._outcome("dropped", item, reason="stale", age_s=round(now - item.t_trigger, 3))
        return dropped

    def _play_sting(self) -> None:
        if not self.sfx_dir.is_dir():
            return
        stings = sorted(self.sfx_dir.glob("*.wav"))
        if not stings:
            return
        try:
            pcm, sr = _load_pcm(str(stings[0]), SAMPLE_RATE)
            self.sink.play(pcm, sr)
        except Exception as exc:  # noqa: BLE001 -- a missing/corrupt sting never blocks the line
            logger.warning("stage.sting_fail reason=%s", exc)

    def _choose(self, now: float) -> PlayItem | None:
        """Takes the best waiting line out of the queue. In a crowd (`crowded()`: several
        lines waiting, or lots of recent crossings) the bar rises to `crowd_min_score`, and
        everything below it is dropped rather than left to play late. Best = highest score,
        then newest: the newest runner is still near the speaker, the oldest has probably
        gone past."""
        bar = self.crowd_min_score if self.crowded(now) else self.quiet_min_score
        keep = [i for i in self._queue if i.score >= bar]
        for item in self._queue:
            if item.score < bar:
                logger.info(
                    "stage.drop event=%s reason=low_priority score=%d bar=%d",
                    item.event_id, item.score, bar,
                )
                self._outcome("dropped", item, reason="low_priority", bar=bar)
        if not keep:
            self._queue = []
            return None
        best = max(keep, key=lambda i: (i.score, i.t_trigger))
        keep.remove(best)
        self._queue = keep
        return best

    async def try_play_next(self, now: float | None = None) -> PlayItem | None:
        now = self._clock() if now is None else now
        self.drop_stale(now)
        if not self._queue:
            return None
        gap = self.cooldown_s
        if self.busy_gap_s is not None and self.crowded(now):
            gap = min(gap, self.busy_gap_s)
        if self._last_play_at is not None and (now - self._last_play_at) < gap:
            return None

        item = self._choose(now)
        if item is None:
            return None
        if self.stings:
            self._play_sting()
        try:
            if hasattr(item.audio, "__anext__"):
                pcm = await _drain_chunk_iterator(item.audio)
                sr = SAMPLE_RATE  # voice.Voice always resamples to 48kHz before yielding
            else:
                pcm, sr = _load_pcm(item.audio, item.sr)
            # In a worker thread: `sink.play` blocks for the whole line (aplay runs until the
            # clip ends), and calling it directly froze the app's single event loop -- no
            # camera frames, tracking or verdicts -- for every ~4 s line (found 2026-10-19).
            await asyncio.get_running_loop().run_in_executor(None, self.sink.play, pcm, sr)
        except Exception as exc:  # noqa: BLE001 -- dongle unplugged, aplay error, bad file
            self.audio_ok = False
            self.audio_error = f"{type(exc).__name__}: {exc}"[:200]
            self.audio_failures += 1
            self._last_play_at = self._clock()  # keep the gap, so a dead speaker isn't hammered
            logger.error("stage.play_fail event=%s reason=%s", item.event_id, self.audio_error)
            self._outcome("dropped", item, reason="play_fail", error=self.audio_error)
            return None
        self.audio_ok = True
        self.audio_error = None
        # The gap runs from when the line FINISHES. `sink.play` blocks for the whole clip, so
        # `now` (taken before it) let a 1 s cooldown expire during a 4 s line, and the next
        # line followed with no gap at all (Martin, 2026-10-19).
        self._last_play_at = self._clock()
        logger.info(
            "stage.play event=%s kind=%s wait_s=%.3f", item.event_id, item.kind, now - item.t_trigger
        )
        self._outcome(
            "played", item, t_start=round(now, 4), t_end=round(self._last_play_at, 4),
            wait_s=round(now - item.t_trigger, 3),
        )
        return item
