"""voice: live streaming TTS through the barker FX chain. Spec §4.3, §4.5.

`Voice(cfg)` loads the Kokoro engine once at startup (warms up with one throwaway line) and
exposes `async speak(text) -> AsyncIterator[np.ndarray]`, which yields 48kHz float32 mono
chunks run through a per-utterance `fx.BarkerFX`. Kokoro's pipeline isn't truly incremental
within one sentence (01-04 bake-off: "ttfa is actually full synthesis time"), so the GPU call
runs once in a thread executor (so the asyncio event loop isn't blocked) and the result is
handed out in fixed-size chunks.

`engine` is injectable (same DI pattern as `judge.Judge`/`quip.Quip`): a callable
`text -> (audio: np.ndarray, sample_rate: int)`. Tests pass a fake synthesiser; `kokoro`
itself is imported lazily inside `_load_kokoro()` so this module stays importable (for tests,
type-checking) on a host that never runs the real engine -- a CPU-only dev `.venv` has no GPU
and no `kokoro` package.

**UNVERIFIED (budget-limited this session, flagged per Confirm-Don't-Assert):** the exact
`kokoro.KPipeline` call shape in `_load_kokoro()` below is written from the `kokoro` package's
published usage pattern, NOT confirmed against a live run on gbox1 this session (no tool-call
budget left to build+bring-up the app image and exercise it against real GPU hardware). The
01-04 bake-off's own bench script (`scripts/bench_tts.py`) already calls the real API
successfully -- cross-check that script's exact call shape against this one before trusting
`_load_kokoro()` in production, and run `tests/test_voice.py`'s `--network none` container
smoke test (not yet run) before relying on this in the live pipeline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import logging
import re
from collections.abc import AsyncIterator, Callable, Iterator

import numpy as np
from scipy.signal import resample_poly

from scarebox.config import Config
from scarebox.fx import BarkerFX

logger = logging.getLogger(__name__)

_OUTPUT_SR = 48000
_CHUNK_MS = 20
_WARMUP_TEXT = "Ready."

# asyncio.Future explicitly refuses to carry a raw `StopIteration` across the executor
# boundary (`set_exception` raises `TypeError: StopIteration interacts badly with
# generators...`), which happens silently inside a `call_soon_threadsafe` callback and just
# hangs the awaiting coroutine forever -- found hanging a real test this session, not
# theoretical. `_next_or_sentinel` below translates generator exhaustion into a plain return
# value instead of letting `next()`'s StopIteration cross into asyncio land.
_SEGMENTS_DONE = object()


def _next_or_sentinel(it):
    try:
        return next(it)
    except StopIteration:
        return _SEGMENTS_DONE


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """'Nearly there, runner! Get a move on!' -> ['Nearly there, runner!', 'Get a move on!'].
    A lone one-word interjection ('Arrr!') is merged into the next sentence: a one-word request costs a full model call
    for half a second of audio, and Chatterbox tends to mangle single words."""
    parts = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    out: list[str] = []
    carry = ""
    for p in parts:
        p = f"{carry} {p}".strip() if carry else p
        if len(p.split()) < 2:
            carry = p
            continue
        out.append(p)
        carry = ""
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


Engine = Callable[[str], tuple[np.ndarray, int]]


def chatterbox_http_engine(tts_url: str, timeout_s: float) -> Engine:
    """Client for deploy/chatterbox/server.py (the scarebox-tts container). Contract:
    POST {tts_url}/tts with JSON {"text": ...} -> raw little-endian float32 mono PCM,
    sample rate in the X-Sample-Rate header. Any HTTP/timeout error propagates, so the
    orchestrator treats it like any other live-path failure and plays a library line.
    Shared by `Voice` (live) and `bake.render_manifest` (library), so both speak with the
    one cloned George."""
    import json
    import urllib.request

    url = tts_url.rstrip("/") + "/tts"

    def _synthesize(text: str) -> tuple[np.ndarray, int]:
        req = urllib.request.Request(
            url,
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            sr = int(resp.headers["X-Sample-Rate"])
            audio = np.frombuffer(resp.read(), dtype="<f4").astype(np.float32)
        return audio, sr

    return _synthesize
# 03-03 lever 3 (`voice.stream_segments`): a per-segment engine yields `(audio, sr)` tuples
# ONE Kokoro pipeline segment at a time, instead of the whole-clip `Engine` above waiting for
# every segment before returning anything.
SegmentEngine = Callable[[str], Iterator[tuple[np.ndarray, int]]]


class Voice:
    def __init__(
        self,
        cfg: Config,
        engine: Engine | None = None,
        segment_engine: SegmentEngine | None = None,
    ) -> None:
        self.cfg = cfg.voice
        if engine is not None:
            self._engine: Engine = engine
            self._segment_engine: SegmentEngine | None = segment_engine
        elif self.cfg.engine == "chatterbox":
            self._engine, self._segment_engine = self._load_chatterbox_http()
            if segment_engine is not None:
                self._segment_engine = segment_engine
        else:
            # Both engines share ONE loaded Kokoro model/pipeline (`_load_kokoro` builds both
            # closures over it) so turning `stream_segments` on doesn't double the GPU load.
            self._engine, self._segment_engine = self._load_kokoro()
            if segment_engine is not None:
                self._segment_engine = segment_engine
        # Warm up once at startup so the first real line doesn't pay model/weight load time
        # (spec §4.5) -- 01-04 measured load alone at ~2.1s, every voice.
        self._engine(_WARMUP_TEXT)

    def _load_kokoro(self) -> tuple[Engine, SegmentEngine]:
        """Loads Kokoro from LOCAL files only, the same way `bake.render_manifest` does (that path
        rendered 387 lines with --network none). A bare `KPipeline(lang_code=...)` fetches weights
        from the Hugging Face Hub and fails offline: race day has no internet.

        Returns BOTH a whole-clip `Engine` (concatenates every pipeline segment before
        returning -- current/default behaviour) and a per-segment `SegmentEngine` (03-03
        lever 3) built from the SAME `pipeline`/`model` closure, so loading once covers
        either mode `speak()` picks at call time.
        """
        import torch
        from kokoro import KModel, KPipeline  # lazy: needs the baked en_core_web_sm spaCy model

        kokoro_dir = Path(self.cfg.kokoro_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = (
            KModel(config=str(kokoro_dir / "config.json"), model=str(kokoro_dir / "kokoro-v1_0.pth"))
            .to(device)
            .eval()
        )
        lang_code = "b" if self.cfg.voice.startswith("b") else "a"  # Kokoro: 'b'=British, 'a'=American
        pipeline = KPipeline(lang_code=lang_code, model=model)
        voice_path = str(kokoro_dir / "voices" / f"{self.cfg.voice}.pt")

        def _synthesize(text: str) -> tuple[np.ndarray, int]:
            chunks = [audio for _, _, audio in pipeline(text, voice=voice_path, speed=self.cfg.speed)]
            # Kokoro yields torch tensors (CPU) per chunk; np.asarray each, as bake.render_manifest does.
            audio = np.concatenate([np.asarray(c) for c in chunks]) if chunks else np.zeros(0, dtype=np.float32)
            return audio.astype(np.float32), 24000  # kokoro's native rate, 01-04 bake-off

        def _synthesize_segments(text: str) -> Iterator[tuple[np.ndarray, int]]:
            for _, _, audio in pipeline(text, voice=voice_path, speed=self.cfg.speed):
                yield np.asarray(audio).astype(np.float32), 24000

        return _synthesize, _synthesize_segments

    def _load_chatterbox_http(self) -> tuple[Engine, SegmentEngine]:
        engine = chatterbox_http_engine(self.cfg.tts_url, self.cfg.tts_timeout_s)

        def _synthesize_segments(text: str) -> Iterator[tuple[np.ndarray, int]]:
            # Chatterbox renders a whole request before returning anything, so one request
            # per sentence is what gets the first sentence out early under `stream_segments`.
            for sentence in split_sentences(text):
                yield engine(sentence)

        return engine, _synthesize_segments

    def _resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if sr == _OUTPUT_SR or audio.size == 0:
            return audio
        g = np.gcd(_OUTPUT_SR, sr)
        return resample_poly(audio, _OUTPUT_SR // g, sr // g).astype(np.float32)

    async def speak(self, text: str) -> AsyncIterator[np.ndarray]:
        """Synthesises `text` (GPU call(s) in a thread executor), resamples to 48kHz, and
        yields it through a fresh `BarkerFX` instance in `_CHUNK_MS` chunks -- fresh FX state
        per utterance, no cross-line filter/reverb carryover.

        When `voice.stream_segments` is on (03-03 lever 3) and a segment engine is
        available, each Kokoro pipeline segment is synthesised (and its own chunks yielded)
        as soon as IT is ready, instead of waiting for the whole line -- the first segment's
        chunks (and the caller's `voice_first_audio` timing) land well before later segments
        even start. ONE `BarkerFX` instance still covers every segment of this utterance, so
        its filter/reverb state carries across the segment boundary exactly as it would
        across chunk boundaries within a single whole-clip render.
        """
        fx = BarkerFX(sr=_OUTPUT_SR, preset=self.cfg.fx_chain)
        chunk_n = max(_OUTPUT_SR * _CHUNK_MS // 1000, 1)
        loop = asyncio.get_running_loop()

        if self.cfg.stream_segments and self._segment_engine is not None:
            segments = self._segment_engine(text)
            while True:
                result = await loop.run_in_executor(None, _next_or_sentinel, segments)
                if result is _SEGMENTS_DONE:
                    break
                audio, sr = result
                resampled = self._resample(audio, sr)
                for start in range(0, len(resampled), chunk_n):
                    yield fx.process(resampled[start : start + chunk_n])
            return

        audio, sr = await loop.run_in_executor(None, self._engine, text)
        resampled = self._resample(audio, sr)
        for start in range(0, len(resampled), chunk_n):
            yield fx.process(resampled[start : start + chunk_n])

    def render_to_array(self, text: str) -> np.ndarray:
        """Synchronous whole-clip render (tests, library bake renders) -- same resample+FX
        path as `speak()`, collapsed into one array instead of a chunk stream. `fx.py`'s
        `BarkerFX.process()` is documented to match streamed-vs-block output within float
        tolerance, so this and `speak()` produce the same audio either way."""
        audio, sr = self._engine(text)
        resampled = self._resample(audio, sr)
        fx = BarkerFX(sr=_OUTPUT_SR, preset=self.cfg.fx_chain)
        return fx.process(resampled)
