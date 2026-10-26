"""Unit tests for `voice.Voice` using a fake injected engine -- no GPU, no real `kokoro`
(a CPU-only dev `.venv` has neither). The real Kokoro engine path (`Voice._load_kokoro`) is
UNVERIFIED this session (see voice.py's module docstring) and is only exercised by the live
`--network none` container smoke test this plan didn't get to run -- tracked as follow-up
work, not covered here.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from scarebox.config import load_config
from scarebox.voice import Voice


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("CAMERA_HOST", "x")
    monkeypatch.setenv("CAMERA_USER", "x")
    monkeypatch.setenv("CAMERA_PASSWORD", "x")
    monkeypatch.setenv("AIM_TOKEN", "x")
    return load_config("config/bench.yaml")


class _FakeEngine:
    """`text -> (audio, sample_rate)`, same shape `Voice` expects from a real engine.
    Deterministic: a short burst of silence, length proportional to `len(text)`, at 24kHz
    (Kokoro's native rate, 01-04) so resampling to 48kHz is actually exercised."""

    def __init__(self, sr: int = 24000):
        self.sr = sr
        self.calls: list[str] = []

    def __call__(self, text: str) -> tuple[np.ndarray, int]:
        self.calls.append(text)
        n = max(len(text) * 100, 100)
        # A little sine, not pure silence, so resample/FX don't just push zeros through.
        t = np.arange(n) / self.sr
        audio = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        return audio, self.sr


def test_warms_up_on_construction(cfg):
    engine = _FakeEngine()
    Voice(cfg, engine=engine)
    assert engine.calls == ["Ready."]


def test_render_to_array_resamples_to_48k(cfg):
    engine = _FakeEngine(sr=24000)
    voice = Voice(cfg, engine=engine)
    out = voice.render_to_array("A rex! Marvellously prehistoric.")
    assert out.dtype == np.float32
    # 24kHz -> 48kHz is an exact x2 -- output should be roughly double the synthesised length
    # (resample_poly's filter adds a few samples of edge effect, hence the tolerance).
    expected_in_len = max(len("A rex! Marvellously prehistoric.") * 100, 100)
    assert abs(len(out) - expected_in_len * 2) < 10


def test_render_to_array_passthrough_when_already_48k(cfg):
    engine = _FakeEngine(sr=48000)
    voice = Voice(cfg, engine=engine)
    out = voice.render_to_array("Short.")
    expected_in_len = max(len("Short.") * 100, 100)
    assert len(out) == expected_in_len


def test_speak_yields_48k_float32_chunks(cfg):
    engine = _FakeEngine(sr=24000)
    voice = Voice(cfg, engine=engine)

    async def _collect():
        chunks = []
        async for chunk in voice.speak("A rex! Marvellously prehistoric."):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())
    assert len(chunks) > 1  # chunked, not one giant blob
    for c in chunks:
        assert c.dtype == np.float32
        assert len(c) <= 48000 * 20 // 1000  # 20ms chunk cap


def test_speak_matches_render_to_array_within_tolerance(cfg):
    """fx.py documents block-vs-streamed processing as equivalent within float tolerance --
    this exercises that equivalence through the real Voice/BarkerFX wiring, not just fx.py's
    own unit tests."""
    text = "A rex! Marvellously prehistoric."

    async def _collect(voice):
        chunks = []
        async for chunk in voice.speak(text):
            chunks.append(chunk)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    streamed = asyncio.run(_collect(Voice(cfg, engine=_FakeEngine(sr=24000))))
    blocked = Voice(cfg, engine=_FakeEngine(sr=24000)).render_to_array(text)
    assert streamed.shape == blocked.shape
    assert np.allclose(streamed, blocked, atol=1e-5)


class _FakeSegmentEngine:
    """`text -> Iterator[(audio, sr)]`, one call per `speak()` invocation, N short segments
    (03-03 lever 3: `voice.stream_segments`)."""

    def __init__(self, n_segments: int = 3, sr: int = 24000):
        self.n_segments = n_segments
        self.sr = sr
        self.calls: list[str] = []

    def __call__(self, text: str):
        self.calls.append(text)

        def _gen():
            for i in range(self.n_segments):
                t = np.arange(50) / self.sr
                yield (0.1 * np.sin(2 * np.pi * (220.0 + i) * t)).astype(np.float32), self.sr

        return _gen()


def test_stream_segments_off_ignores_segment_engine(cfg):
    # With stream_segments off, speak() must use the whole-clip engine even when a segment
    # engine happens to be injected. (bench.yaml turned it on 2026-10-16 for Chatterbox, so
    # force it off here.)
    import dataclasses

    cfg = dataclasses.replace(cfg, voice=dataclasses.replace(cfg.voice, stream_segments=False))
    engine = _FakeEngine(sr=24000)
    seg_engine = _FakeSegmentEngine()
    voice = Voice(cfg, engine=engine, segment_engine=seg_engine)

    asyncio.run(_collect_chunks(voice, "A rex! Marvellously prehistoric."))
    assert seg_engine.calls == []  # never touched


def test_stream_segments_on_uses_segment_engine_and_matches_single_fx_pass(cfg):
    """ONE `BarkerFX` instance covers every segment of the utterance: streaming 3 segments
    through `speak()` must equal running all 3 segments' raw audio concatenated through a
    single `BarkerFX.process()` call -- not 3 independent fresh-state renders."""
    import dataclasses

    from scarebox.fx import BarkerFX

    streamed_cfg = dataclasses.replace(cfg, voice=dataclasses.replace(cfg.voice, stream_segments=True))
    seg_engine = _FakeSegmentEngine(n_segments=3)
    voice = Voice(streamed_cfg, engine=_FakeEngine(sr=24000), segment_engine=seg_engine)

    chunks = asyncio.run(_collect_chunks(voice, "Three short segments."))
    assert seg_engine.calls == ["Three short segments."]
    assert len(chunks) > 1  # still chunked
    streamed = np.concatenate(chunks)

    # Resample PER SEGMENT (exactly what speak() does -- resample_poly has edge effects, so
    # resampling the whole concatenated clip in one go is a different, not-comparable, signal)
    # then run the concatenated resampled audio through ONE BarkerFX call -- isolates the
    # "single FX instance across segments" claim from resample-boundary noise.
    raw_segments = [audio for audio, sr in seg_engine("unused -- same deterministic generator")]
    resampled_concat = np.concatenate([voice._resample(seg, seg_engine.sr) for seg in raw_segments])
    expected = BarkerFX(sr=48000, preset=streamed_cfg.voice.fx_chain).process(resampled_concat)
    assert streamed.shape == expected.shape
    assert np.allclose(streamed, expected, atol=1e-5)


async def _collect_chunks(voice, text):
    chunks = []
    async for chunk in voice.speak(text):
        chunks.append(chunk)
    return chunks


def test_fresh_fx_state_per_utterance(cfg):
    """Two calls to speak() the same line should produce identical output -- FX state must
    not leak across utterances (voice.py creates a fresh BarkerFX per call)."""
    voice = Voice(cfg, engine=_FakeEngine(sr=24000))

    async def _collect():
        chunks = []
        async for chunk in voice.speak("Same line twice."):
            chunks.append(chunk)
        return np.concatenate(chunks)

    first = asyncio.run(_collect())
    second = asyncio.run(_collect())
    assert np.array_equal(first, second)


# --- engine "chatterbox": HTTP client for deploy/chatterbox/server.py (2026-10-16) ---------


from scarebox.voice import split_sentences  # noqa: E402


def test_split_sentences_splits_on_terminal_punctuation():
    assert split_sentences("Nearly there, runner! Get a move on!") == [
        "Nearly there, runner!",
        "Get a move on!",
    ]


def test_split_sentences_merges_one_word_interjection_forward():
    assert split_sentences("Arrr! Hoist the sails and hurry, matey!") == [
        "Arrr! Hoist the sails and hurry, matey!"
    ]


def test_split_sentences_trailing_single_word_joins_previous():
    assert split_sentences("Run, run, run! Onward!") == ["Run, run, run! Onward!"]


def test_split_sentences_single_word_line_survives():
    assert split_sentences("Splendid!") == ["Splendid!"]


@pytest.fixture
def fake_tts_server():
    import http.server
    import json
    import threading

    seen: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body["text"])
            pcm = (0.1 * np.ones(2400, dtype="<f4")).tobytes()
            self.send_response(200)
            self.send_header("X-Sample-Rate", "24000")
            self.send_header("Content-Length", str(len(pcm)))
            self.end_headers()
            self.wfile.write(pcm)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", seen
    server.shutdown()


def _chatterbox_cfg(cfg, url, stream_segments=False):
    import dataclasses

    voice_cfg = dataclasses.replace(
        cfg.voice, engine="chatterbox", tts_url=url, stream_segments=stream_segments
    )
    return dataclasses.replace(cfg, voice=voice_cfg)


def test_chatterbox_engine_posts_text_and_resamples(cfg, fake_tts_server):
    url, seen = fake_tts_server
    voice = Voice(_chatterbox_cfg(cfg, url))
    out = voice.render_to_array("Nearly there, runner! Get a move on!")
    assert seen[0] == "Ready."  # warm-up goes through the server too
    assert seen[-1] == "Nearly there, runner! Get a move on!"
    assert out.dtype == np.float32
    assert len(out) == 4800  # 2400 samples @24k -> 48k


def test_chatterbox_stream_segments_sends_one_request_per_sentence(cfg, fake_tts_server):
    url, seen = fake_tts_server
    voice = Voice(_chatterbox_cfg(cfg, url, stream_segments=True))

    async def collect():
        return [c async for c in voice.speak("Nearly there, runner! Get a move on!")]

    chunks = asyncio.run(collect())
    assert seen[1:] == ["Nearly there, runner!", "Get a move on!"]
    assert sum(len(c) for c in chunks) == 9600


def test_chatterbox_server_down_raises(cfg):
    import urllib.error

    with pytest.raises((urllib.error.URLError, OSError)):
        Voice(_chatterbox_cfg(cfg, "http://127.0.0.1:9"))
