import asyncio
import types

import pytest

# scarebox.app pulls in the perceiver (OpenCV, PyTorch, supervision), which a CPU-only dev
# machine may not have; skip the module with a reason instead of erroring at collection.
for _mod in ("cv2", "torch", "supervision"):
    pytest.importorskip(_mod, reason=f"scarebox.app needs {_mod} (OpenCV/GPU stack), not installed")

from scarebox.app import _backend_health_urls, _make_sink, _wait_for_backends, _watchdog
from scarebox.stage import AlsaSink, WavSink


def _cfg(sink: str):
    return types.SimpleNamespace(stage=types.SimpleNamespace(sink=sink, alsa_device="scarebox_out"))


def test_make_sink_wav_never_touches_a_device():
    sink = _make_sink(_cfg("wav"))
    assert isinstance(sink, WavSink)


def test_make_sink_alsa_always_returns_a_real_alsa_sink():
    # No silent substitution (02-02's bug): "alsa" means AlsaSink, full stop, regardless of
    # whether /dev/snd or aplay exist on THIS host -- openability is selftest's job.
    sink = _make_sink(_cfg("alsa"))
    assert isinstance(sink, AlsaSink)
    # The configured card name becomes a PCM aplay can open (2026-10-19, stage.alsa_pcm_name).
    assert sink.device == "plughw:CARD=scarebox_out,DEV=0"


def test_make_sink_rejects_unknown_value():
    with pytest.raises(ValueError, match="unknown stage.sink"):
        _make_sink(_cfg("mp3"))


def _fake_clock():
    state = {"t": 0.0}

    def clock():
        return state["t"]

    def advance(dt):
        state["t"] += dt

    return clock, advance


async def _run_watchdog_ticks(clock, advance, frame_timeout_s, tick_timeout_s, ticks):
    """Drives `_watchdog` through `ticks` (dt, expect_exit) steps using a fake clock and a
    `poll_s` of 0 so the test doesn't actually sleep. Returns the exit codes recorded."""
    stop = asyncio.Event()
    now = clock()
    last_frame_at = [now]
    last_tick_at = [now]
    exits: list[int] = []

    def exit_fn(code):
        exits.append(code)
        stop.set()

    task = asyncio.create_task(
        _watchdog(
            stop,
            last_frame_at,
            last_tick_at,
            frame_timeout_s,
            tick_timeout_s,
            clock=clock,
            poll_s=0.0,
            exit_fn=exit_fn,
        )
    )
    for dt, touch in ticks:
        advance(dt)
        if touch == "frame":
            last_frame_at[0] = clock()
            last_tick_at[0] = clock()
        elif touch == "tick":
            last_tick_at[0] = clock()
        await asyncio.sleep(0.01)  # let the watchdog task observe this step
    stop.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return exits


def test_watchdog_trips_on_stale_frames():
    clock, advance = _fake_clock()
    exits = asyncio.run(
        _run_watchdog_ticks(clock, advance, frame_timeout_s=30.0, tick_timeout_s=20.0, ticks=[(31.0, None)])
    )
    assert exits == [1]


def test_watchdog_trips_on_stale_tick_even_with_fresh_frames():
    clock, advance = _fake_clock()
    exits = asyncio.run(
        _run_watchdog_ticks(
            clock,
            advance,
            frame_timeout_s=30.0,
            tick_timeout_s=20.0,
            # Frame keeps arriving (resets both), but the tick is never refreshed again
            # after that -- 21s later, tick_age trips before frame_age would.
            ticks=[(5.0, "frame"), (21.0, None)],
        )
    )
    assert exits == [1]


def test_watchdog_does_not_trip_while_fresh():
    clock, advance = _fake_clock()
    exits = asyncio.run(
        _run_watchdog_ticks(
            clock,
            advance,
            frame_timeout_s=30.0,
            tick_timeout_s=20.0,
            ticks=[(10.0, "frame"), (10.0, "frame"), (10.0, "frame")],
        )
    )
    assert exits == []



# --- cold boot: wait for the model servers before the self-test (2026-10-23) ---------------


def test_wait_for_backends_waits_until_every_server_answers():
    # The field cold boot started the app ~25 s before the judge and guard; its self-test
    # failed and it stayed in library-only mode all run.
    ready_after = {"judge": 3, "guard": 5}
    calls = {"judge": 0, "guard": 0}
    kept_alive = []

    async def probe(url):
        calls[url] += 1
        return calls[url] >= ready_after[url]

    ok = asyncio.run(_wait_for_backends(["judge", "guard"], 300.0, lambda: kept_alive.append(1),
                                        poll_s=0, probe=probe))
    assert ok is True
    assert calls["guard"] == 5 and calls["judge"] == 3  # a ready server isn't re-polled
    assert len(kept_alive) == 5  # the frame watchdog is fed on every poll


def test_wait_for_backends_gives_up_after_the_timeout():
    t = [0.0]

    def clock():
        t[0] += 10.0
        return t[0]

    async def probe(url):
        return False

    ok = asyncio.run(_wait_for_backends(["judge"], 60.0, lambda: None, clock=clock, poll_s=0, probe=probe))
    assert ok is False


def test_backend_health_urls_cover_judge_guard_and_chatterbox():
    cfg = types.SimpleNamespace(
        judge=types.SimpleNamespace(endpoint="http://127.0.0.1:8100/v1"),
        censor=types.SimpleNamespace(guard_endpoint="http://127.0.0.1:8101/v1/"),
        voice=types.SimpleNamespace(engine="chatterbox", tts_url="http://127.0.0.1:8102"),
    )
    assert _backend_health_urls(cfg) == [
        "http://127.0.0.1:8100/v1/models", "http://127.0.0.1:8101/v1/models", "http://127.0.0.1:8102/health",
    ]
    cfg.voice.engine = "kokoro"
    assert len(_backend_health_urls(cfg)) == 2
