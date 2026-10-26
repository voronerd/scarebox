import asyncio
import types
import wave

import numpy as np

from scarebox.stage import NullSink, PlayItem, Stage, WavSink


def _cfg(cooldown_s=1.0, queue_max=2, max_line_age_s=8.0, stings=False):
    return types.SimpleNamespace(
        stage=types.SimpleNamespace(cooldown_s=cooldown_s, queue_max=queue_max, stings=stings, alsa_device="x"),
        orchestrator=types.SimpleNamespace(max_line_age_s=max_line_age_s),
    )


def _tone(n=100) -> np.ndarray:
    return np.full(n, 1000, dtype=np.int16)


def _fake_clock():
    state = {"t": 0.0}

    def clock():
        return state["t"]

    def advance(dt):
        state["t"] += dt

    return clock, advance


def test_queue_overflow_drops_oldest():
    clock, advance = _fake_clock()
    stage = Stage(_cfg(queue_max=2), NullSink(), clock=clock, sfx_dir="no/such/dir")
    stage.enqueue(PlayItem(event_id="a", t_trigger=0.0, audio=_tone()))
    stage.enqueue(PlayItem(event_id="b", t_trigger=0.0, audio=_tone()))
    stage.enqueue(PlayItem(event_id="c", t_trigger=0.0, audio=_tone()))
    ids = [item.event_id for item in stage.queue]
    assert ids == ["b", "c"]


def test_stale_items_dropped():
    clock, advance = _fake_clock()
    stage = Stage(_cfg(max_line_age_s=8.0), NullSink(), clock=clock, sfx_dir="no/such/dir")
    stage.enqueue(PlayItem(event_id="a", t_trigger=0.0, audio=_tone()))
    advance(9.0)
    dropped = stage.drop_stale()
    assert [d.event_id for d in dropped] == ["a"]
    assert stage.queue == []


class _RecordingSink:
    def __init__(self):
        self.plays: list[tuple[np.ndarray, int]] = []

    def play(self, pcm, sr):
        self.plays.append((pcm, sr))


def test_streamed_live_audio_chunk_iterator_is_played():
    """03-03: the orchestrator enqueues a live `PlayItem` whose `audio` is the async chunk
    iterator straight from `voice.Voice.speak()` -- `try_play_next()` has to drain it and
    play the result, same as a plain ndarray/wav path."""

    async def _chunks():
        yield np.full(10, 0.5, dtype=np.float32)
        yield np.full(10, -0.5, dtype=np.float32)

    clock, advance = _fake_clock()
    sink = _RecordingSink()
    stage = Stage(_cfg(), sink, clock=clock, sfx_dir="no/such/dir")
    stage.enqueue(PlayItem(event_id="live1", t_trigger=0.0, audio=_chunks(), kind="live"))

    played = asyncio.run(stage.try_play_next())
    assert played is not None and played.event_id == "live1"
    assert len(sink.plays) == 1
    pcm, sr = sink.plays[0]
    assert sr == 48_000
    assert pcm.dtype == np.int16
    assert len(pcm) == 20  # both chunks drained and concatenated
    # float32 0.5/-0.5 -> int16 PCM, same sign and roughly the expected magnitude.
    assert pcm[0] > 0 and pcm[10] < 0


def test_cooldown_respected_with_fake_clock():
    clock, advance = _fake_clock()
    stage = Stage(_cfg(cooldown_s=1.0, max_line_age_s=8.0), NullSink(), clock=clock, sfx_dir="no/such/dir")
    stage.enqueue(PlayItem(event_id="a", t_trigger=0.0, audio=_tone()))
    stage.enqueue(PlayItem(event_id="b", t_trigger=0.0, audio=_tone()))

    played = asyncio.run(stage.try_play_next())
    assert played is not None and played.event_id == "a"

    # cooldown hasn't elapsed yet -- must not play "b"
    played_too_soon = asyncio.run(stage.try_play_next())
    assert played_too_soon is None

    advance(1.0)
    played_after_cooldown = asyncio.run(stage.try_play_next())
    assert played_after_cooldown is not None and played_after_cooldown.event_id == "b"


def test_cooldown_counts_from_the_end_of_the_line():
    # 2026-10-19: the gap was timed from when a line STARTED, so a 1 s cooldown ran out while
    # a 4 s line was still playing and the next line followed with no gap. The sink here
    # "plays" for 4 s of fake time.
    clock, advance = _fake_clock()

    class SlowSink:
        def play(self, pcm, sr):
            advance(4.0)

    stage = Stage(_cfg(cooldown_s=1.0, max_line_age_s=30.0), SlowSink(), clock=clock, sfx_dir="no/such/dir")
    stage.enqueue(PlayItem(event_id="a", t_trigger=0.0, audio=_tone()))
    stage.enqueue(PlayItem(event_id="b", t_trigger=0.0, audio=_tone()))
    assert asyncio.run(stage.try_play_next()).event_id == "a"
    assert asyncio.run(stage.try_play_next()) is None  # line just ended: gap not over yet
    advance(1.0)
    assert asyncio.run(stage.try_play_next()).event_id == "b"


def test_crowd_plays_the_most_distinctive_line_and_drops_generic_ones():
    # Busy mode (2026-10-19): several lines waiting -> the best costume line wins, lines
    # below the crowd bar are dropped rather than played late.
    from scarebox.stage import SCORE_COSTUME, SCORE_GENERIC, SCORE_NO_COSTUME, SCORE_SPECIFIC

    clock, _advance = _fake_clock()
    stage = Stage(_cfg(cooldown_s=0.0, queue_max=6, max_line_age_s=30.0), NullSink(), clock=clock, sfx_dir="no/such/dir")
    stage.crowd_min_score, stage.quiet_min_score = SCORE_COSTUME, SCORE_NO_COSTUME
    stage.enqueue(PlayItem(event_id="plain", t_trigger=0.0, audio=_tone(), score=SCORE_NO_COSTUME))
    stage.enqueue(PlayItem(event_id="mystery", t_trigger=0.1, audio=_tone(), score=SCORE_GENERIC))
    stage.enqueue(PlayItem(event_id="pirate", t_trigger=0.2, audio=_tone(), score=SCORE_COSTUME))
    stage.enqueue(PlayItem(event_id="spidey", t_trigger=0.3, audio=_tone(), score=SCORE_SPECIFIC))

    assert asyncio.run(stage.try_play_next()).event_id == "spidey"
    assert [i.event_id for i in stage.queue] == ["pirate"]  # plain + mystery dropped
    assert asyncio.run(stage.try_play_next()).event_id == "pirate"  # alone: quiet bar


def test_lone_line_plays_unless_generic():
    from scarebox.stage import SCORE_GENERIC, SCORE_NO_COSTUME

    clock, _advance = _fake_clock()
    stage = Stage(_cfg(cooldown_s=0.0, max_line_age_s=30.0), NullSink(), clock=clock, sfx_dir="no/such/dir")
    stage.crowd_min_score, stage.quiet_min_score = 2, SCORE_NO_COSTUME
    stage.enqueue(PlayItem(event_id="plain", t_trigger=0.0, audio=_tone(), score=SCORE_NO_COSTUME))
    assert asyncio.run(stage.try_play_next()).event_id == "plain"
    stage.enqueue(PlayItem(event_id="mystery", t_trigger=0.0, audio=_tone(), score=SCORE_GENERIC))
    assert asyncio.run(stage.try_play_next()) is None
    assert stage.queue == []


def test_busy_mode_switches_on_with_the_crossing_rate_and_off_when_it_thins():
    # 2026-10-19: a crowd is detected from the crossing rate, before lines pile up. A lone
    # no-costume line is dropped while 3+ people crossed in the last 10 s, played otherwise.
    from scarebox.stage import SCORE_COSTUME, SCORE_NO_COSTUME

    clock, advance = _fake_clock()
    stage = Stage(_cfg(cooldown_s=0.0, max_line_age_s=30.0), NullSink(), clock=clock, sfx_dir="no/such/dir")
    stage.crowd_min_score, stage.quiet_min_score = SCORE_COSTUME, SCORE_NO_COSTUME
    for _ in range(3):
        stage.note_crossing(clock())
    assert stage.crowded()
    stage.enqueue(PlayItem(event_id="plain", t_trigger=clock(), audio=_tone(), score=SCORE_NO_COSTUME))
    assert asyncio.run(stage.try_play_next()) is None  # crowd: goading dropped

    advance(11.0)  # crowd gone
    assert not stage.crowded()
    stage.enqueue(PlayItem(event_id="plain2", t_trigger=clock(), audio=_tone(), score=SCORE_NO_COSTUME))
    assert asyncio.run(stage.try_play_next()).event_id == "plain2"


def test_busy_gap_replaces_the_normal_gap_only_when_crowded():
    clock, advance = _fake_clock()
    stage = Stage(_cfg(cooldown_s=2.5, max_line_age_s=30.0), NullSink(), clock=clock, sfx_dir="no/such/dir")
    stage.busy_gap_s = 1.0
    stage.enqueue(PlayItem(event_id="a", t_trigger=clock(), audio=_tone()))
    assert asyncio.run(stage.try_play_next()).event_id == "a"
    stage.enqueue(PlayItem(event_id="b", t_trigger=clock(), audio=_tone()))
    advance(1.2)
    assert asyncio.run(stage.try_play_next()) is None  # quiet: the full 2.5 s gap applies
    for _ in range(3):
        stage.note_crossing(clock())
    assert asyncio.run(stage.try_play_next()).event_id == "b"  # crowded: 1.0 s is enough


def test_a_group_counts_as_several_crossings():
    clock, _advance = _fake_clock()
    stage = Stage(_cfg(), NullSink(), clock=clock, sfx_dir="no/such/dir")
    stage.note_crossing(clock(), people=3)
    assert stage.crowded()


def test_a_failed_play_is_recorded_and_the_next_line_still_plays():
    # 2026-10-19: aplay failing used to raise out of try_play_next and kill the speaker loop
    # for good. Now it's recorded (speaker flag on the phone page) and playback carries on.
    clock, advance = _fake_clock()

    class FlakySink:
        def __init__(self):
            self.calls = 0

        def play(self, pcm, sr):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("aplay: audio open error: No such device")

    sink = FlakySink()
    stage = Stage(_cfg(cooldown_s=0.0, max_line_age_s=30.0), sink, clock=clock, sfx_dir="no/such/dir")
    stage.enqueue(PlayItem(event_id="a", t_trigger=0.0, audio=_tone()))
    stage.enqueue(PlayItem(event_id="b", t_trigger=0.0, audio=_tone()))
    assert asyncio.run(stage.try_play_next()) is None
    assert stage.audio_ok is False and "No such device" in stage.audio_error
    assert asyncio.run(stage.try_play_next()) is not None
    assert stage.audio_ok is True and stage.audio_error is None and stage.audio_failures == 1


def test_playing_a_line_does_not_freeze_the_event_loop():
    # 2026-10-19: sink.play blocks for the whole line; called directly it froze the app's
    # only event loop (no frames, tracking or verdicts) for every ~4 s line.
    import time as _time

    class SlowRealSink:
        def play(self, pcm, sr):
            _time.sleep(0.3)

    stage = Stage(_cfg(cooldown_s=0.0, max_line_age_s=30.0), SlowRealSink(), sfx_dir="no/such/dir")
    stage.enqueue(PlayItem(event_id="a", t_trigger=_time.monotonic(), audio=_tone()))
    ticks = []

    async def ticker():
        for _ in range(20):
            ticks.append(_time.monotonic())
            await asyncio.sleep(0.01)

    async def play_after_ticker_starts():
        await asyncio.sleep(0.03)
        await stage.try_play_next()

    async def both():
        await asyncio.gather(ticker(), play_after_ticker_starts())

    asyncio.run(both())
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert max(gaps) < 0.15, f"event loop stalled {max(gaps):.2f}s during playback"


def test_meta_lines_always_outrank_runner_lines():
    item = PlayItem(event_id="selftest", t_trigger=0.0, audio=_tone(), kind="meta")
    from scarebox.stage import SCORE_META

    assert item.score == SCORE_META


def test_wavsink_file_appears(tmp_path):
    sink = WavSink(tmp_path)
    sink.play(_tone(480), 48_000)
    files = list(tmp_path.glob("*.wav"))
    assert len(files) == 1
    with wave.open(str(files[0]), "rb") as wf:
        assert wf.getnframes() == 480
        assert wf.getframerate() == 48_000
        assert wf.getsampwidth() == 2
        assert wf.getnchannels() == 1


def test_alsa_card_name_becomes_a_pcm_aplay_can_open():
    # 2026-10-19 on gbox1: `aplay -D scarebox_out` -> "audio open error: No such file or
    # directory". The config names the card; aplay needs a PCM.
    from scarebox.stage import AlsaSink, alsa_pcm_name

    assert alsa_pcm_name("scarebox_out") == "plughw:CARD=scarebox_out,DEV=0"
    assert AlsaSink("scarebox_out").device == "plughw:CARD=scarebox_out,DEV=0"
    for already_pcm in ("plughw:CARD=x,DEV=0", "hw:1,0", "default"):
        assert alsa_pcm_name(already_pcm) == already_pcm
