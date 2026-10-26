import asyncio

import numpy as np
import pytest

from scarebox.censor import Pass, Reject
from scarebox.events import Verdict
from scarebox.selftest import run_selftest
from scarebox.stage import AlsaSink, WavSink


class _FakeJudge:
    def __init__(self, verdict="ok"):
        self._verdict = verdict

    async def judge(self, event):
        if self._verdict == "ok":
            return Verdict(costume="ghost", category="costume", confidence=0.9, is_costume=True, group_size=1)
        if self._verdict == "none":
            return None
        raise RuntimeError("judge server down")


class _FakeCensor:
    def __init__(self, outcome="pass"):
        self._outcome = outcome

    async def check(self, text, category=""):
        if self._outcome == "pass":
            return Pass()
        if self._outcome == "reject":
            return Reject(reason="nope")
        raise RuntimeError("guard server down")


class _FakeVoice:
    def __init__(self, outcome="ok"):
        self._outcome = outcome

    async def speak(self, text):
        if self._outcome == "ok":
            yield np.ones(10, dtype=np.float32)
        elif self._outcome == "empty":
            return
            yield  # pragma: no cover -- makes this an async generator
        else:
            raise RuntimeError("voice engine down")
            yield  # pragma: no cover


async def _ok_frame():
    return object()


async def _no_frame():
    await asyncio.sleep(0)
    return None


def _kwargs(**overrides):
    base = dict(
        get_frame=_ok_frame,
        judge=_FakeJudge("ok"),
        censor=_FakeCensor("pass"),
        voice=_FakeVoice("ok"),
        sink=WavSink.__new__(WavSink),  # not an AlsaSink; audio check short-circuits True
        requested_alsa=False,
    )
    base.update(overrides)
    return base


def test_all_pass_plays_meta_ready():
    result = asyncio.run(run_selftest(**_kwargs()))
    assert result.ok is True
    assert result.failed_check is None
    assert result.meta_id == "meta-ready"
    assert result.library_only is False


def test_camera_failure_does_not_demote_to_library_only():
    result = asyncio.run(run_selftest(**_kwargs(get_frame=_no_frame)))
    assert result.failed_check == "camera"
    assert result.meta_id == "meta-err-camera"
    assert result.library_only is False


def test_judge_failure_is_library_only():
    result = asyncio.run(run_selftest(**_kwargs(judge=_FakeJudge("error"))))
    assert result.failed_check == "brain"
    assert result.meta_id == "meta-err-brain"
    assert result.library_only is True


def test_judge_returning_none_also_fails():
    result = asyncio.run(run_selftest(**_kwargs(judge=_FakeJudge("none"))))
    assert result.failed_check == "brain"


def test_censor_failure_is_library_only():
    result = asyncio.run(run_selftest(**_kwargs(censor=_FakeCensor("reject"))))
    assert result.failed_check == "censor"
    assert result.meta_id == "meta-err-censor"
    assert result.library_only is True


def test_censor_server_down_also_fails():
    result = asyncio.run(run_selftest(**_kwargs(censor=_FakeCensor("error"))))
    assert result.failed_check == "censor"


def test_voice_failure_is_library_only():
    result = asyncio.run(run_selftest(**_kwargs(voice=_FakeVoice("error"))))
    assert result.failed_check == "voice"
    assert result.meta_id == "meta-err-voice"
    assert result.library_only is True


def test_voice_empty_audio_also_fails():
    result = asyncio.run(run_selftest(**_kwargs(voice=_FakeVoice("empty"))))
    assert result.failed_check == "voice"


def test_audio_check_skipped_when_wav_sink_requested():
    # requested_alsa=False (stage.sink: wav, bench): no device probe at all, even if the
    # sink object somehow were an AlsaSink that would blow up if actually played.
    broken_alsa = AlsaSink("nonexistent-device")
    broken_alsa.play = lambda pcm, sr: (_ for _ in ()).throw(FileNotFoundError("aplay"))
    result = asyncio.run(run_selftest(**_kwargs(sink=broken_alsa, requested_alsa=False)))
    assert result.ok is True


def test_audio_device_unopenable_fails_loudly_not_silently():
    broken_alsa = AlsaSink("nonexistent-device")
    broken_alsa.play = lambda pcm, sr: (_ for _ in ()).throw(FileNotFoundError("aplay"))
    result = asyncio.run(run_selftest(**_kwargs(sink=broken_alsa, requested_alsa=True)))
    assert result.failed_check == "audio"
    assert result.meta_id == "meta-err-audio"
    # Audio failing is NOT a brain/censor/voice failure -- it doesn't demote to library-only;
    # `app.py` handles it by swapping the sink, not by abandoning the live pipeline.
    assert result.library_only is False


def test_audio_device_opens_when_working():
    working_alsa = AlsaSink("real-device")
    working_alsa.play = lambda pcm, sr: None
    result = asyncio.run(run_selftest(**_kwargs(sink=working_alsa, requested_alsa=True)))
    assert result.ok is True
