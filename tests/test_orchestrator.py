"""Unit tests for `orchestrator.Orchestrator` -- the live per-event state machine (spec
§4.4). All fakes, no network/GPU: `judge`/`quip`/`voice` are duck-typed stand-ins (same
pattern as `tests/test_quip.py`'s `_FakeCensor`), driven with small REAL sleeps (a few ms)
rather than a frozen clock, since the orchestrator races a background `asyncio.Task` against
a real deadline computed from `event.t_trigger` + `cfg.orchestrator.budget_s`.
"""

from __future__ import annotations

import asyncio
import logging
import time
import types

import numpy as np
import pytest

from scarebox.events import Event, LineResult, Verdict
from scarebox.orchestrator import Orchestrator
from scarebox.stage import NullSink, Stage
from scarebox.tuning import Tuning


def _cfg(budget_s=1.0, roast_non_costumes=True, single_call=False, max_line_age_s=8.0):
    return types.SimpleNamespace(
        orchestrator=types.SimpleNamespace(budget_s=budget_s, max_line_age_s=max_line_age_s),
        quip=types.SimpleNamespace(roast_non_costumes=roast_non_costumes, single_call=single_call),
        stage=types.SimpleNamespace(cooldown_s=0.0, queue_max=10, stings=False),
    )


def _event(t_trigger=None, group_size=1):
    # Real `time.monotonic()` "now" (not 0.0) -- the orchestrator's default clock is
    # `time.monotonic`, and the deadline is `t_trigger + budget_s` compared against it.
    t_trigger = time.monotonic() if t_trigger is None else t_trigger
    crops = [np.zeros((4, 4, 3), dtype=np.uint8)]
    return Event(t_trigger=t_trigger, track_ids=list(range(group_size)), group_size=group_size, crops=crops)


def _verdict(**overrides) -> Verdict:
    base = dict(costume="pirate hat", category="pirate", confidence=0.9, is_costume=True, group_size=1, details="")
    base.update(overrides)
    return Verdict(**base)


class FakeJudge:
    """`.judge(event)` -- optionally slow, optionally raising, optionally immune to
    cancellation (simulating a thread-executor GPU call asyncio can't actually interrupt,
    the realistic cause of a `late` result)."""

    def __init__(self, verdict=None, delay: float = 0.0, exc: Exception | None = None, ignore_cancel: bool = False):
        self.verdict = verdict
        self.delay = delay
        self.exc = exc
        self.ignore_cancel = ignore_cancel
        self.calls = 0

    async def judge(self, event):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        if self.delay:
            try:
                await asyncio.sleep(self.delay)
            except asyncio.CancelledError:
                if not self.ignore_cancel:
                    raise
        return self.verdict


class FakeQuip:
    def __init__(
        self,
        line: str | None = "A pirate! Arrr.",
        reason: str | None = None,
        detail: str | None = None,
        delay: float = 0.0,
        single_verdict=None,
    ):
        self.line = line
        self.reason = reason
        self.detail = detail
        self.delay = delay
        self.single_verdict = single_verdict
        self.quip_calls = 0
        self.single_calls = 0

    async def quip_with_reason(self, verdict, category):
        self.quip_calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.line, self.reason, self.detail

    async def judge_and_quip_with_reason(self, event):
        self.single_calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.single_verdict, self.line, self.reason, self.detail

    # 04-04: the orchestrator's live path now calls these `_with_blocked` variants instead
    # (so it can surface a censor-blocked line's TEXT on the aim page's event card, not just
    # its reason) -- `blocked` is left empty here since no test in this file models an
    # actual censor reject; `tests/test_quip.py` covers `blocked` being filled for real.
    async def quip_with_blocked(self, verdict, category, blocked):
        return await self.quip_with_reason(verdict, category)

    async def judge_and_quip_with_blocked(self, event, blocked):
        return await self.judge_and_quip_with_reason(event)


class FakeVoice:
    def __init__(self, n_chunks: int = 3):
        self.n_chunks = n_chunks
        self.requested: list[str] = []

    def speak(self, text):
        self.requested.append(text)

        async def _gen():
            for _ in range(self.n_chunks):
                yield np.zeros(10, dtype=np.float32)

        return _gen()


class FakeLibrary:
    def __init__(self, line_text: str | None = "A generic jest."):
        self.line_text = line_text
        self.picked_categories: list[str] = []
        self.picked_costumes: list[str] = []

    def pick(self, category, exclude_last: int = 10, costume: str = "", prefer_short: bool = False):
        self.picked_categories.append(category)
        self.picked_costumes.append(costume)
        if self.line_text is None:
            return None
        return LineResult(text=self.line_text, category=category, path="library", wav="fake.wav")


def _orch(judge, quip, voice, library, cfg, tuning=None):
    stage = Stage(cfg, NullSink(), sfx_dir="no/such/dir")
    return Orchestrator(judge, quip, voice, library, stage, cfg, tuning=tuning), stage


def test_live_on_time_produces_live_line():
    cfg = _cfg(budget_s=1.0)
    judge = FakeJudge(verdict=_verdict())
    quip = FakeQuip(line="A pirate! Arrr.")
    voice = FakeVoice()
    library = FakeLibrary()
    orch, stage = _orch(judge, quip, voice, library, cfg)

    record = asyncio.run(orch.handle_event(_event()))

    assert record["path"] == "live"
    assert record["line"] == "A pirate! Arrr."
    assert record["censor"] == "pass"
    assert record["category"] == "pirate"
    assert len(stage.queue) == 1 and stage.queue[0].kind == "live"
    # Only the FIRST chunk is consumed before enqueueing -- the orchestrator hands stage the
    # rest of the iterator rather than collecting the whole render itself.
    assert "voice_first_audio" in record["ms"]


def test_slow_judge_falls_back_to_mystery():
    cfg = _cfg(budget_s=0.02)
    judge = FakeJudge(verdict=_verdict(), delay=0.3)  # way past the deadline
    quip = FakeQuip()
    voice = FakeVoice()
    library = FakeLibrary()
    orch, stage = _orch(judge, quip, voice, library, cfg)

    record = asyncio.run(orch.handle_event(_event()))

    assert record["category"] == "mystery"
    assert record["path"] == "library"
    assert library.picked_categories == ["mystery"]
    assert record["censor"] is None


def test_slow_quip_falls_back_with_judged_category():
    cfg = _cfg(budget_s=0.02)
    judge = FakeJudge(verdict=_verdict(category="pirate"))
    quip = FakeQuip(delay=0.3)
    voice = FakeVoice()
    library = FakeLibrary()
    orch, stage = _orch(judge, quip, voice, library, cfg)

    record = asyncio.run(orch.handle_event(_event()))

    assert record["category"] == "pirate"  # kept even though quip never finished in time
    assert record["path"] == "library"
    assert library.picked_categories == ["pirate"]


def test_censor_rejects_all_candidates_falls_back():
    cfg = _cfg(budget_s=1.0)
    judge = FakeJudge(verdict=_verdict(category="zombie"))
    quip = FakeQuip(line=None, reason="guard-risk")
    voice = FakeVoice()
    library = FakeLibrary()
    orch, stage = _orch(judge, quip, voice, library, cfg)

    record = asyncio.run(orch.handle_event(_event()))

    assert record["path"] == "library"
    assert record["category"] == "zombie"
    assert record["reject_reason"] == "guard-risk"
    assert record["censor"] is None


@pytest.mark.parametrize("reason", ["deterministic", "guard-risk", "guard-timeout", "parse", "quip-none"])
def test_reject_reason_bucket_passed_through(reason):
    cfg = _cfg(budget_s=1.0)
    judge = FakeJudge(verdict=_verdict())
    quip = FakeQuip(line=None, reason=reason)
    orch, stage = _orch(judge, quip, FakeVoice(), FakeLibrary(), cfg)

    record = asyncio.run(orch.handle_event(_event()))
    assert record["reject_reason"] == reason
    assert record["path"] == "library"


def test_reject_detail_passed_through():
    cfg = _cfg(budget_s=1.0)
    judge = FakeJudge(verdict=_verdict(category="zombie"))
    quip = FakeQuip(line=None, reason="guard-risk", detail="guard:violence")
    orch, stage = _orch(judge, quip, FakeVoice(), FakeLibrary(), cfg)

    record = asyncio.run(orch.handle_event(_event()))

    assert record["reject_reason"] == "guard-risk"
    assert record["reject_detail"] == "guard:violence"
    assert record["path"] == "library"


def test_reject_detail_none_on_live_success():
    cfg = _cfg(budget_s=1.0)
    judge = FakeJudge(verdict=_verdict())
    quip = FakeQuip(line="A pirate! Arrr.")
    orch, stage = _orch(judge, quip, FakeVoice(), FakeLibrary(), cfg)

    record = asyncio.run(orch.handle_event(_event()))

    assert record["path"] == "live"
    assert record["reject_reason"] is None
    assert record["reject_detail"] is None


def test_late_live_result_discarded(caplog):
    caplog.set_level(logging.INFO, logger="scarebox.orchestrator")
    cfg = _cfg(budget_s=0.01)
    # ignore_cancel=True: simulates a GPU call asyncio's cooperative cancellation can't
    # actually interrupt (e.g. loop.run_in_executor) -- it keeps going and returns a real
    # verdict even after the orchestrator has already moved to fallback.
    judge = FakeJudge(verdict=_verdict(), delay=0.05, ignore_cancel=True)
    quip = FakeQuip(line="Too late, matey.")
    voice = FakeVoice()
    library = FakeLibrary()
    orch, stage = _orch(judge, quip, voice, library, cfg)

    record = asyncio.run(orch.handle_event(_event()))
    assert record["path"] == "library"  # fallback committed first

    # Give the stray live task (and its done-callback) time to actually finish and fire.
    asyncio.run(asyncio.sleep(0.15))

    assert "orchestrator.late" in caplog.text
    # The late result was NEVER enqueued -- exactly the one fallback item is in the queue.
    assert len(stage.queue) == 1
    assert stage.queue[0].kind == "library"


def test_exception_means_fallback(caplog):
    caplog.set_level(logging.WARNING, logger="scarebox.orchestrator")
    cfg = _cfg(budget_s=1.0)
    judge = FakeJudge(exc=RuntimeError("vlm exploded"))
    orch, stage = _orch(judge, FakeQuip(), FakeVoice(), FakeLibrary(), cfg)

    record = asyncio.run(orch.handle_event(_event()))

    assert record["path"] == "library"
    assert record["category"] == "mystery"
    assert "orchestrator.live_exception" in caplog.text


def test_no_costume_skips_quip_when_roast_disabled():
    cfg = _cfg(budget_s=1.0, roast_non_costumes=False)
    judge = FakeJudge(verdict=_verdict(category="no_costume", costume="", is_costume=False))
    quip = FakeQuip(line="should never be used")
    library = FakeLibrary()
    orch, stage = _orch(judge, quip, FakeVoice(), library, cfg)

    record = asyncio.run(orch.handle_event(_event()))

    # 2026-10-19: switched off means silence for non-costumed runners (Martin's original
    # ask: generic lines for non-costumes, "we can turn that on or off"). It used to still
    # play a pre-recorded no_costume line.
    assert quip.quip_calls == 0
    assert record["path"] == "silent"
    assert library.picked_categories == []
    assert stage.queue == []


def test_no_costume_gets_a_prerecorded_line_never_the_live_writer():
    # 2026-10-19 busy mode: the single live slot is for costumes; non-costumed runners get
    # pre-recorded finish-line goading, scored below any costume line.
    from scarebox.stage import SCORE_NO_COSTUME

    cfg = _cfg(budget_s=1.0, roast_non_costumes=True)
    judge = FakeJudge(verdict=_verdict(category="no_costume", costume="", is_costume=False))
    quip = FakeQuip(line="Warm encouragement!")
    library = FakeLibrary()
    orch, stage = _orch(judge, quip, FakeVoice(), library, cfg)

    record = asyncio.run(orch.handle_event(_event()))

    assert quip.quip_calls == 0
    assert record["path"] == "library"
    assert library.picked_categories == ["no_costume"]
    assert stage.queue[0].score == SCORE_NO_COSTUME


def test_single_call_skips_separate_judge_call():
    cfg = _cfg(budget_s=1.0, single_call=True)
    judge = FakeJudge(verdict=_verdict())  # must never be invoked in single_call mode
    quip = FakeQuip(line="Combined call line.", single_verdict=_verdict(category="ghost"))
    orch, stage = _orch(judge, quip, FakeVoice(), FakeLibrary(), cfg)

    record = asyncio.run(orch.handle_event(_event()))

    assert judge.calls == 0
    assert quip.single_calls == 1
    assert quip.quip_calls == 0
    assert record["path"] == "live"
    assert record["category"] == "ghost"
    assert record["ms"]["judge"] == record["ms"]["quip"]


def test_dropped_when_judge_fails_and_library_empty():
    cfg = _cfg(budget_s=1.0)
    judge = FakeJudge(verdict=None)
    library = FakeLibrary(line_text=None)
    orch, stage = _orch(judge, FakeQuip(), FakeVoice(), library, cfg)

    record = asyncio.run(orch.handle_event(_event()))

    assert record["path"] == "dropped"
    assert record["line"] is None
    assert stage.queue == []


def test_group_keeps_category_through_fallback():
    cfg = _cfg(budget_s=0.02)
    judge = FakeJudge(verdict=_verdict(category="group", group_size=3))  # fast -- quip is slow
    quip = FakeQuip(delay=0.3)
    library = FakeLibrary()
    orch, stage = _orch(judge, quip, FakeVoice(), library, cfg)

    record = asyncio.run(orch.handle_event(_event(group_size=3)))

    assert record["category"] == "group"
    assert library.picked_categories == ["group"]


def test_live_cap_sends_overflow_event_to_judged_library_line():
    # Two events at once with max_live_inflight=1: the first runs the full live pipeline, the
    # second gets a judge-only pass and a library line for its judged category (03-03 hub fix).
    cfg = _cfg(budget_s=1.0)
    cfg.orchestrator.max_live_inflight = 1
    judge = FakeJudge(verdict=_verdict())
    quip = FakeQuip(line="A pirate! Arrr.", delay=0.2)
    orch, stage = _orch(judge, quip, FakeVoice(), FakeLibrary(), cfg)

    async def both():
        return await asyncio.gather(orch.handle_event(_event()), orch.handle_event(_event()))

    first, second = asyncio.run(both())
    assert sorted([first["path"], second["path"]]) == ["library", "live"]
    capped = first if first["path"] == "library" else second
    assert capped["category"] == "pirate"
    assert capped["reject_reason"] == "live_cap"
    assert capped["reject_detail"] == "live_cap"
    assert quip.quip_calls == 1  # the capped event never reached quip
    assert orch._live_inflight == 0  # slot released after the live task finished


def test_fallback_passes_judged_costume_to_library():
    # Costume packs (2026-10-16): the library fallback gets the judge's costume string so a
    # judged pirate can hear a pirate-pack line. A no-costume verdict passes "" instead.
    cfg = _cfg(budget_s=0.2)
    judge = FakeJudge(verdict=_verdict(costume="Captain Hook"))
    quip = FakeQuip(line="A pirate! Arrr.", delay=1.0)  # misses the deadline -> library
    library = FakeLibrary()
    orch, _stage = _orch(judge, quip, FakeVoice(), library, cfg)
    result = asyncio.run(orch.handle_event(_event()))
    assert result["path"] == "library"
    assert library.picked_costumes == ["Captain Hook"]


# -- 04-04: Tuning wiring (aim control room's /settings panel) ------------------------------


def _tuning(**overrides) -> Tuning:
    base = dict(
        min_bbox_height=0.25,
        detector_min_confidence=0.5,
        stage_cooldown_s=0.0,
        orchestrator_budget_s=1.0,
        group_window_s=1.5,
        volume_pct=100.0,
        live_lines_enabled=True,
        roast_non_costumes=True,
    )
    base.update(overrides)
    return Tuning(**base)


def test_tuning_budget_s_overrides_cfg_budget():
    # `cfg.orchestrator.budget_s` says 0.02s (would miss the deadline); `tuning.
    # orchestrator_budget_s` says 1.0s -- the live line should make it when a Tuning is
    # wired in, proving the orchestrator reads the mutable value, not the frozen cfg one.
    cfg = _cfg(budget_s=0.02)
    tuning = _tuning(orchestrator_budget_s=1.0)
    judge = FakeJudge(verdict=_verdict())
    quip = FakeQuip(line="A pirate! Arrr.")
    orch, _stage = _orch(judge, quip, FakeVoice(), FakeLibrary(), cfg, tuning=tuning)

    record = asyncio.run(orch.handle_event(_event()))

    assert record["path"] == "live"


def test_tuning_roast_non_costumes_off_skips_quip():
    cfg = _cfg(budget_s=1.0, roast_non_costumes=True)  # cfg says roast -- tuning overrides it
    tuning = _tuning(roast_non_costumes=False)
    judge = FakeJudge(verdict=_verdict(is_costume=False, category="no_costume"))
    quip = FakeQuip(line="should never be used")
    orch, _stage = _orch(judge, quip, FakeVoice(), FakeLibrary(), cfg, tuning=tuning)

    record = asyncio.run(orch.handle_event(_event()))

    assert quip.quip_calls == 0
    assert record["path"] == "silent"


def test_judging_cap_skips_extra_crossings():
    # Busy mode: with max_judging=1, a second crossing while the first is still being
    # handled is skipped outright instead of piling another call onto the judge.
    cfg = _cfg(budget_s=1.0)
    tuning = _tuning(max_judging=1.0)
    judge = FakeJudge(verdict=_verdict(), delay=0.3)
    orch, _stage = _orch(judge, FakeQuip(line="A pirate! Arrr."), FakeVoice(), FakeLibrary(), cfg, tuning=tuning)

    async def both():
        return await asyncio.gather(orch.handle_event(_event()), orch.handle_event(_event()))

    first, second = asyncio.run(both())
    assert sorted([first["path"], second["path"]])[-1] == "skipped"
    skipped = first if first["path"] == "skipped" else second
    assert skipped["reject_reason"] == "busy"
    assert judge.calls == 1
    assert orch._judging == 0


def test_line_scores_rank_specific_over_category():
    from scarebox.stage import SCORE_COSTUME, SCORE_GENERIC, SCORE_SPECIFIC

    v = _verdict()
    assert Orchestrator._score(v, "pirate", live=True, pack=None) == SCORE_SPECIFIC
    assert Orchestrator._score(v, "pirate", live=False, pack="captain-hook") == SCORE_SPECIFIC
    assert Orchestrator._score(v, "pirate", live=False, pack=None) == SCORE_COSTUME
    assert Orchestrator._score(None, "mystery", live=False, pack=None) == SCORE_GENERIC


def test_eye_catching_rating_moves_the_line_score():
    # Martin's video (2026-10-20): pack matches went mostly to ordinary witches and skeletons
    # while inflatables timed out in the queue. A rated stand-out now outranks any ordinary
    # costume line; ears-and-a-hat drops below the busy-mode bar.
    from scarebox.stage import SCORE_COSTUME, SCORE_NO_COSTUME, SCORE_SPECIFIC

    standout = Orchestrator._score(_verdict(standout=3), "inflatable", live=False, pack=None)
    ordinary_pack = Orchestrator._score(_verdict(standout=2), "witch_wizard", live=False, pack="witch")
    assert standout > ordinary_pack == SCORE_SPECIFIC
    assert Orchestrator._score(_verdict(standout=1), "animal", live=False, pack="cat") == SCORE_NO_COSTUME
    assert Orchestrator._score(_verdict(standout=0), "pirate", live=False, pack=None) == SCORE_COSTUME


def test_tuning_live_lines_off_skips_quip_but_judges():
    cfg = _cfg(budget_s=1.0)
    tuning = _tuning(live_lines_enabled=False)
    judge = FakeJudge(verdict=_verdict())
    quip = FakeQuip(line="should never be used")
    library = FakeLibrary()
    orch, _stage = _orch(judge, quip, FakeVoice(), library, cfg, tuning=tuning)

    record = asyncio.run(orch.handle_event(_event()))

    assert judge.calls == 1  # judge still runs -- category/pack still work
    assert quip.quip_calls == 0  # but quip/voice never do
    assert record["path"] == "library"
    assert record["category"] == "pirate"
    assert record["reject_reason"] == "live_lines_off"


def test_blocked_text_surfaces_on_censor_reject():
    cfg = _cfg(budget_s=1.0)
    quip = FakeQuip(line=None, reason="guard-risk", detail="guard:violence")

    async def judge_and_quip_with_blocked(event, blocked):
        blocked["text"] = "a line the censor hated"
        blocked["reason"] = "guard:violence"
        return await quip.judge_and_quip_with_reason(event)

    async def quip_with_blocked(verdict, category, blocked):
        blocked["text"] = "a line the censor hated"
        blocked["reason"] = "guard:violence"
        return await quip.quip_with_reason(verdict, category)

    quip.judge_and_quip_with_blocked = judge_and_quip_with_blocked
    quip.quip_with_blocked = quip_with_blocked
    judge = FakeJudge(verdict=_verdict())
    orch, _stage = _orch(judge, quip, FakeVoice(), FakeLibrary(), cfg)

    record = asyncio.run(orch.handle_event(_event()))

    assert record["path"] == "library"
    assert record["blocked_text"] == "a line the censor hated"
    assert record["blocked_reason"] == "guard:violence"


class _PerCropJudge:
    """Returns a verdict chosen by the single crop's fill value, so a group's members can be
    told apart (spotlight test)."""

    def __init__(self, by_value):
        self.by_value = by_value
        self.calls = 0

    async def judge(self, event):
        self.calls += 1
        return self.by_value[int(event.crops[0][0, 0, 0])]


def test_spotlight_judges_each_member_and_names_the_standout():
    # Martin, 2026-10-19: in a group, pick the standout costume and give THEM the line
    # (a generic "royal court!" group line names nobody).
    class PackLibrary(FakeLibrary):
        def match_pack(self, costume, group=False):
            return "inflatable-trex" if "t. rex" in costume.lower() else None

    cfg = _cfg(budget_s=1.0)
    tuning = _tuning(spotlight=True, live_lines_enabled=False)
    judge = _PerCropJudge({
        1: _verdict(costume="none", category="no_costume", is_costume=False),
        2: _verdict(costume="witch hat", category="witch_wizard"),
        3: _verdict(costume="inflatable T. rex", category="inflatable"),
    })
    library = PackLibrary()
    orch, _stage = _orch(judge, FakeQuip(line="unused"), FakeVoice(), library, cfg, tuning=tuning)
    crops = [np.full((4, 4, 3), v, dtype=np.uint8) for v in (1, 2, 3)]
    ev = Event(t_trigger=time.monotonic(), track_ids=[10, 11, 12], group_size=3, crops=crops,
               crop_track_ids=[10, 11, 12])

    record = asyncio.run(orch.handle_event(ev))

    assert judge.calls == 3
    assert record["spotlight"]["track"] == 12
    assert record["category"] == "inflatable"  # the standout's own category, not "group"
    assert library.picked_costumes == ["inflatable T. rex"]
    assert len(record["spotlight"]["considered"]) == 3


def test_spotlight_prefers_a_rated_standout_over_a_pack_match():
    class PackLibrary(FakeLibrary):
        def match_pack(self, costume, group=False):
            return "witch" if "witch" in costume.lower() else None

    cfg = _cfg(budget_s=1.0)
    tuning = _tuning(spotlight=True, live_lines_enabled=False)
    judge = _PerCropJudge({
        1: _verdict(costume="witch", category="witch_wizard", standout=2),
        2: _verdict(costume="homemade crayon robot", category="homemade_box", standout=3),
    })
    orch, _stage = _orch(judge, FakeQuip(line="unused"), FakeVoice(), PackLibrary(), cfg, tuning=tuning)
    crops = [np.full((4, 4, 3), v, dtype=np.uint8) for v in (1, 2)]
    ev = Event(t_trigger=time.monotonic(), track_ids=[10, 11], group_size=2, crops=crops,
               crop_track_ids=[10, 11])

    record = asyncio.run(orch.handle_event(ev))

    assert record["spotlight"]["track"] == 11
    assert record["standout"] == 3
    assert [c["standout"] for c in record["spotlight"]["considered"]] == [2, 3]


def test_spotlight_off_keeps_the_group_line():
    cfg = _cfg(budget_s=1.0)
    tuning = _tuning(spotlight=False, live_lines_enabled=False)
    judge = FakeJudge(verdict=_verdict(costume="kids", category="group"))
    library = FakeLibrary()
    orch, _stage = _orch(judge, FakeQuip(line="unused"), FakeVoice(), library, cfg, tuning=tuning)
    ev = Event(t_trigger=time.monotonic(), track_ids=[1, 2], group_size=2,
               crops=[np.zeros((4, 4, 3), np.uint8)] * 2, crop_track_ids=[1, 2])
    record = asyncio.run(orch.handle_event(ev))
    assert judge.calls == 1 and record["spotlight"] is None and record["category"] == "group"
