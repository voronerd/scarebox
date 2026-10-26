"""orchestrator: the live per-event state machine. Spec §4.4.

```
TRIGGERED -> JUDGING -> QUIPPING -> CENSORING (inside quip) -> VOICING -> QUEUED -> PLAYED
    |            |            |                                   |
    +------------+------------+-----------------------------------+--> (deadline fires first)
                                                                         |
                                     FALLBACK: library.pick(category or 'mystery') -> QUEUED
```

`Orchestrator(judge, quip, voice, library, stage, cfg, clock)` runs the live pipeline for one
`Event` as a background `asyncio.Task` and races it against `event.t_trigger + cfg.orchestrator
.budget_s` (never a hardcoded number -- the drift check in the 03-03 plan greps for one). If the
live task isn't done by the deadline, it's cancelled and `library.pick()` supplies the fallback
line, filed under whatever category `judge` already returned (kept even when `quip`/`voice` ran
out of time) -- a generic `mystery` line only when `judge` itself never answered.

Cancellation is cooperative: a task awaiting `voice.speak()`'s `loop.run_in_executor` GPU call
can't actually be interrupted once the executor call is running in its own thread, so a
cancelled live task occasionally still produces a real result slightly after `handle_event` has
already committed to the fallback. That case is "late": discarded, logged, never enqueued twice
-- see `_on_live_done`.

Non-costume runners (`verdict.is_costume is False`) skip `quip` entirely when `quip.
roast_non_costumes` is `false`: the roast persona never gets asked to write a line for someone
it shouldn't be heckling, so this goes straight to the library's `no_costume` category, same
handling as a deadline fallback.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from collections.abc import AsyncIterator

from scarebox.config import Config
from scarebox.events import Event, Verdict
from scarebox.library import Library, resolve_category
from scarebox.quip import Quip
from scarebox.judge import Judge
from scarebox.stage import PlayItem, Stage
from scarebox.voice import Voice

logger = logging.getLogger(__name__)

# How often `handle_event` polls the live task against the deadline. Small enough that the
# deadline is enforced to within a few ms; real enough (not a fake-clock tick) that it costs
# almost nothing in a fast unit test with a small `budget_s`.
_POLL_S = 0.005


@dataclasses.dataclass
class LiveOutcome:
    """What `Orchestrator._run_live` produces for one event -- never written straight to the
    events log itself; `handle_event` folds it into the spec §4.4 record."""

    verdict: Verdict | None
    category: str
    costume: str
    confidence: float
    line: str | None
    # "pass" once a line has actually cleared censor (inside quip); `None` otherwise -- never
    # anything else, so `path == "live"` always implies `censor == "pass"` (spec §1 success
    # criterion 4, the phase's drift check).
    censor: str | None
    # One of deterministic/guard-risk/guard-timeout/parse/quip-none (quip.py's
    # `_censor_bucket`), or `None` when a live line was produced or quip was never attempted.
    reject_reason: str | None
    # The specific reason underneath the bucket above: the censor's own `Reject.reason`
    # string (e.g. "blocklist:fat", "guard:violence", "guard_timeout"), the pre-censor filter
    # that missed ("costume_noun_miss", "too_long"), or the call failure ("timeout",
    # "no_crops", ...). `None` exactly when `reject_reason` is `None` -- see quip.py's
    # `_survivor`/`_censor_bucket` comment for the full mapping.
    reject_detail: str | None
    # True for the roast_non_costumes=false short-circuit -- `line` is always None here even
    # if a combined judge_and_quip call happened to generate one, because it must never be
    # spoken (hub amendment: never roast a non-costumed runner when the config says not to).
    skip_quip: bool
    audio: AsyncIterator | None
    ms: dict[str, float]
    # 04-04 (aim control room): the actual text of a live candidate that reached the censor
    # and was rejected, and the censor's own reason for it -- `None`/`None` unless that
    # specific thing happened this event (see `quip.Quip._survivor`'s `blocked` param).
    # Never written to `events.jsonl`'s `line` field; only the aim page's RAM-only event
    # card surfaces it, and never in the live view.
    blocked_text: str | None = None
    blocked_reason: str | None = None


@dataclasses.dataclass
class _Progress:
    """Mutable, written INCREMENTALLY by `_run_live` as soon as each stage finishes --
    `handle_event` reads this when the deadline fires mid-flight (quip/voice still running),
    since `_run_live`'s own return value doesn't exist yet at that point. This is how `judge`'s
    category survives a `quip`/`voice` timeout (spec §4.4: "The category from judge is kept
    even if quip or voice is late")."""

    verdict: Verdict | None = None
    category: str = "mystery"
    # Spotlight (2026-10-19): set when a group was judged member by member --
    # {"track": the chosen person's track id, "considered": [{track, costume, category}]}.
    spotlight: dict | None = None


async def _prepend_chunk(first, rest: AsyncIterator) -> AsyncIterator:
    """Puts an already-fetched first chunk back in front of the remaining async iterator --
    `_run_live` has to consume one chunk from `voice.speak()` itself (to know a line is
    actually voicable before the deadline), but `stage.Stage` needs the WHOLE stream."""
    yield first
    async for chunk in rest:
        yield chunk


class Orchestrator:
    def __init__(
        self,
        judge: Judge,
        quip: Quip,
        voice: Voice,
        library: Library,
        stage: Stage,
        cfg: Config,
        clock=time.monotonic,
        tuning=None,
    ) -> None:
        self.judge = judge
        self.quip = quip
        self.voice = voice
        self.library = library
        self.stage = stage
        self.cfg = cfg
        self._clock = clock
        # 04-04: a `tuning.Tuning` instance (mutable; `app.py`'s `/settings` route writes
        # straight into it) overriding `cfg.orchestrator.budget_s` / `cfg.quip
        # .roast_non_costumes` / a new live-lines-on-off switch cfg never had. `None` (the
        # default, and every pre-04-04 caller/test) keeps the original frozen-cfg behaviour
        # exactly -- see `_budget_s`/`_roast_non_costumes`/`_live_lines_enabled` below.
        self.tuning = tuning
        # event.id -> "live" | "fallback", set the moment `handle_event` commits to a path,
        # read (and popped) by `_on_live_done` to tell a legitimate on-time result apart from
        # a stray one arriving after the fallback already shipped.
        self._resolved: dict[str, str] = {}
        # Live pipelines (quip + censor + voice) currently running. Above the cap an event gets
        # a judge-only pass and a category-matched library line: on a busy finish line the GPU
        # can't voice every event in time anyway, and running them all starves the guard
        # (03-03 ladder: 76% of censor rejects were guard timeouts under that contention).
        self._live_inflight = 0
        # Runners being judged right now (busy mode, 2026-10-19). Every crossing used to get its
        # own judge call with no cap, so a hallway of kids queued calls up and every verdict
        # arrived late. Above `_max_judging()` a crossing is skipped outright.
        self._judging = 0

    def _max_judging(self) -> int | None:
        """None = no cap (no Tuning applied: tests, bench)."""
        return int(self.tuning.max_judging) if self.tuning is not None else None

    def _budget_s(self) -> float:
        return self.tuning.budget_s if self.tuning is not None else self.cfg.orchestrator.budget_s

    def _roast_non_costumes(self) -> bool:
        if self.tuning is not None:
            return self.tuning.roast_non_costumes
        return self.cfg.quip.roast_non_costumes

    def _live_lines_enabled(self) -> bool:
        return self.tuning is None or self.tuning.live_lines_enabled

    def _spotlight_on(self) -> bool:
        return self.tuning is not None and getattr(self.tuning, "spotlight", False)

    def _standout_rank(self, verdict) -> tuple:
        """Higher = more worth a line. Any clear costume beats a vague or missing one; among
        costumes the judge's eye-catching rating decides, and having a costume pack only
        breaks ties (packs cover common costumes like witches too)."""
        if verdict is None:
            return (0, 0, 0)
        if not verdict.is_costume or verdict.category in ("no_costume", "mystery"):
            return (1, 0, 0)
        match = getattr(self.library, "match_pack", None)
        pack = match(verdict.costume) if match else None
        return (2, getattr(verdict, "standout", 0), 1 if pack else 0)

    async def _spotlight_judge(self, event: Event, progress: "_Progress"):
        """Spotlight (Martin, 2026-10-19: "looking at a group and selecting the standout one").
        Judges each group member's crop on its own, concurrently, and returns the standout's
        verdict; a group line ("an entire royal court!") names nobody. Ties go to the earlier
        crop, which is the larger, sharper one (perceive orders crops that way)."""
        track_ids = event.crop_track_ids or event.track_ids
        subs = [
            dataclasses.replace(event, crops=[crop], group_size=1, track_ids=[tid], crop_track_ids=[tid])
            for crop, tid in zip(event.crops, track_ids)
        ]
        results = await asyncio.gather(*(self.judge.judge(s) for s in subs), return_exceptions=True)
        verdicts = [None if isinstance(r, BaseException) else r for r in results]
        considered = [
            {"track": tid, "costume": getattr(v, "costume", ""), "category": getattr(v, "category", None),
             "standout": getattr(v, "standout", 0)}
            for tid, v in zip(track_ids, verdicts)
        ]
        best = max(range(len(verdicts)), key=lambda i: (self._standout_rank(verdicts[i]), -i))
        if verdicts[best] is None:
            return None
        progress.spotlight = {"track": track_ids[best], "considered": considered}
        logger.info(
            "orchestrator.spotlight event=%s chose=%s from=%d", event.id, track_ids[best], len(verdicts)
        )
        return dataclasses.replace(verdicts[best], group_size=1)

    async def _run_live(
        self,
        event: Event,
        progress: _Progress,
        judge_only: bool = False,
        skip_reason: str = "live_cap",
    ) -> LiveOutcome:
        ms: dict[str, float] = {}
        clock = self._clock
        t_start = clock()

        line: str | None = None
        reject_reason: str | None = None
        reject_detail: str | None = None
        blocked: dict = {}  # 04-04: filled by `quip`'s `*_with_blocked` calls below

        if self.cfg.quip.single_call and not judge_only:
            verdict, line, reject_reason, reject_detail = await self.quip.judge_and_quip_with_blocked(
                event, blocked
            )
            call_ms = round((clock() - t_start) * 1000.0, 1)
            # One round-trip did both jobs -- there's no separate judge-only timing to report.
            ms["judge"] = call_ms
            ms["quip"] = call_ms
        elif self._spotlight_on() and event.group_size > 1 and len(event.crops) > 1:
            verdict = await self._spotlight_judge(event, progress)
            ms["judge"] = round((clock() - t_start) * 1000.0, 1)
        else:
            verdict = await self.judge.judge(event)
            ms["judge"] = round((clock() - t_start) * 1000.0, 1)

        if verdict is None:
            return LiveOutcome(
                verdict=None,
                category="mystery",
                costume="",
                confidence=0.0,
                line=None,
                censor=None,
                reject_reason=None,
                reject_detail=None,
                skip_quip=False,
                audio=None,
                ms=ms,
                blocked_text=blocked.get("text"),
                blocked_reason=blocked.get("reason"),
            )

        if progress.spotlight is not None:
            # The line is about the standout member, so it keeps their own category instead of
            # the generic "group" one.
            category = verdict.category
        else:
            category = resolve_category(verdict.category, event.group_size)
        # Publish now, before quip/voice run -- if the deadline fires while they're still in
        # flight, `handle_event` reads this instead of waiting for `_run_live` to return.
        progress.verdict = verdict
        progress.category = category
        if judge_only:
            # `skip_reason` is "live_cap" (the GPU-contention cap, unchanged default) or
            # "live_lines_off" (04-04's Settings toggle) -- `handle_event` picks which.
            return LiveOutcome(
                verdict=verdict,
                category=category,
                costume=getattr(verdict, "costume", ""),
                confidence=getattr(verdict, "confidence", 0.0),
                line=None,
                censor=None,
                reject_reason=skip_reason,
                reject_detail=skip_reason,
                skip_quip=False,
                audio=None,
                ms=ms,
                blocked_text=blocked.get("text"),
                blocked_reason=blocked.get("reason"),
            )

        if not verdict.is_costume:
            # No costume never gets the live writer (busy mode, 2026-10-19): the one live slot
            # goes to costumes, and a pre-recorded goading line is just as good here.
            # skip_quip=True means "play nothing" (heckling non-costumes switched off).
            roast = self._roast_non_costumes()
            return LiveOutcome(
                verdict=verdict,
                category=category,
                costume=verdict.costume,
                confidence=verdict.confidence,
                line=None,
                censor=None,
                reject_reason="no_costume_library" if roast else None,
                reject_detail="no_costume_library" if roast else None,
                skip_quip=not roast,
                audio=None,
                ms=ms,
                blocked_text=blocked.get("text"),
                blocked_reason=blocked.get("reason"),
            )

        if not self.cfg.quip.single_call:
            t_quip = clock()
            line, reject_reason, reject_detail = await self.quip.quip_with_blocked(
                verdict, category, blocked
            )
            ms["quip"] = round((clock() - t_quip) * 1000.0, 1)

        if line is None:
            return LiveOutcome(
                verdict=verdict,
                category=category,
                costume=verdict.costume,
                confidence=verdict.confidence,
                line=None,
                censor=None,
                reject_reason=reject_reason,
                reject_detail=reject_detail,
                skip_quip=False,
                audio=None,
                ms=ms,
                blocked_text=blocked.get("text"),
                blocked_reason=blocked.get("reason"),
            )

        t_voice = clock()
        chunk_iter = self.voice.speak(line)
        try:
            first_chunk = await chunk_iter.__anext__()
        except StopAsyncIteration:
            logger.warning("orchestrator.voice_empty event=%s", event.id)
            return LiveOutcome(
                verdict=verdict,
                category=category,
                costume=verdict.costume,
                confidence=verdict.confidence,
                line=None,
                censor=None,
                reject_reason="quip-none",
                reject_detail="voice_empty",
                skip_quip=False,
                audio=None,
                ms=ms,
                blocked_text=blocked.get("text"),
                blocked_reason=blocked.get("reason"),
            )
        # Timed from the voice call starting to the FIRST chunk existing, per spec §4.4's
        # `ms.voice_first_audio` -- not the whole render (that happens inside `stage` after
        # this event is already queued, off the deadline clock; see `_prepend_chunk`/stage.py).
        ms["voice_first_audio"] = round((clock() - t_voice) * 1000.0, 1)

        return LiveOutcome(
            verdict=verdict,
            category=category,
            costume=verdict.costume,
            confidence=verdict.confidence,
            line=line,
            censor="pass",
            reject_reason=None,
            reject_detail=None,
            skip_quip=False,
            audio=_prepend_chunk(first_chunk, chunk_iter),
            ms=ms,
            blocked_text=blocked.get("text"),
            blocked_reason=blocked.get("reason"),
        )

    def _on_live_done(self, event_id: str, task: asyncio.Task, events_log=None) -> None:
        """Fires exactly once per event, whenever the live task finishes -- cleanly, with an
        exception, or cancelled. Only logs/discards a `late` result when `handle_event`
        already took the fallback branch for this event (recorded in `self._resolved`);
        otherwise this is the SAME result `handle_event` already consumed, or a clean cancel
        with nothing produced, and there's nothing to do beyond tidying up the dict entry.

        Writes a SEPARATE `{"id", "late": true}` line to `events_log` (not a rewrite of the
        event's own record, which is already on disk by the time this fires) so `scripts/
        latency_report.py` can count lates from `runs/events.jsonl` alone.
        """
        try:
            if task.cancelled():
                return
            if self._resolved.get(event_id) != "fallback":
                return
            if task.exception() is not None:
                return
            outcome: LiveOutcome | None = task.result()
            if outcome is not None and outcome.line is not None:
                logger.info("orchestrator.late event=%s discarded", event_id)
                if events_log is not None:
                    events_log.write(json.dumps({"id": event_id, "late": True}) + "\n")
                    events_log.flush()
        finally:
            self._resolved.pop(event_id, None)

    def _release_live_slot(self) -> None:
        self._live_inflight = max(self._live_inflight - 1, 0)

    def _skip_busy(self, event: Event, events_log) -> dict:
        record = {
            "id": event.id,
            "t_trigger": event.t_trigger,
            "group_size": event.group_size,
            "category": None,
            "costume": "",
            "confidence": 0.0,
            "is_costume": False,
            "path": "skipped",
            "pack": None,
            "line": None,
            "censor": None,
            "reject_reason": "busy",
            "reject_detail": f"{self._judging} already being judged",
            "ms": {"total": 0.0},
        }
        logger.info("orchestrator.skip event=%s reason=busy judging=%d", event.id, self._judging)
        if events_log is not None:
            events_log.write(json.dumps(record) + "\n")
            events_log.flush()
        return record

    @staticmethod
    def _score(verdict, category: str | None, live: bool, pack: str | None) -> int:
        """How much the line is about THIS runner's costume (stage.SCORE_*), raised for an
        eye-catching costume and lowered for a barely-there one, so in a crowd the speaker's
        time goes to the costumes people point at."""
        from scarebox.stage import (
            SCORE_COSTUME, SCORE_GENERIC, SCORE_NO_COSTUME, SCORE_SPECIFIC, SCORE_STANDOUT_BONUS,
        )

        if verdict is None or category in (None, "mystery"):
            return SCORE_GENERIC
        if not verdict.is_costume or category == "no_costume":
            return SCORE_NO_COSTUME
        standout = getattr(verdict, "standout", 0)
        if standout == 1:
            return SCORE_NO_COSTUME  # ears and a hat: below the busy-mode bar
        base = SCORE_SPECIFIC if (live or pack) else SCORE_COSTUME
        return base + (SCORE_STANDOUT_BONUS if standout == 3 else 0)

    async def handle_event(self, event: Event, events_log=None) -> dict:
        # Every crossing feeds the crowd meter, including ones about to be skipped.
        note = getattr(self.stage, "note_crossing", None)
        if note is not None:
            note(event.t_trigger, people=event.group_size)
        cap = self._max_judging()
        if cap is not None and self._judging >= cap:
            return self._skip_busy(event, events_log)
        self._judging += 1
        try:
            return await self._handle_event(event, events_log)
        finally:
            self._judging = max(self._judging - 1, 0)

    async def _handle_event(self, event: Event, events_log=None) -> dict:
        t0 = self._clock()
        deadline = event.t_trigger + self._budget_s()
        progress = _Progress()

        # 04-04: the Settings "live lines" toggle (judge still runs for category/pack, but
        # quip/censor/voice never do) is handled the same way the pre-existing GPU-contention
        # cap is -- a judge-only `_run_live` call -- just with its own `reject_reason` so a
        # dashboard can tell "the lever was off" apart from "the GPU was busy".
        live_lines_off = not self._live_lines_enabled()
        slot_available = self._live_inflight < getattr(self.cfg.orchestrator, "max_live_inflight", 1)
        full_live = slot_available and not live_lines_off
        if full_live:
            self._live_inflight += 1
        skip_reason = "live_lines_off" if live_lines_off else "live_cap"
        live_task = asyncio.create_task(
            self._run_live(event, progress, judge_only=not full_live, skip_reason=skip_reason)
        )
        if full_live:
            live_task.add_done_callback(lambda _t: self._release_live_slot())
        live_task.add_done_callback(lambda t: self._on_live_done(event.id, t, events_log))

        while not live_task.done() and self._clock() < deadline:
            await asyncio.sleep(_POLL_S)

        outcome: LiveOutcome | None
        if live_task.done():
            self._resolved[event.id] = "live"
            try:
                outcome = live_task.result()
            except Exception as exc:  # noqa: BLE001 -- any live-pipeline exception -> fallback
                logger.warning("orchestrator.live_exception event=%s reason=%s", event.id, exc)
                outcome = None
        else:
            self._resolved[event.id] = "fallback"
            live_task.cancel()
            outcome = None

        # `judge`'s category/verdict survive even when `quip`/`voice` never finished in time
        # (spec §4.4) -- `outcome` only exists once `_run_live` has fully returned, but
        # `progress` was updated the moment `judge` itself resolved, inside the still-running
        # task.
        category = outcome.category if outcome is not None else progress.category
        verdict = outcome.verdict if outcome is not None else progress.verdict
        ms = dict(outcome.ms) if outcome is not None else {}

        path = "dropped"
        line_text: str | None = None
        censor: str | None = None
        reject_reason = outcome.reject_reason if outcome is not None else None
        reject_detail = outcome.reject_detail if outcome is not None else None

        pack: str | None = None
        line_score: int | None = None
        live_ready = outcome is not None and outcome.line is not None and not outcome.skip_quip
        if live_ready:
            line_score = self._score(verdict, category, live=True, pack=None)
            self.stage.enqueue(
                PlayItem(
                    event_id=event.id,
                    t_trigger=event.t_trigger,
                    audio=outcome.audio,
                    kind="live",
                    score=line_score,
                )
            )
            path = "live"
            line_text = outcome.line
            censor = "pass"
        elif outcome is not None and outcome.skip_quip:
            # Non-costumed runner with heckling non-costumes switched off: say nothing. (This
            # used to fall through to a no_costume library line, ignoring the switch.)
            path = "silent"
        else:
            fallback_category = category or "mystery"
            # Costume packs: a judged "Spider-Man" gets a Spider-Man line when one is playable.
            costume = verdict.costume if verdict is not None and verdict.is_costume else ""
            crowded = bool(
                getattr(self.tuning, "busy_mode", False) and getattr(self.stage, "crowded", lambda: False)()
            )
            picked = self.library.pick(fallback_category, costume=costume, prefer_short=crowded)
            if picked is not None:
                line_score = self._score(verdict, category, live=False, pack=picked.pack)
                self.stage.enqueue(
                    PlayItem(
                        event_id=event.id,
                        t_trigger=event.t_trigger,
                        audio=picked.wav,
                        kind="library",
                        score=line_score,
                    )
                )
                path = "library"
                line_text = picked.text
                pack = picked.pack
            else:
                logger.warning(
                    "orchestrator.dropped event=%s category=%s", event.id, fallback_category
                )

        ms["total"] = round((self._clock() - t0) * 1000.0, 1)
        record = {
            "id": event.id,
            "t_trigger": event.t_trigger,
            "group_size": event.group_size,
            "category": category,
            "costume": verdict.costume if verdict else "",
            "confidence": verdict.confidence if verdict else 0.0,
            "is_costume": bool(verdict.is_costume) if verdict is not None else False,
            "path": path,
            "pack": pack,
            "score": line_score,  # stage.SCORE_*: the line's priority in busy mode
            "standout": getattr(verdict, "standout", 0) if verdict is not None else 0,
            # Spotlight: which group member the line is about, and what each one looked like.
            "spotlight": progress.spotlight,
            "line": line_text,
            "censor": censor,
            "reject_reason": reject_reason,
            "reject_detail": reject_detail,
            # 04-04: the aim page's BLOCKED card field -- the actual censor-rejected line
            # text/reason, never `events.jsonl`'s own `line` (what actually got spoken).
            "blocked_text": outcome.blocked_text if outcome is not None else None,
            "blocked_reason": outcome.blocked_reason if outcome is not None else None,
            "ms": ms,
        }
        if events_log is not None:
            events_log.write(json.dumps(record) + "\n")
            events_log.flush()
        return record
