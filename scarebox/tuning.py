"""tuning: a small mutable runtime settings object for the aim control room (04-04).

The frozen `config.Config` dataclasses are deliberately immutable (a race config is loaded
once and never mutated under your feet mid-run) -- but the aim page's Settings panel needs
values Martin can nudge WHILE the pipeline is live, on his phone, without restarting the
box. `Tuning` is the single mutable seam: it starts from `cfg`'s frozen defaults, optionally
overlaid by `runs/tuning.json` (a previous session's saved settings), and every consumer
(`Perceiver`, `Stage`, `Orchestrator`) reads its CURRENT value at use time through a plain
attribute lookup that `app.py` pushes a fresh value into whenever `/settings` changes it --
never a second copy of the same number living in two places.

This module is pure stdlib (json/dataclasses/pathlib) -- no cv2/torch -- so `tests/
test_tuning.py` runs on a bare venv, same bar as `aim.py`.

The trigger line is deliberately NOT a `Tuning` field: it already has its own drag-to-set
mechanism (`aim.py`'s `/trigger` route, persisted to `runs/trigger_line.json`), and that
stays exactly as it was -- `/settings` only reports the current line for display, read
through the same `get_trigger_line` callable `AimDeps` already carries.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TUNING_PATH = Path("runs/tuning.json")
# Martin, 2026-10-19: the USB webcam has no motor (no PTZ), so a digital crop is the only
# "zoom" available. `DEFAULT_CROP` is the full frame; `MIN_CROP_SIZE` stops a drag shrinking
# it into a sliver that starves the detector of pixels.
DEFAULT_CROP: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
MIN_CROP_SIZE = 0.2


def clamp_crop(crop) -> tuple[float, float, float, float]:
    """Pulls a possibly-malformed `[x0, y0, x1, y1]` back into a valid normalised crop
    rectangle: each coordinate in 0..1, `x0 < x1`/`y0 < y1` (swapped if given backwards), and
    each side at least `MIN_CROP_SIZE` (grown outward, then clipped back into frame)."""
    x0, y0, x1, y1 = (float(v) for v in crop)
    x0, x1 = sorted((min(1.0, max(0.0, x0)), min(1.0, max(0.0, x1))))
    y0, y1 = sorted((min(1.0, max(0.0, y0)), min(1.0, max(0.0, y1))))
    if x1 - x0 < MIN_CROP_SIZE:
        x1 = min(1.0, x0 + MIN_CROP_SIZE)
        x0 = max(0.0, x1 - MIN_CROP_SIZE)
    if y1 - y0 < MIN_CROP_SIZE:
        y1 = min(1.0, y0 + MIN_CROP_SIZE)
        y0 = max(0.0, y1 - MIN_CROP_SIZE)
    return (x0, y0, x1, y1)


def validate_crop(value) -> tuple[bool, str | None, tuple | None]:
    """`(ok, error, parsed_or_None)` for a `/settings` POST's `crop` value -- unlike the
    scalar settings, a bad crop is REJECTED, not silently clamped (clamping a backwards or
    degenerate rectangle could silently crop to somewhere the phone never dragged to)."""
    try:
        x0, y0, x1, y1 = (float(v) for v in value)
    except (TypeError, ValueError):
        return False, "crop: expected [x0, y0, x1, y1]", None
    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        return False, "crop: coordinates must be within 0..1", None
    if x1 <= x0 or y1 <= y0:
        return False, "crop: x1 must be > x0 and y1 must be > y0", None
    if (x1 - x0) < MIN_CROP_SIZE or (y1 - y0) < MIN_CROP_SIZE:
        return False, f"crop: minimum size is {MIN_CROP_SIZE}x{MIN_CROP_SIZE}", None
    return True, None, (x0, y0, x1, y1)


def apply_crop_to_frame(image: np.ndarray, crop) -> np.ndarray:
    """Slices `image` (the full-resolution BGR frame) to the normalised `crop` rectangle --
    the ONE place in the frame path this happens (`app.py`'s `main()` loop, before
    `perceiver.process_frame`), so detection/tracking/the trigger line/judge crops all work
    in the SAME cropped coordinate space. A cheap numpy view, no cv2 needed -- this makes
    the function testable on a bare venv even though `app.py` itself (cv2 at import time)
    is not."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = crop
    ix0, iy0 = int(round(x0 * w)), int(round(y0 * h))
    ix1, iy1 = int(round(x1 * w)), int(round(y1 * h))
    ix1 = max(ix1, ix0 + 1)
    iy1 = max(iy1, iy0 + 1)
    return image[iy0:iy1, ix0:ix1]


@dataclasses.dataclass(frozen=True)
class _Range:
    lo: float
    hi: float


# Every numeric setting `/settings` accepts, and the (lo, hi) its value is clamped into.
# This dict IS the schema: `/settings` GET reports it (as "ranges"), POST validates against
# its keys, `Tuning.clamp()` walks it.
RANGES: dict[str, _Range] = {
    "min_bbox_height": _Range(0.05, 0.9),
    "detector_min_confidence": _Range(0.1, 0.9),
    "stage_cooldown_s": _Range(0.0, 10.0),
    "orchestrator_budget_s": _Range(1.0, 8.0),
    "group_window_s": _Range(0.2, 5.0),
    "volume_pct": _Range(0.0, 100.0),
    # Busy mode (Martin, 2026-10-19, a hallway of kids): how many runners may be judged at
    # once. Extra crossings while the judge is this busy are skipped outright -- the speaker
    # couldn't have used them anyway, and every extra call slows the ones in flight.
    "max_judging": _Range(1.0, 4.0),
    # Crowd meter: this many crossings within 10 s switches busy mode on automatically.
    "crowd_crossings": _Range(2.0, 10.0),
    # Gap after each line while busy mode says it's crowded (the normal gap applies otherwise).
    # The timeline simulator on Martin's video: a 2.5 s -> 1.5 s gap names ~8% more of the
    # must-announce kids, because one speaker gets through more lines (2026-10-19).
    "busy_gap_s": _Range(0.0, 5.0),
    # How long after a runner crosses the trigger line their line may still start. Set it to
    # how far down the path the speaker is: 5 s was shorter than judging (~2 s) plus one line
    # and gap (~5.5 s), so a stand-out arriving mid-line always expired (2026-10-20 replay:
    # 71 of 123 rated stand-outs dropped this way).
    "max_line_age_s": _Range(3.0, 15.0),
}
# Boolean-valued settings -- no numeric range, just a type check.
#   busy_mode:    when several lines are waiting, only play distinctive-costume ones.
#   skip_generic: never play the "can't tell what you are" fallback lines.
#   spotlight:    judge a group's members one by one; the line goes to the standout costume.
BOOL_FIELDS: tuple[str, ...] = (
    "live_lines_enabled", "roast_non_costumes", "busy_mode", "skip_generic", "spotlight",
)
ALL_FIELDS: frozenset[str] = frozenset(RANGES) | frozenset(BOOL_FIELDS)


@dataclasses.dataclass
class Tuning:
    """Mutable. One instance lives for the life of the process; `app.py` owns it and pushes
    each field into the object that actually reads it (see module docstring)."""

    min_bbox_height: float
    detector_min_confidence: float
    stage_cooldown_s: float
    orchestrator_budget_s: float
    group_window_s: float
    volume_pct: float
    live_lines_enabled: bool
    roast_non_costumes: bool
    max_judging: float = 2.0
    crowd_crossings: float = 3.0
    busy_gap_s: float = 1.0
    max_line_age_s: float = 9.0
    spotlight: bool = True
    busy_mode: bool = True
    skip_generic: bool = True
    # Digital pan/zoom (Martin, 2026-10-19): normalised [x0, y0, x1, y1], default full frame.
    # A `tuple` default needs `default_factory` (dataclasses rejects a bare mutable/tuple
    # literal default only for list/dict/set -- a tuple is fine as a plain default, kept as
    # a factory anyway for symmetry with how `defaults_from_cfg` always constructs it fresh).
    crop: tuple[float, float, float, float] = dataclasses.field(
        default_factory=lambda: DEFAULT_CROP
    )

    # Orchestrator reads `tuning.budget_s`/`tuning.roast_non_costumes` -- same short names
    # `cfg.orchestrator.budget_s`/`cfg.quip.roast_non_costumes` always had -- so swapping a
    # frozen-cfg read for a tuning read at a call site is a one-word change, not a rename.
    @property
    def budget_s(self) -> float:
        return self.orchestrator_budget_s

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["crop"] = list(self.crop)
        return d

    def clamp(self) -> None:
        for name, r in RANGES.items():
            setattr(self, name, min(r.hi, max(r.lo, float(getattr(self, name)))))
        for name in BOOL_FIELDS:
            setattr(self, name, bool(getattr(self, name)))
        self.crop = clamp_crop(self.crop)


def defaults_from_cfg(cfg) -> Tuning:
    """The starting point before any `runs/tuning.json` override -- every value's home in
    `config/*.yaml` today. `cfg` only needs to duck-type `perceive`/`stage`/`orchestrator`/
    `quip` sections with these attributes -- tests pass a `types.SimpleNamespace`, same
    convention as `tests/test_orchestrator.py`."""
    return Tuning(
        min_bbox_height=float(cfg.perceive.min_bbox_height),
        detector_min_confidence=float(cfg.perceive.min_confidence),
        stage_cooldown_s=float(cfg.stage.cooldown_s),
        orchestrator_budget_s=float(cfg.orchestrator.budget_s),
        group_window_s=float(cfg.perceive.group_window_s),
        # Applied to the dongle at startup, so this default IS the race volume: 80% is what
        # Martin set to keep the PA's line-in from distorting (2026-10-19). 100 here would
        # quietly undo that on every restart.
        volume_pct=80.0,
        live_lines_enabled=True,
        roast_non_costumes=bool(cfg.quip.roast_non_costumes),
        max_judging=2.0,
        crowd_crossings=3.0,
        busy_gap_s=1.0,
        max_line_age_s=float(getattr(getattr(cfg, "orchestrator", None), "max_line_age_s", 9.0)),
        spotlight=True,
        busy_mode=True,
        skip_generic=True,
        crop=DEFAULT_CROP,
    )


def defaults_dict(cfg) -> dict[str, Any]:
    """`/settings`' "defaults" block -- what `POST /settings/reset` returns to."""
    return defaults_from_cfg(cfg).as_dict()


def load_tuning(cfg, path: str | Path = DEFAULT_TUNING_PATH) -> Tuning:
    """`defaults_from_cfg(cfg)`, overlaid by `path` if a previous session saved one. A
    corrupt/unreadable file is logged and ignored (falls back to cfg defaults) -- a bad
    `runs/tuning.json` must never stop the box from starting on race morning."""
    tuning = defaults_from_cfg(cfg)
    path = Path(path)
    if path.exists():
        try:
            saved = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("tuning.load_fail reason=%s", exc)
            saved = {}
        for key, value in saved.items():
            if key == "crop" and isinstance(value, (list, tuple)) and len(value) == 4:
                tuning.crop = tuple(float(v) for v in value)
            elif key in ALL_FIELDS:
                setattr(tuning, key, value)
    tuning.clamp()
    return tuning


def save_tuning(tuning: Tuning, path: str | Path = DEFAULT_TUNING_PATH) -> None:
    """Write-tmp-then-`os.replace` (same pattern as `aim.save_trigger_line`) so a reader
    never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(tuning.as_dict()))
    os.replace(tmp, path)


def validate_and_update(tuning: Tuning, patch: dict[str, Any]) -> tuple[bool, str | None]:
    """Validates `patch` (a `/settings` POST body) against `ALL_FIELDS`, clamps numeric
    values into range, and applies it to `tuning` IN PLACE. Returns `(ok, error)`; on a
    rejected patch (unknown key or wrong type) `tuning` is left completely unchanged --
    partial application of a bad request would leave the running pipeline in a state the
    phone UI never actually asked for. `crop` is validated (not clamped) separately -- see
    `validate_crop` -- since a backwards/degenerate rectangle is a client bug worth a 400,
    not something to silently reinterpret."""
    unknown = set(patch) - ALL_FIELDS - {"crop"}
    if unknown:
        return False, f"unknown setting: {sorted(unknown)[0]}"
    staged = dataclasses.replace(tuning)
    for key, value in patch.items():
        if key == "crop":
            ok, err, parsed = validate_crop(value)
            if not ok:
                return False, err
            staged.crop = parsed
        elif key in BOOL_FIELDS:
            if not isinstance(value, bool):
                return False, f"{key}: expected a boolean"
            setattr(staged, key, value)
        else:
            try:
                setattr(staged, key, float(value))
            except (TypeError, ValueError):
                return False, f"{key}: expected a number"
    staged.clamp()
    for key in patch:
        setattr(tuning, key, getattr(staged, key))
    return True, None


def apply_to_stage(tuning: Tuning, stage) -> None:
    """`Stage.try_play_next()` reads `self.cooldown_s` fresh every call, so this is the
    whole update."""
    stage.cooldown_s = tuning.stage_cooldown_s
    # Line priority (stage.SCORE_*): with more than one line waiting, only lines scoring at
    # least `crowd_min_score` are kept; a lone line needs `quiet_min_score`.
    stage.crowd_min_score = 2 if tuning.busy_mode else 0
    stage.crowd_crossings = int(tuning.crowd_crossings)
    stage.busy_gap_s = tuning.busy_gap_s if tuning.busy_mode else None
    stage.max_line_age_s = tuning.max_line_age_s
    stage.quiet_min_score = 1 if tuning.skip_generic else 0


def apply_to_orchestrator(tuning: Tuning, orchestrator) -> None:
    """Hands the orchestrator a live reference to `tuning` -- it reads `tuning.budget_s`/
    `tuning.roast_non_costumes`/`tuning.live_lines_enabled` at the start of each event
    instead of the frozen `cfg.orchestrator.budget_s`/`cfg.quip.roast_non_costumes` it used
    before 04-04 (see `orchestrator.Orchestrator.__init__`'s `tuning` parameter)."""
    orchestrator.tuning = tuning


def apply_to_perceiver(tuning: Tuning, perceiver) -> None:
    """`Perceiver` has no `__slots__`, and `process_frame`/`detector.detect` already read
    `self.min_bbox_height`/`self.detector.conf_threshold` fresh every frame -- plain
    attribute assignment is the whole update, no `perceive.py` edit needed for either.
    `group_window_s` is the exception: `perceive.py`'s own grouping docstring notes it is
    "no longer read" since the 2026-10-15 hub amendment (the only cap on event duration is
    `best_crop_frames` now) -- this still sets the attribute, for API completeness and so a
    future grouping revision has somewhere to read it from, but it is INERT today. Flagged
    here, not hidden."""
    perceiver.min_bbox_height = tuning.min_bbox_height
    perceiver.detector.conf_threshold = tuning.detector_min_confidence
    perceiver.group_window_s = tuning.group_window_s
