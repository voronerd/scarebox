"""Event, Verdict and LineResult dataclasses shared across the pipeline.

Spec references: §4.3 (component contracts -- `Event`, `Verdict`), §4.4 (event state machine,
the `events.jsonl` log line that `LineResult` mirrors), §6 (line library, `LineResult` is what
`library.pick()` returns).
"""

from __future__ import annotations

import dataclasses
import uuid

import numpy as np


def _short_id() -> str:
    """A short uuid4 (first 8 hex chars) -- unique enough for a few events a minute, and
    short enough to read in logs/aim-page tables."""
    return uuid.uuid4().hex[:8]


@dataclasses.dataclass
class Event:
    """One runner (or group) crossing the trigger line. Spec §4.3.

    `id`: short uuid4. `t_trigger`: monotonic seconds, when the triggering crossing was
    detected. `track_ids`: every tracker ID folded into this event (>1 for a group).
    `group_size`: len(track_ids), kept explicit since judge/quip/library all read it directly.
    `crops`: up to 3 best crops (BGR ndarrays), largest area first. `bbox_heights`: the
    triggering box height (fraction of frame height) per track_id, same order as `track_ids`.
    """

    id: str = dataclasses.field(default_factory=_short_id)
    t_trigger: float = 0.0
    track_ids: list[int] = dataclasses.field(default_factory=list)
    group_size: int = 1
    crops: list[np.ndarray] = dataclasses.field(default_factory=list)
    bbox_heights: list[float] = dataclasses.field(default_factory=list)
    # Which track each crop belongs to, same order as `crops` (spotlight, 2026-10-19: a group's
    # members are judged one by one and the standout gets the line, so a crop must map back
    # to its person).
    crop_track_ids: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Verdict:
    """judge's output. Spec §4.3. `category` must be one of `prompts.CATEGORIES`."""

    costume: str
    category: str
    confidence: float
    is_costume: bool
    group_size: int
    details: str = ""
    # 1-3, how eye-catching the costume is (prompts.STANDOUT_RUBRIC); 0 = not rated.
    standout: int = 0


@dataclasses.dataclass
class LineResult:
    """What `library.pick()` returns, and what each `runs/events.jsonl` line (spec §4.4)
    ultimately records as `path`/`line`. `path` is one of `live`, `library`, `dropped`.

    `wav` (added 02-02) is the rendered audio file `stage` should actually play -- kept
    separate from `text` since the library manifest's line and its rendered voice file are
    two different things, and a `--tts-placeholder` bench run substitutes a synthetic beep
    for this path without touching `text`/`category`.

    `pack` (04-04, aim control room): the costume-pack key (e.g. `"spider-man"`) when this
    line came from `Library.pick()`'s pack branch, `None` for a plain category line -- lets
    the aim page's event cards tell "Spider-Man, web-sling home!" apart from a generic
    category line without re-deriving it from the text.
    """

    text: str
    category: str
    path: str = "library"
    wav: str | None = None
    pack: str | None = None
