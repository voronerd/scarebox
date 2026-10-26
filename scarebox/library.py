"""library: the pre-baked line picker. Spec §6, §4.3.

`Library(manifest_path, require_approved)` loads `library/manifest.json`
(`[{id, category, text, wav, status: approved|rejected|pending, reviewed_at}]`).
`pick(category, exclude_last=10)` returns a `LineResult` chosen at random among lines in
`category` that are `approved` (or `approved`/`pending` when `require_approved` is False --
bench-only, spec §6.3) AND have a `wav` file that actually exists on disk, excluding the
last `exclude_last` ids played. An empty/unknown `category` falls back to `mystery`;
if `mystery` itself has nothing playable, `pick` returns `None` (the orchestrator logs the
event as `dropped` rather than play a wrong line).

Constructing with `require_approved=True` raises `LibraryError` naming the pending lines --
the race config refuses to start on an unvetted library (spec §6.3 CRITICAL rule). Rejected
lines are allowed in the manifest (they're a vetting record) but are never eligible.

Costume packs (2026-10-16): lines carrying a `pack` key (e.g. `"pack": "spider-man"`) name one
specific costume. `pick(category, costume=...)` matches the judge's costume string against the
packs file's aliases (whole-word, longest alias wins) and prefers that pack's lines, falling
back to the category when nothing matches or the pack has nothing playable.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from scarebox.events import LineResult

_DEFAULT_MYSTERY_CATEGORY = "mystery"


class LibraryError(Exception):
    """Raised when a `require_approved=True` library has pending lines."""


# Words the judge wraps around a costume name that carry no identity ("a spider-man costume").
_FILLER = re.compile(r"\b(?:a|an|the|costume|costumes|outfit|dressed|as|wearing|in|with)\b")


def normalise_costume(text: str) -> str:
    """'A Spider-Man costume!' -> 'spider man'. Hyphens and punctuation become spaces so
    'spider-man', 'spider man' and 'Spider-Man' all compare equal."""
    text = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    text = _FILLER.sub(" ", text)
    return " ".join(text.split())


class PackMatcher:
    """Alias table from the packs file: `[{key, category, aliases: [...]}, ...]`."""

    def __init__(self, packs: list[dict]) -> None:
        self._aliases: list[tuple[str, str, str]] = []  # (normalised alias, key, category)
        # 04-04: key -> its human-facing `display` name (e.g. "spider-man" -> "Spider-Man"),
        # for the aim page's box label/event-card pack badge -- Martin reads display names,
        # not alias-table keys.
        self._display: dict[str, str] = {}
        for pack in packs:
            self._display[pack["key"]] = pack.get("display") or pack["key"]
            for alias in [pack.get("display", ""), *pack.get("aliases", [])]:
                norm = normalise_costume(alias)
                if norm:
                    self._aliases.append((norm, pack["key"], pack["category"]))
        # Longest alias first, so "inflatable t rex" beats "t rex" and "dinosaur".
        self._aliases.sort(key=lambda a: len(a[0]), reverse=True)

    def display(self, key: str) -> str | None:
        return self._display.get(key)

    @classmethod
    def from_file(cls, path: str | Path) -> "PackMatcher":
        return cls(json.loads(Path(path).read_text()))

    def match(self, costume: str, group: bool = False) -> str | None:
        """Pack key for a judge costume string, or None. A group only matches group packs
        (a squad of superheroes isn't one Spider-Man)."""
        text = f" {normalise_costume(costume)} "
        if not text.strip():
            return None
        for alias, key, category in self._aliases:
            if group and category != "group":
                continue
            if not group and category == "group":
                continue
            if f" {alias} " in text:
                return key
        return None


def resolve_category(judge_category: str, group_size: int) -> str:
    """Hub amendment, 02-02 Task 3 (judge calibration decision): a detected group
    (`group_size >= 2`) always gets the `group` library category, regardless of the judge's
    per-costume read. The judge's category call is close to coin-flip-or-worse on
    multi-person crops -- on the silver set it described the first/biggest person instead of
    recognising a group at all (100% wrong-category on `group`) -- and a `group` line is
    written to fit any mixed-costume cluster, so it's always the right choice once two or
    more people are in frame."""
    return "group" if group_size >= 2 else judge_category


class Library:
    def __init__(
        self,
        manifest_path: str | Path,
        require_approved: bool,
        require_wav: bool = True,
        packs_path: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.require_approved = require_approved
        # Bench-only (02-02 Task 3, `--tts-placeholder`): the 02-02 seed manifest ships every
        # line with `wav: null` (02-03 bakes/renders the real audio). With `require_wav=True`
        # (the production default) no line is ever eligible until it has a real wav on disk --
        # correct for the race, but it means a placeholder-beep bench run before 02-03 would
        # see every event "dropped" with nothing to measure. `require_wav=False` drops the
        # on-disk check so pending text-only lines are still pick-able; `app.py` substitutes
        # the placeholder beep for the (possibly None) `wav` at enqueue time either way.
        self.require_wav = require_wav
        self._lines: list[dict] = json.loads(self.manifest_path.read_text())
        if require_approved:
            not_approved = [
                line["id"]
                for line in self._lines
                if line.get("status") not in ("approved", "rejected")
            ]
            if not_approved:
                raise LibraryError(
                    f"library.require_approved is true but {len(not_approved)} line(s) are "
                    f"not approved: {not_approved[: min(5, len(not_approved))]}"
                )
        self._recent: list[str] = []
        self._packs: PackMatcher | None = None
        if packs_path and Path(packs_path).exists():
            self._packs = PackMatcher.from_file(packs_path)

    def _eligible_statuses(self) -> frozenset[str]:
        return frozenset({"approved"}) if self.require_approved else frozenset({"approved", "pending"})

    def _candidates(self, category: str, exclude: set[str], pack: str | None = None) -> list[dict]:
        statuses = self._eligible_statuses()
        out = []
        for line in self._lines:
            if pack is not None:
                if line.get("pack") != pack:
                    continue
            elif line["category"] != category or line.get("pack"):
                # A pack line names one specific costume, so it never stands in for its
                # category: Batman must not hear "Spider-Man, web-sling home!".
                continue
            if line.get("status") not in statuses:
                continue
            if self.require_wav:
                wav = line.get("wav")
                if not wav or not Path(wav).exists():
                    continue
            if line["id"] in exclude:
                continue
            out.append(line)
        return out

    def match_pack(self, costume: str, group: bool = False) -> str | None:
        return self._packs.match(costume, group=group) if self._packs else None

    def pack_display(self, key: str | None) -> str | None:
        """04-04: the pack's human-facing `display` name (e.g. "Spider-Man" for key
        "spider-man") -- the aim page's box label/event-card badge show this, never the
        alias-table key. `None` for `key=None` or an unknown key."""
        if key is None or self._packs is None:
            return None
        return self._packs.display(key)

    @staticmethod
    def _shorter_half(lines: list[dict]) -> list[dict]:
        """Busy mode: the shorter half of the candidates by word count (a shorter line frees
        the speaker sooner for the next standout costume)."""
        if len(lines) < 4:
            return lines
        ranked = sorted(lines, key=lambda x: len(x["text"].split()))
        return ranked[: max(2, len(ranked) // 2)]

    def pick(
        self, category: str, exclude_last: int = 30, costume: str = "", prefer_short: bool = False
    ) -> LineResult | None:
        cat = category or _DEFAULT_MYSTERY_CATEGORY
        exclude = set(self._recent[-exclude_last:]) if exclude_last else set()

        pack = self.match_pack(costume, group=cat == "group") if costume else None
        if pack is not None:
            # Only fresh pack lines: a repeated "Spider-Man!" line is worse than a fresh
            # superhero one, so an exhausted pack falls through to the category.
            pack_lines = self._candidates(cat, exclude, pack=pack)
            if pack_lines:
                chosen = random.choice(self._shorter_half(pack_lines) if prefer_short else pack_lines)
                self._recent.append(chosen["id"])
                return LineResult(
                    text=chosen["text"],
                    category=chosen["category"],
                    path="library",
                    wav=chosen["wav"],
                    pack=pack,
                )

        candidates = self._candidates(cat, exclude)
        if not candidates:
            # Retry ignoring the no-repeat exclusion before giving up on this category --
            # a thin category (or everything excluded) shouldn't force a `mystery` fallback.
            candidates = self._candidates(cat, set())
        if not candidates and cat != _DEFAULT_MYSTERY_CATEGORY:
            return self.pick(_DEFAULT_MYSTERY_CATEGORY, exclude_last=exclude_last)
        if not candidates:
            return None

        chosen = random.choice(self._shorter_half(candidates) if prefer_short else candidates)
        self._recent.append(chosen["id"])
        return LineResult(
            text=chosen["text"], category=chosen["category"], path="library", wav=chosen["wav"]
        )

    def categories(self) -> list[str]:
        """Distinct categories present in the manifest, sorted. `aim.py`'s `/test` button
        (04-02) picks a random one from these -- it doesn't care which category, only that
        `pick()` finds SOMETHING playable to confirm the speaker path end to end."""
        return sorted({line["category"] for line in self._lines})

    def get(self, line_id: str) -> LineResult | None:
        """Fetch ONE specific manifest line by id, bypassing `pick()`'s random-choice and
        `require_approved` gates -- `selftest.py` needs the EXACT named meta line
        (`meta-ready`/`meta-err-*`), not a random same-category pick. Still honours
        `require_wav`: a meta line with nothing rendered yet can't be played either."""
        for line in self._lines:
            if line["id"] != line_id:
                continue
            if self.require_wav:
                wav = line.get("wav")
                if not wav or not Path(wav).exists():
                    return None
            return LineResult(
                text=line["text"], category=line["category"], path="library", wav=line["wav"]
            )
        return None
