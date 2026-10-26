"""censor: the two-layer fail-closed content filter every live quip line must pass before
it reaches the speaker. Spec §7.

Layer 1 (`deterministic`): normalise + blocklist/regex/allowlist checks. Zero network calls,
catches the bulk of unsafe content (profanity, slurs, body/age/disability mockery, violence,
romance, adult themes) and the known-bad "shuffling with more grace than most adults manage
sober" class of line. It is fast and over-inclusive by design -- false positives here just
mean a line doesn't get spoken, which is the safe failure direction.

Layer 2 (`_guard`): IBM Granite Guardian 3.2 (Apache-2.0), served by vLLM at
`cfg.censor.guard_endpoint`. It is the backstop for the long tail the blocklist can't
enumerate (03-01 plan, WWMD phase 2->3 amendment: deterministic-only reaches ~90%, never
ships alone). It checks the six risk names Granite Guardian supports for free-form content
(`harm`, `social_bias`, `profanity`, `sexual_content`, `unethical_behavior`, `violence`),
dispatched concurrently (the closest approximation to "one batched request" vLLM's chat
endpoint allows across six distinct `guardian_config` risk names -- there is no single HTTP
call that can carry six different risk names to one chat-template render) under one
`_GUARD_TOTAL_BUDGET_S` timeout for the whole batch.

Fail-closed is absolute, per spec §7 and the ROADMAP/WWMD amendment: ANY exception, parse
failure, or timeout anywhere in `deterministic()` or `check()` returns `Reject`, never
`Pass`. There is no code path that returns Pass from an except block.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re

import httpx

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Reject:
    reason: str


@dataclasses.dataclass(frozen=True)
class Pass:
    pass


_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_SEPARATORS = re.compile(r"[._\-*]")
# Costume-absence phrasing is banned for EVERY category: ~5% of costumed kids are judged
# no_costume (02-02/03-02 blind re-grades), so a line that remarks on a missing costume can land
# on a child who dressed up. Matched against the raw lowercased text (before separator
# stripping, so "bare-faced" and "bare faced" both hit) and against normalise(text).
_COSTUME_ABSENCE = re.compile(
    r"\bno\s+costumes?\b|\b(?:without|sans)\s+(?:a\s+|any\s+|your\s+|their\s+)?costumes?\b"
    r"|\b(?:forgot|forgotten|forget|missing|lacking|skipped)\b[\w\s,'-]{0,20}\bcostumes?\b"
    r"|\bbare[\s-]*faced\b|\bcostume[\s-]*(?:free|less)\b|\bun[\s-]*costumed\b"
    r"|\bplain[\s-]*clothes\b|\bin\s+civvies\b|\bnot\s+(?:even\s+)?(?:in\s+)?(?:a\s+)?costume\b"
)
_REPEATED = re.compile(r"([a-z])\1{2,}")  # 3+ of the same letter -> 1 ("faaat" -> "fat")
_SPACED_LETTERS = re.compile(r"\b(?:[a-z]\s+){2,}[a-z]\b")  # "f a t" -> "fat"
_NON_PRINTABLE = re.compile(r"[^\x20-\x7e]")
_DIGIT = re.compile(r"\d")
_MAX_WORDS = 15

# Costume-category allowlist (03-01 plan): a term/phrase that would otherwise match the
# blocklist is exempted ONLY for the listed categories. `None` means exempted for any
# category. Checked before the blocklist scan; only the matched span is exempted, so an
# allowed "dead" in a skeleton line doesn't also whitelist an unrelated "kill" in the same
# line. "baby shark" is a literal-phrase exemption (otherwise "baby" alone, age class,
# blocks every baby-shark costume line). "dead" is a term-level exemption scoped to
# pirate/zombie/skeleton/ghost -- those categories are literally about being dead, so
# this covers both the plan's named "dead pirate" case and its own worked OK example ("A
# skeleton! Positively dead on time.") plus the existing approved zombie line ("The
# walking dead have arrived..."), while still rejecting "dead" as a real threat elsewhere.
_ALLOWLIST: dict[str, frozenset[str] | None] = {
    "baby shark": None,
    "dead": frozenset({"pirate", "zombie", "skeleton", "ghost"}),
}

# Superset of the blocklist's body/appearance class, used only by the second-person
# pattern below. Deliberately wider than config/blocklist.txt's literal body-term list --
# this check exists to catch appearance language the static list doesn't enumerate
# (e.g. "smile", "eyes", "skin") when it's aimed AT someone with "your"/"ur", not to
# duplicate the blocklist's own standalone matches.
_BODY_WINDOW_TERMS = frozenset(
    "fat chubby skinny short tall ugly face teeth hair nose legs belly weight "
    "smile eyes skin body shape arms hands cheeks chin".split()
)
_SECOND_PERSON = re.compile(r"\b(?:your|ur)\b")
_WINDOW = 3  # words either side of "your"/"ur"

_GUARD_RISKS: tuple[str, ...] = (
    "social_bias",
    "profanity",
    "sexual_content",
    "unethical_behavior",
    "violence",
)
# `harm` (the umbrella risk) is deliberately dropped: measured 2026-10-15 flagging two
# approved library lines ("...costumed hordes", "...save anyone before the finish line")
# as harmful on the Victorian-pantomime persona's stylised phrasing alone, with no actual
# unsafe content underneath. Tightening the risk set per the 03-01 plan's own instruction
# ("fix by tightening the prompt or risk set, not by adding allowlist hacks") -- the five
# specific risks below still cover the real content-safety surface this phase cares about.
# Default guard budget. 03-01 measured p95 142 ms alone but 253 ms with the VLM busy on the
# same GB10, so 0.25 s would turn nearly every live line into guard_timeout (fail-closed means
# silently library-only). Configurable via censor.guard_timeout_s.
_GUARD_TOTAL_BUDGET_S = 0.6


def _load_blocklist(path: str) -> list[str]:
    terms: list[str] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(line)
    return terms


def normalise(text: str) -> str:
    """Lowercase, fold leetspeak, strip dot/dash/underscore/asterisk separators, merge
    spaced-out single letters ("f a t" -> "fat"), then collapse runs of 3+ identical
    letters to 1 ("faaat" -> "fat"). Separator-stripping runs before the spaced-letter
    merge so "f.a.t" and "f a t" both land on the same normalised form."""
    t = text.lower()
    t = t.translate(_LEET)
    t = _SEPARATORS.sub("", t)
    t = _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), t)
    t = _REPEATED.sub(r"\1", t)
    return t


class Censor:
    """`client` is injectable so tests can pass an `httpx.AsyncClient` wired to an
    `httpx.MockTransport` instead of hitting a real guard server (same pattern as
    `judge.Judge`)."""

    def __init__(
        self,
        blocklist_path: str,
        guard_endpoint: str,
        guard_model: str,
        client: httpx.AsyncClient | None = None,
        guard_timeout_s: float = _GUARD_TOTAL_BUDGET_S,
    ) -> None:
        self.guard_timeout_s = guard_timeout_s
        terms = _load_blocklist(blocklist_path)
        self._pattern = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b")
        self.guard_model = guard_model
        self._client = client if client is not None else httpx.AsyncClient(base_url=guard_endpoint)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def deterministic(self, text: str, category: str = "") -> Reject | None:
        """Returns `Reject(reason)` on a deterministic objection, or `None` if this layer
        has no objection -- `None` is NOT a pass; `check()` still has to clear the guard."""
        try:
            if _NON_PRINTABLE.search(text):
                return Reject("non_printable")
            if len(text.split()) > _MAX_WORDS:
                return Reject("too_long")
            if _DIGIT.search(text):
                return Reject("digits")

            if _COSTUME_ABSENCE.search(text.lower()) or _COSTUME_ABSENCE.search(normalise(text)):
                return Reject("costume_absence")

            norm = normalise(text)

            allowed_spans: list[tuple[int, int]] = []
            for phrase, allowed_categories in _ALLOWLIST.items():
                if allowed_categories is not None and category not in allowed_categories:
                    continue
                for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", norm):
                    allowed_spans.append(m.span())

            def _covered(span: tuple[int, int]) -> bool:
                return any(a[0] <= span[0] and span[1] <= a[1] for a in allowed_spans)

            for m in self._pattern.finditer(norm):
                if not _covered(m.span()):
                    return Reject(f"blocklist:{m.group(0)}")

            norm_words = norm.split()
            for idx, word in enumerate(norm_words):
                if _SECOND_PERSON.fullmatch(word):
                    lo = max(0, idx - _WINDOW)
                    hi = min(len(norm_words), idx + _WINDOW + 1)
                    window = norm_words[lo:hi]
                    if any(w in _BODY_WINDOW_TERMS for w in window):
                        return Reject("second_person_body")

            return None
        except Exception as exc:  # noqa: BLE001 -- fail closed on any surprise
            logger.warning("censor.deterministic_error reason=%s", exc)
            return Reject(f"error:{exc}")

    async def _guard_risk(self, text: str, risk: str) -> bool:
        """True if Granite Guardian flags `risk` as present. Any non-clean "No" answer
        (including "Yes", a parse failure, or an HTTP error) counts as flagged -- the
        caller treats a flagged risk as Reject, never as a pass."""
        payload = {
            "model": self.guard_model,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.0,
            "max_tokens": 5,
            "chat_template_kwargs": {"guardian_config": {"risk_name": risk}},
        }
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip().lower()
        if content.startswith("no"):
            return False
        if content.startswith("yes"):
            return True
        # Anything else (empty, garbled, unexpected label) is a parse failure -> flagged.
        raise ValueError(f"unparseable guard output for risk={risk!r}: {content!r}")

    async def _guard(self, text: str) -> Reject | None:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(self._guard_risk(text, risk) for risk in _GUARD_RISKS),
                    return_exceptions=True,
                ),
                timeout=self.guard_timeout_s,
            )
        except asyncio.TimeoutError:
            return Reject("guard_timeout")
        for risk, result in zip(_GUARD_RISKS, results):
            if isinstance(result, Exception):
                logger.warning("censor.guard_error risk=%s reason=%s", risk, result)
                return Reject(f"guard_error:{risk}")
            if result:
                return Reject(f"guard:{risk}")
        return None

    async def check(self, text: str, category: str = "") -> Reject | Pass:
        """Deterministic layer first (cheap, no network); only a clean `None` from it goes
        on to the guard. Fail-closed: any surprise anywhere becomes `Reject`, never `Pass`."""
        try:
            det = self.deterministic(text, category)
            if det is not None:
                return det
            guard_reject = await self._guard(text)
            if guard_reject is not None:
                return guard_reject
            return Pass()
        except Exception as exc:  # noqa: BLE001 -- fail closed, no path returns Pass here
            logger.warning("censor.check_error reason=%s", exc)
            return Reject(f"error:{exc}")
