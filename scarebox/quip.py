"""quip: the live joke writer. Spec §4.3, §5.3, §6 (few-shots from the library).

`Quip(cfg, censor)` makes a text-only chat-completion call to the VLM server
(`cfg.quip.endpoint`, same server/model the judge uses -- spec §4.5) to turn a judge
`Verdict` into ONE short, safe, costume-specific line, or `None` if nothing survives.

The call asks for `prompts.NUM_CANDIDATES` (2) candidate lines via vLLM structured output
(`response_format: json_schema`, `prompts.QUIP_JSON_SCHEMA`) so a syntactically valid
response always parses -- the 01-03 bake-off measured a 15-17% schema-miss rate on
free-form JSON. 6-8 few-shot examples are drawn at call time from the library (approved
lines in the category, or pending if none are approved yet) so the model matches the
established voice instead of inventing its own register each time.

Candidates are filtered by: contains the costume head noun (or the category word if the
costume is empty/"none"), is `<= quip.max_words`, and clears `Censor.check()`. The shortest
surviving candidate is returned (spec §4.3: shorter clears the deadline orchestrator's
TTS/FX budget with more room). Any network error, timeout, or structurally invalid
response returns `None` and logs `quip.fail` with the reason -- same fail-safe shape as
`judge.Judge.judge()`.

`judge_and_quip()` is the single-call fallback (behind `quip.single_call: true`, spec §4.5
lever 2, flagged by 01-03): one VLM call returns BOTH the terse verdict and the candidate
lines, trading a bigger single prompt/response for skipping a whole extra round-trip.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import httpx

from scarebox.censor import Censor, Pass
from scarebox.config import Config
from scarebox.events import Event, Verdict
from scarebox.judge import _encode_data_url
from scarebox.library import resolve_category
from scarebox.prompts import (
    JUDGE_AND_QUIP_JSON_SCHEMA,
    QUIP_JSON_SCHEMA,
    costume_noun_in_line,
    judge_and_quip_messages,
    line_word_count,
    quip_messages,
    validate_judge_and_quip,
    validate_quip,
)

logger = logging.getLogger(__name__)

# 03-03: the orchestrator's `reject_reason` event-log field (spec §4.4 amendment) buckets
# every way a live line can fail to reach the speaker into five values. The first four mirror
# `censor.Reject.reason` strings (so `scripts/latency_report.py` can read ADR 0001's ">30%
# rejected" trigger honestly); "quip-none" covers everything that never got as far as a
# censor call at all (a candidate failed the costume-noun/word-count filter, or the quip/
# judge_and_quip network call itself timed out, errored, or returned a malformed response).
#
# `reject_reason` is the bucket; `reject_detail` (added alongside it, same plumbing) is the
# specific reason underneath -- the actual blocklist term, the guard risk name, "guard_timeout",
# or (for the quip-none bucket) which filter/call failure it was ("costume_noun_miss",
# "too_long", "timeout", "no_crops", ...). `reject_reason` is what a dashboard buckets on;
# `reject_detail` is what a human reads to find out WHY.
def _censor_bucket(reason: str) -> str:
    if reason == "guard_timeout":
        return "guard-timeout"
    if reason.startswith("guard:"):
        return "guard-risk"
    if reason.startswith("guard_error:") or reason.startswith("error:"):
        return "parse"
    return "deterministic"  # non_printable / too_long / digits / blocklist:* / second_person_body


_FEW_SHOT_N = 8

# Typographic characters the line-writing model likes to emit, mapped to their plain-ASCII
# equivalents before the censor sees the line. The censor rejects anything outside printable
# ASCII (a deliberate guard against look-alike-letter tricks), so an em dash alone was
# killing otherwise-good live lines (2026-10-19 detection report: "Skeletons don't pause --
# they pounce!" blocked as non_printable). Only this fixed list is mapped; anything else
# non-ASCII still reaches the censor unchanged and is still rejected.
_TYPOGRAPHY = {
    "—": ", ", "–": ", ", "…": "...", "‘": "'", "’": "'",
    "“": '"', "”": '"', " ": " ",
}


def straighten_typography(line: str) -> str:
    for k, v in _TYPOGRAPHY.items():
        line = line.replace(k, v)
    line = " ".join(line.split())
    return line.replace(" ,", ",").replace(",,", ",")
# Combined call's output is a terse verdict (~25-30 tokens) PLUS 2 lines (~20-30 tokens) in
# one response -- generous over either alone, since there's no second round-trip to amortise
# the margin against.
_JUDGE_AND_QUIP_MAX_TOKENS = 110


def _verdict_text(verdict: Verdict | dict) -> tuple[str, str, bool]:
    """Pull (costume, details, is_costume) out of either a Verdict or a plain dict -- `judge`
    callers pass a real Verdict, tests sometimes pass a bare dict (same convention as
    `prompts.quip_messages`)."""
    if isinstance(verdict, Verdict):
        return verdict.costume, verdict.details, verdict.is_costume
    return (
        verdict.get("costume", ""),
        verdict.get("details", ""),
        verdict.get("is_costume", True),
    )


def _few_shot_lines(manifest_path: str, category: str, n: int = _FEW_SHOT_N) -> list[str]:
    """6-8 approved library lines for `category`, falling back to pending ones if none are
    approved yet (spec §6 -- few-shots must come from the library, never hardcoded unvetted
    text). Returns `[]` on any read/parse problem or an empty category: an empty few-shot
    list is a degraded prompt, not a crash."""
    try:
        lines = json.loads(Path(manifest_path).read_text())
    except Exception as exc:  # noqa: BLE001 -- missing/corrupt manifest shouldn't crash quip
        logger.warning("quip.few_shots_error reason=%s", exc)
        return []
    pool = [
        line["text"]
        for line in lines
        if line.get("category") == category and line.get("status") == "approved"
    ]
    if not pool:
        pool = [
            line["text"]
            for line in lines
            if line.get("category") == category and line.get("status") == "pending"
        ]
    if len(pool) > n:
        pool = random.sample(pool, n)
    return pool


class Quip:
    """`client` is injectable so tests can pass an `httpx.AsyncClient` wired to an
    `httpx.MockTransport` instead of hitting a real server (same pattern as `judge.Judge`)."""

    def __init__(self, cfg: Config, censor: Censor, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg.quip
        self._manifest = cfg.library.manifest
        self._censor = censor
        self._client = client if client is not None else httpx.AsyncClient(base_url=self.cfg.endpoint)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _survivor(
        self,
        candidates: list[str],
        costume: str,
        category: str,
        is_costume: bool = True,
        blocked: dict | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        """Filters `candidates` to costume-noun-containing, within `max_words`, censor-clean
        lines, and returns `(shortest_survivor_or_None, reject_reason_or_None,
        reject_detail_or_None)` -- the reason/detail are only meaningful when the first
        element is `None` (03-03: the orchestrator's `reject_reason`/`reject_detail`
        event-log fields, see `_censor_bucket` above). `reject_detail` is the raw string
        underneath the bucket: the censor's own `Reject.reason` (e.g. `"blocklist:fat"`,
        `"guard:violence"`, `"guard_timeout"`) when a candidate reached the censor, or which
        pre-censor filter missed (`"costume_noun_miss"`, `"too_long"`) when it never did.

        The noun filter is SKIPPED entirely when `is_costume` is False: `NO_COSTUME_RULE`
        requires those lines to never mention a costume or its absence at all (hub amendment,
        02-02 Task 3), so requiring the costume/category noun to appear would be actively
        wrong for exactly the lines this path has to pass.

        `blocked` (04-04, aim control room): an IN/OUT dict the caller owns. Left untouched
        (empty) unless a candidate actually reached `Censor.check()` and was rejected, in
        which case it's filled with `{"text": <the rejected line>, "reason": <Reject.reason>}`
        -- the aim page's event-card BLOCKED field, "never in the live view". A candidate that
        never reached the censor (missed the costume-noun/word-count filter) has no line worth
        showing as blocked, so `blocked` stays empty for that case.
        """
        noun_target = costume if costume and costume.strip().lower() != "none" else category
        survivors: list[str] = []
        reason: str | None = None
        detail: str | None = None
        for line in candidates:
            line = straighten_typography(line)
            if is_costume and not costume_noun_in_line(noun_target, line):
                reason = reason or "quip-none"
                detail = detail or "costume_noun_miss"
                continue
            if line_word_count(line) > self.cfg.max_words:
                reason = reason or "quip-none"
                detail = detail or "too_long"
                continue
            result = await self._censor.check(line, category)
            if isinstance(result, Pass):
                survivors.append(line)
            else:
                # A real censor reject is more specific/actionable than a bare "quip-none"
                # filter miss, so it wins even if an earlier candidate was filter-rejected.
                reason = _censor_bucket(result.reason)
                detail = result.reason
                if blocked is not None:
                    blocked["text"] = line
                    blocked["reason"] = result.reason
        if not survivors:
            return None, reason, detail
        return min(survivors, key=line_word_count), None, None

    async def _quip_impl(
        self, verdict: Verdict | dict, category: str, blocked: dict | None = None
    ) -> tuple[str | None, str | None, str | None]:
        costume, details, is_costume = _verdict_text(verdict)
        verdict_dict = {
            "costume": costume,
            "category": category,
            "details": details,
            "is_costume": is_costume,
        }
        few_shots = _few_shot_lines(self._manifest, category)

        payload = {
            "model": self.cfg.model,
            "messages": quip_messages(verdict_dict, few_shots=few_shots),
            "response_format": {"type": "json_schema", "json_schema": QUIP_JSON_SCHEMA},
            "temperature": 0.9,
            "max_tokens": 50,
        }
        try:
            resp = await self._client.post(
                "/chat/completions", json=payload, timeout=self.cfg.timeout_s
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            obj = json.loads(content)
        except httpx.TimeoutException:
            logger.warning("quip.fail category=%s reason=timeout", category)
            return None, "quip-none", "timeout"
        except Exception as exc:  # noqa: BLE001 -- network error, bad JSON, bad shape, garbage
            logger.warning("quip.fail category=%s reason=%s", category, exc)
            return None, "quip-none", f"error:{exc}"

        ok, reason = validate_quip(obj)
        if not ok:
            logger.warning("quip.fail category=%s reason=invalid_quip:%s", category, reason)
            return None, "quip-none", f"invalid_quip:{reason}"

        return await self._survivor(obj["lines"], costume, category, is_costume, blocked=blocked)

    async def quip(self, verdict: Verdict | dict, category: str) -> str | None:
        """One text-only quip call for `verdict`, filed under the (possibly group-resolved)
        `category` -- `library.resolve_category()`'s output, which can differ from
        `verdict.category` for a group. Returns the shortest surviving candidate, or `None`.
        """
        line, _, _ = await self._quip_impl(verdict, category)
        return line

    async def quip_with_reason(
        self, verdict: Verdict | dict, category: str
    ) -> tuple[str | None, str | None, str | None]:
        """Same as `quip()` but also returns `(reject_reason, reject_detail)` (03-03, see
        `_censor_bucket`/`_survivor`) when nothing survived -- the orchestrator's events-log
        fields."""
        return await self._quip_impl(verdict, category)

    async def quip_with_blocked(
        self, verdict: Verdict | dict, category: str, blocked: dict
    ) -> tuple[str | None, str | None, str | None]:
        """Same as `quip_with_reason()`, plus fills `blocked` (04-04, see `_survivor`'s
        docstring) with the actual censor-blocked line text/reason when one exists -- the
        orchestrator's `LiveOutcome.blocked_text`/`blocked_reason`."""
        return await self._quip_impl(verdict, category, blocked=blocked)

    async def _judge_and_quip_impl(
        self, event: Event, blocked: dict | None = None
    ) -> tuple[Verdict | None, str | None, str | None, str | None]:
        if not event.crops:
            logger.warning("judge_and_quip.fail event=%s reason=no_crops", event.id)
            return None, None, "quip-none", "no_crops"
        try:
            image_urls = [_encode_data_url(c) for c in event.crops[:3]]
        except Exception as exc:  # noqa: BLE001 -- any encode failure is a call failure
            logger.warning("judge_and_quip.fail event=%s reason=encode_error:%s", event.id, exc)
            return None, None, "quip-none", f"encode_error:{exc}"

        payload = {
            "model": self.cfg.model,
            "messages": judge_and_quip_messages(image_urls),
            "response_format": {"type": "json_schema", "json_schema": JUDGE_AND_QUIP_JSON_SCHEMA},
            "temperature": 0.9,
            "max_tokens": _JUDGE_AND_QUIP_MAX_TOKENS,
        }
        try:
            resp = await self._client.post(
                "/chat/completions", json=payload, timeout=self.cfg.single_call_timeout_s
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            obj = json.loads(content)
        except httpx.TimeoutException:
            logger.warning("judge_and_quip.fail event=%s reason=timeout", event.id)
            return None, None, "quip-none", "timeout"
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge_and_quip.fail event=%s reason=%s", event.id, exc)
            return None, None, "quip-none", f"error:{exc}"

        if isinstance(obj, dict) and isinstance(obj.get("f"), (int, float)):
            obj["f"] = max(0.0, min(1.0, float(obj["f"])))

        ok, reason = validate_judge_and_quip(obj)
        if not ok:
            logger.warning("judge_and_quip.fail event=%s reason=invalid:%s", event.id, reason)
            return None, None, "quip-none", f"invalid:{reason}"

        category = resolve_category(obj["k"], event.group_size)
        verdict = Verdict(
            costume=obj["c"],
            category=category,
            confidence=float(obj["f"]),
            is_costume=obj["i"],
            group_size=obj["n"],
            details=obj["d"],
        )
        survivor, reject_reason, reject_detail = await self._survivor(
            obj["l"], verdict.costume, category, verdict.is_costume, blocked=blocked
        )
        return verdict, survivor, reject_reason, reject_detail

    async def judge_and_quip(self, event: Event) -> tuple[Verdict | None, str | None]:
        """Single-call fallback (`quip.single_call: true`): one VLM call returns both the
        verdict and the candidate lines. Returns `(verdict, line)`; either half can be
        `None` independently of the other (a valid verdict with no surviving line is a
        normal outcome, same as the two-call path). Uses `quip.single_call_timeout_s`
        (03-03), not the shared `quip.timeout_s` -- 03-02 measured 0/20 survive at the
        shared 1.5s budget because the combined verdict+lines decode needs more room."""
        verdict, line, _, _ = await self._judge_and_quip_impl(event)
        return verdict, line

    async def judge_and_quip_with_reason(
        self, event: Event
    ) -> tuple[Verdict | None, str | None, str | None, str | None]:
        """Same as `judge_and_quip()` but also returns `(reject_reason, reject_detail)`
        (03-03) when no line survived -- the orchestrator's events-log fields."""
        return await self._judge_and_quip_impl(event)

    async def judge_and_quip_with_blocked(
        self, event: Event, blocked: dict
    ) -> tuple[Verdict | None, str | None, str | None, str | None]:
        """Same as `judge_and_quip_with_reason()`, plus fills `blocked` (04-04, see
        `_survivor`'s docstring) with the actual censor-blocked line text/reason when one
        exists."""
        return await self._judge_and_quip_impl(event, blocked=blocked)
