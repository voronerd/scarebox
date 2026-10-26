"""Tests for `scarebox.censor`. Spec §7.

Unit tests (MockTransport, no network) run everywhere, including the pure-logic
`.venv/bin/pytest` on a CPU-only dev machine. The live two-layer integration test needs a real Granite Guardian
server reachable at `CENSOR_GUARD_ENDPOINT` (default `http://127.0.0.1:8101/v1`) and is
skipped automatically when that server isn't reachable -- in practice
that only happens inside the `app` container on gbox1 with `guard` up.
"""

from __future__ import annotations

import asyncio
import os
import socket
import urllib.parse
from pathlib import Path

import httpx
import pytest

from scarebox.censor import Censor, Pass, Reject, normalise

REPO_ROOT = Path(__file__).resolve().parent.parent
BLOCKLIST_PATH = REPO_ROOT / "config" / "blocklist.txt"
LINES_PATH = REPO_ROOT / "tests" / "redteam" / "lines.txt"
OK_PATH = REPO_ROOT / "tests" / "redteam" / "ok.txt"
GUARD_ONLY_MARKER = "GUARD-ONLY"

GUARD_ENDPOINT = os.environ.get("CENSOR_GUARD_ENDPOINT", "http://127.0.0.1:8101/v1")
GUARD_MODEL = os.environ.get("CENSOR_GUARD_MODEL", "granite-guardian")


def _guard_reachable() -> bool:
    parsed = urllib.parse.urlparse(GUARD_ENDPOINT)
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=1.0):
            return True
    except OSError:
        return False


def _load_lines(path: Path) -> list[str]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _load_ok_lines() -> list[tuple[str, str]]:
    """Returns (text, category) pairs. ok.txt lines are "text |CAT| category" -- category
    matters for the costume allowlist (e.g. "dead" is only exempted for pirate/zombie/
    skeleton/ghost)."""
    out = []
    for text in _load_lines(OK_PATH):
        if " |CAT| " in text:
            body, _, cat = text.partition(" |CAT| ")
            out.append((body.strip(), cat.strip()))
        else:
            out.append((text, ""))
    return out


def _load_redteam_sections() -> tuple[list[str], list[str]]:
    """Returns (deterministic_catch_lines, guard_only_lines), split at the marker comment."""
    det_lines: list[str] = []
    guard_only: list[str] = []
    in_guard_section = False
    for raw in LINES_PATH.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        # Only the actual sentinel comment line (not incidental prose mentioning the
        # phrase earlier in the file's own header) starts with "# ---".
        if stripped.startswith("# ---") and GUARD_ONLY_MARKER in stripped:
            in_guard_section = True
            continue
        if not stripped or stripped.startswith("#"):
            continue
        (guard_only if in_guard_section else det_lines).append(stripped)
    return det_lines, guard_only


def _blocklist_terms() -> list[str]:
    terms = []
    for raw in BLOCKLIST_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        terms.append(line)
    return terms


def _censor(client: httpx.AsyncClient | None = None) -> Censor:
    return Censor(str(BLOCKLIST_PATH), GUARD_ENDPOINT, GUARD_MODEL, client=client)


# --- normalise() -----------------------------------------------------------------------


def test_normalise_leetspeak():
    assert normalise("Y0u ar3 s0 f4t") == "you are so fat"


def test_normalise_repeated_letters():
    assert "fat" in normalise("you are so faaaat")


def test_normalise_separators():
    assert "fat" in normalise("f.a.t. costume")


def test_normalise_spaced_letters():
    assert "fat" in normalise("f a t kid")


# --- deterministic() --------------------------------------------------------------------


def test_deterministic_catches_blocklist_term():
    c = _censor()
    assert isinstance(c.deterministic("That costume makes you look so fat."), Reject)


def test_deterministic_passes_clean_line():
    c = _censor()
    assert c.deterministic("A splendid creature joins the race.") is None


def test_deterministic_too_long():
    c = _censor()
    text = " ".join(["word"] * 16)
    r = c.deterministic(text)
    assert isinstance(r, Reject)
    assert r.reason == "too_long"


def test_deterministic_digits():
    c = _censor()
    r = c.deterministic("You scored a 9 out of 10 today.")
    assert isinstance(r, Reject)
    assert r.reason == "digits"


def test_deterministic_non_printable():
    c = _censor()
    r = c.deterministic("That costume is \x07 dreadful.")
    assert isinstance(r, Reject)
    assert r.reason == "non_printable"


def test_deterministic_second_person_body():
    c = _censor()
    r = c.deterministic("Your smile looks so strange today.")
    assert isinstance(r, Reject)
    assert r.reason == "second_person_body"


def test_allowlist_baby_shark_passes():
    c = _censor()
    assert c.deterministic("A baby shark costume swims by.", category="animal") is None


def test_allowlist_baby_alone_still_blocked():
    c = _censor()
    assert isinstance(c.deterministic("What a baby you are.", category="animal"), Reject)


def test_allowlist_dead_passes_for_zombie_category():
    c = _censor()
    assert c.deterministic("The walking dead have arrived.", category="zombie") is None


def test_allowlist_dead_still_blocked_outside_scoped_categories():
    c = _censor()
    assert isinstance(c.deterministic("You look dead tired today.", category="princess"), Reject)


def test_killer_does_not_false_positive_on_kill():
    # "kill" has a word boundary after the 'l'; "killer" doesn't satisfy it, so this must
    # NOT be caught by the "kill" blocklist entry -- exercises regex word-boundary precision.
    c = _censor()
    assert c.deterministic("A vampire with a killer cape.", category="vampire") is None


@pytest.mark.parametrize("term", _blocklist_terms())
def test_every_blocklist_term_is_caught_in_isolation(term: str):
    c = _censor()
    r = c.deterministic(f"That costume is so very {term} today.")
    assert isinstance(r, Reject), f"blocklist term {term!r} was not caught"


# --- red-team fixture rates (deterministic layer only, no guard) -----------------------


def test_redteam_deterministic_reject_rate_at_least_90pct():
    det_lines, _guard_only = _load_redteam_sections()
    c = _censor()
    rejected = sum(1 for line in det_lines if isinstance(c.deterministic(line), Reject))
    rate = rejected / len(det_lines)
    assert rate >= 0.90, f"deterministic-only reject rate {rate:.2%} on {len(det_lines)} lines"


def test_redteam_guard_only_lines_are_not_caught_deterministically():
    """Sanity check on the fixture itself: the guard-only section must genuinely slip past
    the deterministic layer, or the "only the guard catches these" claim is false."""
    _det_lines, guard_only = _load_redteam_sections()
    assert len(guard_only) >= 10
    c = _censor()
    for line in guard_only:
        assert c.deterministic(line) is None, f"guard-only line was caught deterministically: {line!r}"


def test_ok_set_deterministic_pass_rate_100pct():
    ok_lines = _load_ok_lines()
    assert len(ok_lines) >= 60
    c = _censor()
    passed = sum(1 for text, cat in ok_lines if c.deterministic(text, category=cat) is None)
    assert passed == len(ok_lines), f"{len(ok_lines) - passed} ok lines rejected deterministically"


# --- guard-mocked check() (no live server; MockTransport) -------------------------------


def _mock_guard_client(flagged_risks: frozenset[str]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        risk = body["chat_template_kwargs"]["guardian_config"]["risk_name"]
        label = "Yes" if risk in flagged_risks else "No"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": label}}]}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test/v1")


def test_check_passes_when_deterministic_clean_and_guard_says_no():
    async def _run():
        client = _mock_guard_client(frozenset())
        c = _censor(client=client)
        return await c.check("A splendid creature joins the race.")

    result = asyncio.run(_run())
    assert isinstance(result, Pass)


def test_check_rejects_when_guard_flags_a_risk():
    async def _run():
        client = _mock_guard_client(frozenset({"violence"}))
        c = _censor(client=client)
        return await c.check("A splendid creature joins the race.")

    result = asyncio.run(_run())
    assert isinstance(result, Reject)
    assert result.reason == "guard:violence"


def test_check_rejects_on_deterministic_without_calling_guard():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("guard should never be called once deterministic rejects")

    async def _run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test/v1")
        c = _censor(client=client)
        return await c.check("That costume makes you look so fat.")

    result = asyncio.run(_run())
    assert isinstance(result, Reject)


def test_check_fails_closed_on_guard_parse_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "garbled nonsense"}}]})

    async def _run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test/v1")
        c = _censor(client=client)
        return await c.check("A splendid creature joins the race.")

    result = asyncio.run(_run())
    assert isinstance(result, Reject)


def test_check_fails_closed_on_guard_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async def _run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test/v1")
        c = _censor(client=client)
        return await c.check("A splendid creature joins the race.")

    result = asyncio.run(_run())
    assert isinstance(result, Reject)


# --- live two-layer integration (needs a real guard server; skipped otherwise) ----------


@pytest.mark.skipif(not _guard_reachable(), reason="guard server not reachable, skipping live integration test")
def test_live_two_layer_redteam_100pct_rejected():
    async def _run():
        det_lines, guard_only = _load_redteam_sections()
        all_lines = det_lines + guard_only
        c = _censor()
        try:
            return [await c.check(line) for line in all_lines], all_lines
        finally:
            await c.aclose()

    results, all_lines = asyncio.run(_run())
    rejected = sum(1 for r in results if isinstance(r, Reject))
    assert rejected == len(all_lines), f"{len(all_lines) - rejected} redteam lines were NOT rejected"


@pytest.mark.skipif(not _guard_reachable(), reason="guard server not reachable, skipping live integration test")
def test_live_two_layer_ok_100pct_passed():
    async def _run():
        ok_lines = _load_ok_lines()
        c = _censor()
        try:
            return [await c.check(text, category=cat) for text, cat in ok_lines], ok_lines
        finally:
            await c.aclose()

    results, ok_lines = asyncio.run(_run())
    passed = sum(1 for r in results if isinstance(r, Pass))
    assert passed == len(ok_lines), f"{len(ok_lines) - passed} ok lines were rejected"


def test_guard_timeout_is_configurable_and_defaults_above_measured_contention_p95():
    # 03-01 measured guard p95 253 ms with the VLM busy; the default must clear that.
    from scarebox.censor import Censor
    c = Censor(str(BLOCKLIST_PATH), GUARD_ENDPOINT, GUARD_MODEL)
    assert c.guard_timeout_s >= 0.5
    c2 = Censor(str(BLOCKLIST_PATH), GUARD_ENDPOINT, GUARD_MODEL, guard_timeout_s=1.25)
    assert c2.guard_timeout_s == 1.25
