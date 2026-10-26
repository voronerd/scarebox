import asyncio
import json

import httpx
import pytest

from scarebox.censor import Pass, Reject
from scarebox.config import load_config
from scarebox.events import Event, Verdict
from scarebox.quip import Quip


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("CAMERA_HOST", "x")
    monkeypatch.setenv("CAMERA_USER", "x")
    monkeypatch.setenv("CAMERA_PASSWORD", "x")
    monkeypatch.setenv("AIM_TOKEN", "x")
    return load_config("config/bench.yaml")


class _FakeCensor:
    """Pure-python stand-in for `censor.Censor` -- controllable rejects, no network, no
    blocklist file. `Quip` only calls `.check(text, category)`, so this is the whole
    surface it needs."""

    def __init__(self, reject_substrings: tuple[str, ...] = ()):
        self.reject_substrings = reject_substrings
        self.checked: list[str] = []

    async def check(self, text: str, category: str = ""):
        self.checked.append(text)
        for s in self.reject_substrings:
            if s in text:
                return Reject(f"fake:{s}")
        return Pass()


def _verdict(**overrides) -> Verdict:
    base = dict(
        costume="inflatable T. rex",
        category="dinosaur",
        confidence=0.9,
        is_costume=True,
        group_size=1,
        details="holding a juice box",
    )
    base.update(overrides)
    return Verdict(**base)


def _quip_body(lines: list[str]) -> dict:
    return {"choices": [{"message": {"content": json.dumps({"lines": lines})}}]}


def _client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test/v1")


def test_picks_shortest_passing_line(cfg):
    lines = [
        "This magnificent rex is a dinosaur beyond all prehistoric compare, splendidly so!",
        "A rex! Marvellously prehistoric.",
    ]

    def handler(request):
        return httpx.Response(200, json=_quip_body(lines))

    quip = Quip(cfg, _FakeCensor(), client=_client_for(handler))
    line = asyncio.run(quip.quip(_verdict(), "dinosaur"))
    assert line == "A rex! Marvellously prehistoric."


def test_em_dash_line_is_straightened_before_the_censor(cfg):
    # 2026-10-19: a live line with an em dash was blocked as non_printable. It should reach
    # the censor (and the voice) as plain ASCII instead. Uses the REAL censor's
    # deterministic layer, which is what rejected it.
    from scarebox.censor import Censor
    from scarebox.quip import straighten_typography

    assert straighten_typography("Rex don’t pause — they pounce…") == "Rex don't pause, they pounce..."
    censor = Censor(cfg.censor.blocklist, "http://127.0.0.1:9", cfg.censor.guard_model)
    assert censor.deterministic("A rex — marvellous!", "dinosaur") is not None  # the old failure
    assert censor.deterministic(straighten_typography("A rex — marvellous!"), "dinosaur") is None

    def handler(request):
        return httpx.Response(
            200,
            json=_quip_body(
                ["A rex — marvellously prehistoric!", "This rex is far too long a line to ever be the shortest one, surely!"]
            ),
        )

    quip = Quip(cfg, _FakeCensor(), client=_client_for(handler))
    assert asyncio.run(quip.quip(_verdict(), "dinosaur")) == "A rex, marvellously prehistoric!"


def test_returns_none_when_all_fail_censor(cfg):
    lines = ["A rex! Marvellously prehistoric.", "Behold, a dashing rex of old!"]

    def handler(request):
        return httpx.Response(200, json=_quip_body(lines))

    censor = _FakeCensor(reject_substrings=("rex",))
    quip = Quip(cfg, censor, client=_client_for(handler))
    line = asyncio.run(quip.quip(_verdict(), "dinosaur"))
    assert line is None
    # both candidates reached the censor -- the costume-noun filter didn't short-circuit them
    assert len(censor.checked) == 2


def test_enforces_costume_noun(cfg):
    lines = [
        "What a marvellous day for a run, onward now!",  # no "rex" noun -- must be filtered
        "A rex! Marvellously prehistoric indeed.",
    ]

    def handler(request):
        return httpx.Response(200, json=_quip_body(lines))

    censor = _FakeCensor()
    quip = Quip(cfg, censor, client=_client_for(handler))
    line = asyncio.run(quip.quip(_verdict(), "dinosaur"))
    assert line == "A rex! Marvellously prehistoric indeed."
    # the noun-less line never reached the censor at all
    assert censor.checked == ["A rex! Marvellously prehistoric indeed."]


def test_costume_noun_falls_back_to_category_when_costume_empty(cfg):
    lines = ["Onward, mystery runner, the course awaits!", "What a day for a jog, onward!"]

    def handler(request):
        return httpx.Response(200, json=_quip_body(lines))

    quip = Quip(cfg, _FakeCensor(), client=_client_for(handler))
    verdict = _verdict(costume="", category="mystery")
    line = asyncio.run(quip.quip(verdict, "mystery"))
    assert line == "Onward, mystery runner, the course awaits!"


def test_malformed_response_returns_none_not_crash(cfg):
    def handler(request):
        # Not valid JSON at all -- the guided-JSON schema-rejection path.
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json {{"}}]})

    quip = Quip(cfg, _FakeCensor(), client=_client_for(handler))
    line = asyncio.run(quip.quip(_verdict(), "dinosaur"))
    assert line is None


def test_wrong_shape_response_returns_none_not_crash(cfg):
    def handler(request):
        # Valid JSON, but the wrong shape (one line instead of NUM_CANDIDATES=2).
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps({"lines": ["only one"]})}}]}
        )

    quip = Quip(cfg, _FakeCensor(), client=_client_for(handler))
    line = asyncio.run(quip.quip(_verdict(), "dinosaur"))
    assert line is None


def test_timeout_returns_none(cfg):
    def handler(request):
        raise httpx.TimeoutException("simulated timeout")

    quip = Quip(cfg, _FakeCensor(), client=_client_for(handler))
    line = asyncio.run(quip.quip(_verdict(), "dinosaur"))
    assert line is None


def test_no_costume_rule_applied_for_uncostumed_verdict(cfg):
    # is_costume=False should route through NO_COSTUME_RULE in the system prompt -- this
    # test only checks quip still resolves cleanly end-to-end for that path, the prompt
    # wording itself is covered in test_prompts.py.
    lines = ["A fine effort out there today, keep it up!", "Wonderful pace, onward to the finish!"]

    def handler(request):
        return httpx.Response(200, json=_quip_body(lines))

    verdict = _verdict(costume="none", category="no_costume", is_costume=False)
    quip = Quip(cfg, _FakeCensor(), client=_client_for(handler))
    line = asyncio.run(quip.quip(verdict, "no_costume"))
    assert line in lines


def test_reject_detail_carries_censor_reason(cfg):
    lines = ["A rex! Marvellously prehistoric.", "Behold, a dashing rex of old!"]

    def handler(request):
        return httpx.Response(200, json=_quip_body(lines))

    censor = _FakeCensor(reject_substrings=("rex",))
    quip = Quip(cfg, censor, client=_client_for(handler))
    line, reason, detail = asyncio.run(quip.quip_with_reason(_verdict(), "dinosaur"))
    assert line is None
    assert reason == "deterministic"  # _FakeCensor's Reject("fake:rex") isn't guard/guard_timeout
    assert detail == "fake:rex"  # the raw Reject.reason, not just the bucket


def test_reject_detail_carries_filter_miss(cfg):
    lines = [
        "What a marvellous day for a run, onward now!",  # no "rex" noun -- costume-noun miss
    ] * 2  # NUM_CANDIDATES-shaped; both candidates miss the filter, never reach censor

    def handler(request):
        return httpx.Response(200, json=_quip_body(lines))

    censor = _FakeCensor()
    quip = Quip(cfg, censor, client=_client_for(handler))
    line, reason, detail = asyncio.run(quip.quip_with_reason(_verdict(), "dinosaur"))
    assert line is None
    assert reason == "quip-none"
    assert detail == "costume_noun_miss"
    assert censor.checked == []  # never reached the censor at all


def test_judge_and_quip_with_reason_carries_detail(cfg):
    quip = Quip(cfg, _FakeCensor(), client=_client_for(lambda r: httpx.Response(200)))
    verdict, line, reason, detail = asyncio.run(
        quip.judge_and_quip_with_reason(Event(t_trigger=1.0))
    )
    assert verdict is None
    assert line is None
    assert reason == "quip-none"
    assert detail == "no_crops"


def test_judge_and_quip_single_call(cfg):
    import numpy as np

    body = {
        "c": "zombie bride",
        "k": "zombie",
        "f": 0.8,
        "i": True,
        "n": 1,
        "d": "torn veil",
        "l": ["A zombie bride! Delightfully decayed.", "What a zombie, undeath becomes you."],
    }

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    event = Event(t_trigger=1.0, track_ids=[1], group_size=1, crops=[np.zeros((20, 20, 3), dtype="uint8")])
    quip = Quip(cfg, _FakeCensor(), client=_client_for(handler))
    verdict, line = asyncio.run(quip.judge_and_quip(event))
    assert verdict is not None
    assert verdict.category == "zombie"
    assert verdict.costume == "zombie bride"
    assert line in body["l"]


def test_judge_and_quip_no_crops_returns_none_none(cfg):
    quip = Quip(cfg, _FakeCensor(), client=_client_for(lambda r: httpx.Response(200)))
    verdict, line = asyncio.run(quip.judge_and_quip(Event(t_trigger=1.0)))
    assert verdict is None
    assert line is None
