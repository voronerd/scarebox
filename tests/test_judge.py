import asyncio
import dataclasses
import json

import httpx
import numpy as np
import pytest

from scarebox.config import load_config
from scarebox.events import Event
from scarebox.judge import Judge


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("CAMERA_HOST", "x")
    monkeypatch.setenv("CAMERA_USER", "x")
    monkeypatch.setenv("CAMERA_PASSWORD", "x")
    monkeypatch.setenv("AIM_TOKEN", "x")
    return load_config("config/bench.yaml")


def _event(n_crops: int = 1) -> Event:
    crops = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(n_crops)]
    return Event(t_trigger=1.0, track_ids=list(range(n_crops)), group_size=n_crops, crops=crops)


def _verdict_body(**overrides) -> dict:
    # Terse wire keys (03-02): c/k/f/i/n/d for costume/category/confidence/is_costume/
    # group_size/details -- see prompts.JUDGE_JSON_SCHEMA.
    body = {
        "c": "inflatable T. rex",
        "k": "dinosaur",
        "f": 0.9,
        "i": True,
        "n": 1,
        "d": "holding a juice box",
    }
    body.update(overrides)
    return {"choices": [{"message": {"content": json.dumps(body)}}]}


def _client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test/v1")


def test_valid_verdict(cfg):
    # Full six-key schema (03-02): exercises the confidence/is_costume/group_size wire
    # fields the lean schema (config/bench.yaml's default since 03-03) doesn't ask for.
    cfg = dataclasses.replace(cfg, judge=dataclasses.replace(cfg.judge, schema="full"))

    def handler(request):
        return httpx.Response(200, json=_verdict_body())

    judge = Judge(cfg, client=_client_for(handler))
    verdict = asyncio.run(judge.judge(_event()))
    assert verdict is not None
    assert verdict.category == "dinosaur"
    assert verdict.costume == "inflatable T. rex"
    assert verdict.confidence == pytest.approx(0.9)


def test_lean_schema_is_the_bench_default(cfg):
    assert cfg.judge.schema == "lean"


def test_lean_schema_derives_is_costume_and_group_size(cfg):
    # Lean wire shape (03-03 lever 2): only c/k/d -- confidence/is_costume/group_size are
    # never asked for; is_costume is derived from category, group_size from the event (not
    # re-guessed from pixels), confidence is a fixed 1.0 (min_confidence is pinned to 0.0 in
    # both shipped configs, so this is a no-op either way).
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
            {"c": "pirate hat", "k": "pirate", "d": "wooden sword"}
        )}}]})

    judge = Judge(cfg, client=_client_for(handler))
    verdict = asyncio.run(judge.judge(_event(n_crops=3)))
    assert verdict is not None
    assert verdict.category == "pirate"
    assert verdict.is_costume is True
    assert verdict.group_size == 3  # from the event (3 crops), not asked of the model
    assert verdict.confidence == pytest.approx(1.0)


@pytest.mark.parametrize("wire, expected", [(3, 3), (1, 1), (None, 0), (7, 0), (True, 0), ("3", 0)])
def test_lean_schema_reads_the_standout_rating_and_tolerates_junk(cfg, wire, expected):
    # The eye-catching rating decides who gets the line in a crowd; a missing or odd one
    # means "not rated", never a thrown-away verdict.
    body = {"c": "inflatable T. rex", "k": "inflatable", "d": "tiny arms"}
    if wire is not None:
        body["s"] = wire

    def handler(request):
        sent = json.loads(request.content)
        assert "s" in sent["response_format"]["json_schema"]["schema"]["required"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    verdict = asyncio.run(Judge(cfg, client=_client_for(handler)).judge(_event()))
    assert verdict is not None and verdict.standout == expected


def test_lean_schema_no_costume_derives_is_costume_false(cfg):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
            {"c": "none", "k": "no_costume", "d": ""}
        )}}]})

    judge = Judge(cfg, client=_client_for(handler))
    verdict = asyncio.run(judge.judge(_event()))
    assert verdict is not None
    assert verdict.category == "no_costume"
    assert verdict.is_costume is False


def test_lean_schema_group_with_costume_none_is_not_costumed(cfg):
    # 03-03: a "group" verdict with costume "none" produced a live "bare-faced runners?" jab,
    # because is_costume came out True and NO_COSTUME_RULE was never applied.
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
            {"c": "none", "k": "group", "d": "running in tank tops"}
        )}}]})

    judge = Judge(cfg, client=_client_for(handler))
    verdict = asyncio.run(judge.judge(_event()))
    assert verdict is not None
    assert verdict.category == "group"
    assert verdict.is_costume is False


def test_invalid_category_returns_none(cfg):
    def handler(request):
        return httpx.Response(200, json=_verdict_body(k="not_a_real_category"))

    judge = Judge(cfg, client=_client_for(handler))
    verdict = asyncio.run(judge.judge(_event()))
    assert verdict is None


def test_timeout_returns_none(cfg):
    def handler(request):
        raise httpx.TimeoutException("simulated timeout")

    judge = Judge(cfg, client=_client_for(handler))
    verdict = asyncio.run(judge.judge(_event()))
    assert verdict is None


def test_no_crops_returns_none(cfg):
    def handler(request):  # pragma: no cover -- should never be called
        raise AssertionError("judge should not call out with no crops")

    judge = Judge(cfg, client=_client_for(handler))
    verdict = asyncio.run(judge.judge(Event(t_trigger=1.0)))
    assert verdict is None


def test_low_confidence_routes_to_mystery(cfg):
    # config/bench.yaml's real min_confidence is 0.0 (scripts/calibrate_judge.py: confidence is
    # uncorrelated with correctness, so the shipped threshold never trips) -- override it here
    # to exercise the routing mechanism itself, which judge.py still implements for a future
    # threshold that does correlate. Also forces `schema: full` -- the lean schema (03-03
    # default) never asks for confidence at all, so this routing is a full-schema-only path.
    cfg = dataclasses.replace(
        cfg, judge=dataclasses.replace(cfg.judge, min_confidence=0.4, schema="full")
    )

    def handler(request):
        return httpx.Response(200, json=_verdict_body(f=0.1, k="pirate"))

    judge = Judge(cfg, client=_client_for(handler))
    verdict = asyncio.run(judge.judge(_event()))
    assert verdict is not None
    assert verdict.category == "mystery"
    assert verdict.costume == ""
    assert verdict.confidence == pytest.approx(0.1)


def test_confidence_is_clamped(cfg):
    def handler(request):
        return httpx.Response(200, json=_verdict_body(f=1.5))

    judge = Judge(cfg, client=_client_for(handler))
    verdict = asyncio.run(judge.judge(_event()))
    assert verdict is not None
    assert verdict.confidence == 1.0
