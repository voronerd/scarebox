"""Helpers in scripts/ingest_candidates.py (2026-10-16): the dedupe and "pack line names its
costume" gates the big line expansion relies on. The censor path is the live Censor, covered
by tests/test_censor.py."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ingest_candidates", Path(__file__).resolve().parents[1] / "scripts" / "ingest_candidates.py"
)
ic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ic)

SPIDEY = {"display": "Spider-Man", "aliases": ["spiderman", "spider man"]}


def test_names_costume_accepts_display_or_alias():
    assert ic.names_costume("Web-sling home, Spider-Man!", SPIDEY)
    assert ic.names_costume("Spiderman! Swing to the finish!", SPIDEY)


def test_names_costume_rejects_generic_line():
    assert not ic.names_costume("Swing home, hero! Not long now!", SPIDEY)


def test_is_dup_catches_near_duplicates_only():
    seen = [ic.tokens("Get a move on, Spider-Man, please!")]
    assert ic.is_dup("Get a move on, Spider-Man!", seen)
    assert not ic.is_dup("Thwip thwip! The finish line is your villain!", seen)


def test_round_robin_alternates_sources():
    out = ic.round_robin({"a": ["a1", "a2"], "b": ["b1"]})
    assert [s for s, _ in out] == ["a", "b", "a"]


def test_clean_straightens_typography_the_censor_rejects():
    assert ic.clean("A pirate — on MY racecourse… Touché!") == "A pirate, on MY racecourse... Touche!"
    assert ic.clean("“Hurry, old bean”") == "Hurry, old bean"
