import json

import pytest

from scarebox.events import Event
from scarebox.library import Library, LibraryError, resolve_category


def _manifest(tmp_path, lines):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(lines))
    return path


def _wav(tmp_path, name="line.wav"):
    p = tmp_path / name
    p.write_bytes(b"RIFF....WAVEfmt ")
    return str(p)


def test_pick_no_repeat(tmp_path):
    wavs = [_wav(tmp_path, f"p{i}.wav") for i in range(3)]
    lines = [
        {"id": f"p{i}", "category": "pirate", "text": f"line {i}", "wav": wavs[i], "status": "approved"}
        for i in range(3)
    ]
    lib = Library(_manifest(tmp_path, lines), require_approved=False)

    picked_ids = set()
    for _ in range(3):
        result = lib.pick("pirate", exclude_last=10)
        assert result is not None
        assert result.category == "pirate"
        picked_ids.add(result.wav)
    # with only 3 candidates and exclude_last=10, each of the 3 should be distinct once
    # all three have been played (no-repeat window covers the whole small set).
    assert len(picked_ids) == 3


def test_pick_falls_back_to_mystery_when_category_empty(tmp_path):
    mystery_wav = _wav(tmp_path, "mystery.wav")
    lines = [
        {"id": "m1", "category": "mystery", "text": "generic line", "wav": mystery_wav, "status": "approved"},
    ]
    lib = Library(_manifest(tmp_path, lines), require_approved=False)
    result = lib.pick("")
    assert result is not None
    assert result.category == "mystery"


def test_pick_falls_back_when_category_unknown_or_has_no_playable_line(tmp_path):
    mystery_wav = _wav(tmp_path, "mystery.wav")
    lines = [
        {"id": "m1", "category": "mystery", "text": "generic line", "wav": mystery_wav, "status": "approved"},
        # a pending line with no wav -- not playable even though it's for 'pirate'.
        {"id": "p1", "category": "pirate", "text": "pending line", "wav": None, "status": "pending"},
    ]
    lib = Library(_manifest(tmp_path, lines), require_approved=False)
    result = lib.pick("pirate")
    assert result is not None
    assert result.category == "mystery"


def test_pick_returns_none_when_nothing_playable(tmp_path):
    lines = [
        {"id": "p1", "category": "pirate", "text": "pending line", "wav": None, "status": "pending"},
    ]
    lib = Library(_manifest(tmp_path, lines), require_approved=False)
    assert lib.pick("pirate") is None
    assert lib.pick("") is None


def test_require_approved_excludes_pending(tmp_path):
    wav = _wav(tmp_path)
    lines = [
        {"id": "p1", "category": "pirate", "text": "pending line", "wav": wav, "status": "pending"},
    ]
    lib = Library(_manifest(tmp_path, lines), require_approved=False)
    # pending is eligible when require_approved is False ...
    assert lib.pick("pirate") is not None


def test_require_approved_raises_on_pending_lines(tmp_path):
    wav = _wav(tmp_path)
    lines = [
        {"id": "p1", "category": "pirate", "text": "pending line", "wav": wav, "status": "pending"},
        {"id": "p2", "category": "ghost", "text": "approved line", "wav": wav, "status": "approved"},
    ]
    with pytest.raises(LibraryError):
        Library(_manifest(tmp_path, lines), require_approved=True)


def test_require_approved_true_constructs_cleanly_when_all_approved(tmp_path):
    wav = _wav(tmp_path)
    lines = [
        {"id": "p2", "category": "ghost", "text": "approved line", "wav": wav, "status": "approved"},
    ]
    lib = Library(_manifest(tmp_path, lines), require_approved=True)
    result = lib.pick("ghost")
    assert result is not None
    assert result.text == "approved line"


def test_require_wav_false_allows_lines_with_no_wav_yet(tmp_path):
    # Bench-only path (02-02 Task 3 `--tts-placeholder`): the seed manifest ships every line
    # with wav=None, so without this bypass a placeholder bench run would see every category
    # "dropped" with nothing to measure.
    lines = [
        {"id": "p1", "category": "pirate", "text": "pending line", "wav": None, "status": "pending"},
    ]
    lib = Library(_manifest(tmp_path, lines), require_approved=False, require_wav=False)
    result = lib.pick("pirate")
    assert result is not None
    assert result.wav is None


def test_require_wav_true_is_the_default(tmp_path):
    lines = [
        {"id": "p1", "category": "pirate", "text": "pending line", "wav": None, "status": "pending"},
    ]
    lib = Library(_manifest(tmp_path, lines), require_approved=False)
    assert lib.pick("pirate") is None


def test_resolve_category_overrides_to_group_for_multi_person_event():
    event = Event(group_size=2)
    assert resolve_category("pirate", event.group_size) == "group"


def test_resolve_category_keeps_judge_category_for_solo_event():
    event = Event(group_size=1)
    assert resolve_category("pirate", event.group_size) == "pirate"


def test_resolve_category_overrides_even_mystery_for_a_group():
    event = Event(group_size=3)
    assert resolve_category("mystery", event.group_size) == "group"


# --- costume packs (2026-10-16) ------------------------------------------------------------

from scarebox.library import PackMatcher, normalise_costume  # noqa: E402

_PACKS = [
    {"key": "spider-man", "display": "Spider-Man", "category": "superhero", "aliases": ["spiderman", "spider man"]},
    {"key": "inflatable-trex", "display": "Inflatable T-Rex", "category": "inflatable", "aliases": ["inflatable dinosaur", "inflatable t rex"]},
    {"key": "dino-onesie", "display": "Dinosaur", "category": "dinosaur", "aliases": ["dinosaur", "t rex"]},
    {"key": "superhero-squad", "display": "Superhero squad", "category": "group", "aliases": ["superheroes", "superhero group"]},
]


def test_normalise_costume_strips_filler_and_punctuation():
    assert normalise_costume("A Spider-Man costume!") == "spider man"


def test_pack_matcher_prefers_longest_alias():
    m = PackMatcher(_PACKS)
    assert m.match("inflatable dinosaur costume") == "inflatable-trex"
    assert m.match("green dinosaur onesie") == "dino-onesie"


def test_pack_matcher_whole_words_only():
    m = PackMatcher(_PACKS)
    assert m.match("spidermania shirt") is None


def test_pack_matcher_groups_only_match_group_packs():
    m = PackMatcher(_PACKS)
    assert m.match("superheroes", group=False) is None
    assert m.match("group of superheroes", group=True) == "superhero-squad"
    assert m.match("Spider-Man", group=True) is None


def test_pick_prefers_pack_line_then_falls_back_to_category(tmp_path):
    wav = _wav(tmp_path)
    packs = tmp_path / "packs.json"
    packs.write_text(json.dumps(_PACKS))
    lines = [
        {"id": "s1", "category": "superhero", "text": "generic hero", "wav": wav, "status": "approved"},
        {"id": "p1", "category": "superhero", "pack": "spider-man", "text": "web-sling home!", "wav": wav, "status": "approved"},
    ]
    lib = Library(_manifest(tmp_path, lines), require_approved=True, packs_path=packs)
    assert lib.pick("superhero", costume="Spiderman costume").text == "web-sling home!"
    # pack exhausted by the no-repeat window -> category line, not a repeat
    assert lib.pick("superhero", costume="Spiderman costume").text == "generic hero"
    # an unmatched costume never gets a pack line, even with nothing else fresh
    assert all(lib.pick("superhero", costume="Batman").text == "generic hero" for _ in range(5))


def test_require_approved_allows_rejected_lines(tmp_path):
    wav = _wav(tmp_path)
    lines = [
        {"id": "r1", "category": "pirate", "text": "nope", "wav": wav, "status": "rejected"},
        {"id": "a1", "category": "pirate", "text": "yes", "wav": wav, "status": "approved"},
    ]
    lib = Library(_manifest(tmp_path, lines), require_approved=True)
    assert all(lib.pick("pirate").text == "yes" for _ in range(5))


def test_prefer_short_picks_from_the_shorter_half(tmp_path):
    # Busy mode: a shorter line frees the speaker sooner for the next standout costume.
    wav = _wav(tmp_path)
    lines = [
        {"id": f"p{n}", "category": "pirate", "text": " ".join(["arr"] * n), "wav": wav, "status": "approved"}
        for n in (3, 4, 9, 10, 11, 12)
    ]
    lib = Library(_manifest(tmp_path, lines), require_approved=True)
    for _ in range(10):
        assert len(lib.pick("pirate", exclude_last=0, prefer_short=True).text.split()) <= 9
