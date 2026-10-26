from scarebox.prompts import (
    CATEGORIES,
    JUDGE_JSON_SCHEMA,
    NO_COSTUME_RULE,
    QUIP_JSON_SCHEMA,
    costume_noun_in_line,
    judge_messages,
    line_word_count,
    quip_messages,
    validate_quip,
    validate_verdict,
)


def test_categories_cover_spec_list():
    # spec §6.2 -- the full fixed category list.
    assert len(CATEGORIES) == 25
    for cat in ("no_costume", "group", "mystery", "stroller", "dinosaur", "princess", "superhero",
                "inflatable"):
        assert cat in CATEGORIES


def test_valid_verdict_passes():
    # Terse wire keys (03-02): c/k/f/i/n/d -- see prompts.JUDGE_JSON_SCHEMA.
    verdict = {"c": "inflatable T. rex", "k": "dinosaur", "f": 0.9, "i": True, "n": 1, "d": "holding a juice box"}
    ok, reason = validate_verdict(verdict)
    assert ok, reason


def test_invalid_category_rejected():
    verdict = {"c": "inflatable T. rex", "k": "not_a_real_category", "f": 0.9, "i": True, "n": 1, "d": "holding a juice box"}
    ok, reason = validate_verdict(verdict)
    assert not ok
    assert "category" in reason


def test_verdict_missing_field_rejected():
    verdict = {"c": "ghost", "k": "ghost", "f": 0.5, "i": True}
    ok, reason = validate_verdict(verdict)
    assert not ok
    assert "n" in reason


def test_verdict_confidence_out_of_range_rejected():
    verdict = {"c": "ghost", "k": "ghost", "f": 1.5, "i": True, "n": 1, "d": ""}
    ok, _ = validate_verdict(verdict)
    assert not ok


def test_verdict_not_a_dict_rejected():
    ok, reason = validate_verdict(["not", "a", "dict"])
    assert not ok
    assert "not a JSON object" in reason


def test_valid_quip_passes():
    quip = {
        "lines": [
            "Ah, a dashing dinosaur, roar most magnificently!",
            "Onward, mighty T. rex, the finish line awaits!",
        ]
    }
    ok, reason = validate_quip(quip)
    assert ok, reason


def test_quip_wrong_count_rejected():
    ok, _ = validate_quip({"lines": ["Only one line"]})
    assert not ok


def test_quip_missing_key_rejected():
    ok, _ = validate_quip({"not_lines": []})
    assert not ok


def test_quip_empty_line_rejected():
    ok, _ = validate_quip({"lines": ["Fine line here", "", "Another fine line"]})
    assert not ok


def test_costume_noun_detection():
    assert costume_noun_in_line("inflatable T. rex", "What a marvellous rex you are!")
    assert not costume_noun_in_line("inflatable T. rex", "What a marvellous day for a run!")


def test_line_word_count():
    assert line_word_count("one two three") == 3
    assert line_word_count("") == 0


def test_judge_schema_enum_matches_categories():
    # Terse wire keys (03-02): "k" is category, "c"/"f"/"i"/"n"/"d" the rest.
    assert JUDGE_JSON_SCHEMA["schema"]["properties"]["k"]["enum"] == list(CATEGORIES)
    assert JUDGE_JSON_SCHEMA["schema"]["required"] == ["c", "k", "f", "i", "n", "d"]


def test_quip_schema_fixes_two_lines():
    lines_schema = QUIP_JSON_SCHEMA["schema"]["properties"]["lines"]
    assert lines_schema["minItems"] == 2
    assert lines_schema["maxItems"] == 2


def test_judge_messages_includes_images_and_system_prompt():
    messages = judge_messages(["data:image/jpeg;base64,AAAA"])
    assert messages[0]["role"] == "system"
    assert "costume" in messages[0]["content"]
    user_content = messages[1]["content"]
    image_parts = [c for c in user_content if c["type"] == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == "data:image/jpeg;base64,AAAA"


def test_quip_messages_carries_verdict_fields():
    messages = quip_messages({"costume": "zombie bride", "category": "zombie", "details": "torn veil"})
    assert messages[0]["role"] == "system"
    assert "zombie bride" in messages[1]["content"]
    assert "torn veil" in messages[1]["content"]


def test_quip_messages_adds_no_costume_rule_when_not_costumed():
    messages = quip_messages(
        {"costume": "none", "category": "no_costume", "details": "", "is_costume": False}
    )
    assert NO_COSTUME_RULE in messages[0]["content"]


def test_quip_messages_omits_no_costume_rule_when_costumed():
    messages = quip_messages(
        {"costume": "zombie bride", "category": "zombie", "details": "", "is_costume": True}
    )
    assert NO_COSTUME_RULE not in messages[0]["content"]


def test_quip_messages_defaults_is_costume_true_when_absent():
    # verdict dicts built before `is_costume` existed (or hand-written in tests) shouldn't
    # silently pick up the no_costume rule.
    messages = quip_messages({"costume": "zombie bride", "category": "zombie", "details": ""})
    assert NO_COSTUME_RULE not in messages[0]["content"]
