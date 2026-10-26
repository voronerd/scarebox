"""Prompts and JSON schemas for the `judge` and `quip` VLM calls.

Spec references: §4.3 (component contracts), §5.3 (persona prompt, draft),
§6.2 (categories -- the join key between judge and the line library),
§7 (content-safety policy).
"""

from __future__ import annotations

# §6.2 -- the fixed category list. `judge.category` MUST be one of these.
CATEGORIES: tuple[str, ...] = (
    "superhero",
    "princess",
    "fairy_tutu",
    "witch_wizard",
    "ghost",
    "skeleton",
    "zombie",
    "vampire",
    "mummy",
    "pumpkin",
    "monster",
    "dinosaur",
    "animal",
    "inflatable",
    "food",
    "pirate",
    "robot_space",
    "movie_character",
    "sports_uniform",
    "homemade_box",
    "group",
    "no_costume",
    "dog",
    "stroller",
    "mystery",
)

# --- judge ---------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are judging ONE photo of a runner (or small group) at a children's Halloween fun run. "
    "Look only at the costume: what it is, how it is made, any props. Never comment on the "
    "person's body, size, face, age, skin, hair, gender, or ability (spec section 7). "
    "Return a single JSON object, and nothing else, using these SHORT keys ONLY (terse on "
    "purpose, so your answer decodes fast -- never use longer key names): "
    '{"c": str, "k": str, "f": float, "i": bool, "n": int, "d": str}. '
    '"c" (costume) is a short noun phrase for the costume (e.g. "inflatable T. rex"), or "none" '
    "if there is no costume. "
    '"k" (category) MUST be exactly one of: ' + ", ".join(CATEGORIES) + ". "
    '"f" (confidence) is 0.0-1.0: how sure you are of the costume identification. '
    '"i" (is_costume) is false for a plain runner or spectator in ordinary clothes. '
    '"n" (group_size) is how many people the photo is about (1 for a single runner). '
    '"d" (details) is AT MOST 6 WORDS naming one concrete visual specific a joke could use '
    '(e.g. "holding a juice box"), never about the person\'s body or appearance.'
)

JUDGE_USER_TEXT = "Judge the costume in this photo. Return JSON only, no prose."


def judge_messages(image_data_urls: list[str]) -> list[dict]:
    """Chat messages for a judge call.

    `image_data_urls` are 1 (single runner) to 3 (group -- the largest crops, spec §4.3) `data:`
    URLs.
    """
    content: list[dict] = [{"type": "text", "text": JUDGE_USER_TEXT}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


# 03-02 (01-05 addendum carried forward): short keys cut completion tokens materially since
# the model emits the literal key text in the JSON it generates -- c/k/f/i/n/d for
# costume/category/confidence/is_costume/group_size/details. Targets judge p95 <= 900ms /
# <= 30 completion tokens (01-03 measured 55 tokens / 1513ms p95 on the old long-key schema at
# the same ~45 tok/s decode rate). Only the WIRE shape is terse -- `judge.py` maps straight
# back onto the long-named `Verdict` dataclass, so nothing downstream of `Judge.judge()` sees
# these short keys.
JUDGE_JSON_SCHEMA: dict = {
    "name": "verdict",
    "schema": {
        "type": "object",
        "properties": {
            "c": {"type": "string"},
            "k": {"type": "string", "enum": list(CATEGORIES)},
            "f": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "i": {"type": "boolean"},
            "n": {"type": "integer", "minimum": 1},
            "d": {"type": "string"},
        },
        "required": ["c", "k", "f", "i", "n", "d"],
        "additionalProperties": False,
    },
}

_VERDICT_REQUIRED = ("c", "k", "f", "i", "n", "d")


def validate_verdict(obj: object) -> tuple[bool, str]:
    """Structural validation of a judge response against the terse §4.3 Verdict WIRE contract
    (short keys c/k/f/i/n/d -- see `JUDGE_JSON_SCHEMA`'s comment).

    Returns `(valid, reason)`; `reason` is `""` when valid.
    """
    if not isinstance(obj, dict):
        return False, "not a JSON object"
    for key in _VERDICT_REQUIRED:
        if key not in obj:
            return False, f"missing '{key}'"
    if not isinstance(obj["c"], str):
        return False, "'c' (costume) is not a string"
    if obj["k"] not in CATEGORIES:
        return False, f"invalid category '{obj['k']}'"
    conf = obj["f"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        return False, "'f' (confidence) is not a number"
    if not (0.0 <= float(conf) <= 1.0):
        return False, "'f' (confidence) is not in [0, 1]"
    if not isinstance(obj["i"], bool):
        return False, "'i' (is_costume) is not a bool"
    gs = obj["n"]
    if isinstance(gs, bool) or not isinstance(gs, int) or gs < 1:
        return False, "'n' (group_size) is not a positive int"
    if not isinstance(obj["d"], str):
        return False, "'d' (details) is not a string"
    return True, ""


# --- judge, lean schema (03-03 lever 2) -----------------------------------
# `judge.schema: lean` (03-03, default): drops "f" (confidence -- 03-02's live-components
# run showed it uncorrelated with correctness, and `judge.min_confidence: 0.0` already makes
# it a no-op), "i" (is_costume -- derived downstream as `category != "no_costume"`) and "n"
# (group_size -- `perceive` already counts track_ids, so asking the model to re-guess it from
# pixels is redundant token spend). Only "c"/"k"/"d" remain. `judge.schema: full` keeps the
# 03-02 six-key schema above for comparison; both wire shapes are mapped onto the same
# long-named `Verdict` dataclass in `judge.py`, so nothing downstream of `Judge.judge()` has
# to care which one was used on the wire.
# Standout (2026-10-20): which costume in a crowd gets the line. Checked against 150 labelled
# people from Martin's test video: a 3 was a must-announce costume 22/27 times, where a
# costume-pack match (the old signal) was 37/118.
STANDOUT_RUBRIC = (
    '"s" (standout) is how much a crowd would point at this costume: 3 = an inflatable, a '
    "famous named character (Spider-Man, Elsa, a Minion, Mario), a full uniform (firefighter, "
    "police), or a big, elaborate or homemade build; 2 = a clear but ordinary costume (witch, "
    "skeleton, cape, ninja, animal onesie, generic superhero); 1 = barely a costume or none "
    "(ears, a hat, face paint, a themed T-shirt, ordinary clothes)."
)

LEAN_JUDGE_SYSTEM_PROMPT = (
    "You are judging ONE photo of a runner (or small group) at a children's Halloween fun run. "
    "Look only at the costume: what it is, how it is made, any props. Never comment on the "
    "person's body, size, face, age, skin, hair, gender, or ability (spec section 7). "
    "Return a single JSON object, and nothing else, using these SHORT keys ONLY (terse on "
    "purpose, so your answer decodes fast -- never use longer key names): "
    '{"c": str, "k": str, "d": str, "s": int}. '
    '"c" (costume) is a short noun phrase for the costume (e.g. "inflatable T. rex"), or "none" '
    'if there is no costume -- use category "no_costume" in that case. '
    '"k" (category) MUST be exactly one of: ' + ", ".join(CATEGORIES) + ". "
    '"d" (details) is AT MOST 6 WORDS naming one concrete visual specific a joke could use '
    "(e.g. \"holding a juice box\"), never about the person's body or appearance. "
    + STANDOUT_RUBRIC
)

LEAN_JUDGE_USER_TEXT = JUDGE_USER_TEXT


def lean_judge_messages(image_data_urls: list[str]) -> list[dict]:
    """Chat messages for a judge call under `judge.schema: lean` -- same image-handling
    convention as `judge_messages`, terser system prompt (no confidence/is_costume/
    group_size asked for)."""
    content: list[dict] = [{"type": "text", "text": LEAN_JUDGE_USER_TEXT}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": LEAN_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


LEAN_JUDGE_JSON_SCHEMA: dict = {
    "name": "lean_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "c": {"type": "string"},
            "k": {"type": "string", "enum": list(CATEGORIES)},
            "d": {"type": "string"},
            "s": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["c", "k", "d", "s"],
        "additionalProperties": False,
    },
}

_LEAN_VERDICT_REQUIRED = ("c", "k", "d")


def validate_lean_verdict(obj: object) -> tuple[bool, str]:
    """Structural validation of a lean judge response (`c`/`k`/`d` only -- see
    `LEAN_JUDGE_JSON_SCHEMA`). Returns `(valid, reason)`; `reason` is `""` when valid."""
    if not isinstance(obj, dict):
        return False, "not a JSON object"
    for key in _LEAN_VERDICT_REQUIRED:
        if key not in obj:
            return False, f"missing '{key}'"
    if not isinstance(obj["c"], str):
        return False, "'c' (costume) is not a string"
    if obj["k"] not in CATEGORIES:
        return False, f"invalid category '{obj['k']}'"
    if not isinstance(obj["d"], str):
        return False, "'d' (details) is not a string"
    return True, ""


# --- quip ------------------------------------------------------------------

MAX_WORDS = 15  # config/scarebox.example.yaml quip.max_words

# spec §5.3, draft (tuned in Phase 3). Few-shot examples per category family are added then too.
PERSONA_PROMPT = (
    "You are Sir Reginald Pumpernickel-Fortescue, a gloriously pompous English gentleman "
    "adjudicating a children's Halloween fun run from a deckchair beside the LAST STRETCH, just "
    f"before the finish line. You deliver ONE short line (max {MAX_WORDS} words) to a runner: a "
    "playful jab about their COSTUME that GOADS them home, pantomime style ('Not long to go!', "
    "'Get a move on!', 'Run, don't crawl!', 'Chop chop, the finish awaits!'). Tone: theatrical, "
    "haughty, affectionate, silly, like a pantomime dame crossed with a snooty art critic. "
    "Always name the costume. Goad the PACE upward, but never call anyone slow, lazy, unfit, "
    "tired or behind, and never mention the person's body, size, face, age, skin, hair, gender "
    "or ability. No insults that would upset a seven-year-old. No scary threats, no romance, no "
    "brand slogans. Never mention or joke about alcohol, drinking, bars, smoking, vaping, drugs, "
    "or gambling -- this is a children's event. "
    'Return JSON: {"lines": ["...", "..."]}.'
)

NUM_CANDIDATES = 2  # 03-02: 2 not 3 -- cuts completion tokens, 2 is enough to pick a survivor


# Hub amendment, 02-02 Task 3 (judge calibration decision): ~10% of costumed-run crops get judged
# `is_costume: false`. A no_costume line must read the same for everyone, so it must never
# mention or mock the absence of a costume. 02-03 applies the same constant by hand when
# writing the no_costume library lines (library/manifest.json has no "quip call" at bake time).
NO_COSTUME_RULE = (
    "Lines for runners judged to have no costume must be warm, neutral encouragement or "
    "playful finish-line goading ('Nearly there, get a move on!') that would suit anyone, "
    "costumed or not. Never mention, question or joke about the absence of a costume."
)


def quip_messages(verdict: dict, few_shots: list[str] | None = None) -> list[dict]:
    """Chat messages for a text-only quip call from a judge Verdict.

    When `verdict["is_costume"]` is False, `NO_COSTUME_RULE` is appended to the system
    prompt (hub amendment, 02-02 Task 3) -- `is_costume` defaults to True (a costume) when
    absent, since this call's primary callers always set it from a real Verdict.

    `few_shots` (03-02, optional so older callers/tests can omit it): 6-8 APPROVED (or
    PENDING, if the category has no approved lines yet) library lines for the same category,
    shown as worked examples so the model matches the established voice (spec §6) instead of
    inventing its own register from the persona prompt alone.
    """
    system = PERSONA_PROMPT
    if not verdict.get("is_costume", True):
        system = f"{PERSONA_PROMPT} {NO_COSTUME_RULE}"
    examples = ""
    if few_shots:
        bulleted = " ".join(f'"{line}"' for line in few_shots)
        examples = f" Lines in this exact voice that were approved before: {bulleted}. "
    user = (
        f"The costume is: {verdict.get('costume', 'unknown')}. "
        f"Category: {verdict.get('category', 'mystery')}. "
        f"Detail to use if it helps: {verdict.get('details', '')}."
        f"{examples}"
        f" Give {NUM_CANDIDATES} candidate lines."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


QUIP_JSON_SCHEMA: dict = {
    "name": "quip_lines",
    "schema": {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": NUM_CANDIDATES,
                "maxItems": NUM_CANDIDATES,
            }
        },
        "required": ["lines"],
        "additionalProperties": False,
    },
}


def validate_quip(obj: object) -> tuple[bool, str]:
    """Structural validation of a quip response: a JSON object with exactly NUM_CANDIDATES
    string lines (03-02: 2, down from 3 -- see NUM_CANDIDATES)."""
    if not isinstance(obj, dict) or "lines" not in obj:
        return False, "missing 'lines'"
    lines = obj["lines"]
    if not isinstance(lines, list) or len(lines) != NUM_CANDIDATES:
        return False, f"'lines' is not a {NUM_CANDIDATES}-item list"
    for line in lines:
        if not isinstance(line, str) or not line.strip():
            return False, "a line is not a non-empty string"
    return True, ""


# --- judge_and_quip (single-call fallback) --------------------------------------------------
# 01-03 flagged this as needed: neither 4B Qwen3-VL variant clears the 2s judge+quip budget
# across two separate round-trips (judge+quip p95 measured 3.1s). `quip.single_call: true`
# (03-02) switches the orchestrator to ONE call that returns both the terse verdict AND the
# candidate lines, trading a slightly bigger single prompt/response for skipping a whole
# extra network round-trip -- lever (2) in spec §4.5's ordered list, after `best_crop_frames:
# 1`. Uses the same short verdict keys (c/k/f/i/n/d) plus "l" for the two candidate lines.

JUDGE_AND_QUIP_SYSTEM_PROMPT = (
    "You are Sir Reginald Pumpernickel-Fortescue, judging a photo of a runner (or small "
    "group) on the last stretch of a children's Halloween fun run AND writing the line you would "
    "say to them, in one step: a playful jab about the costume that goads them to the finish "
    "('Not long to go!', 'Get a move on!'). Look only at the costume; goad the pace upward but "
    "never call anyone slow, lazy, unfit or tired, and never comment on the person's body, size, "
    "face, age, skin, hair, gender, or ability. No romance, no scary threats, no brand slogans, no "
    "alcohol/drinking/smoking/vaping/drugs/gambling jokes -- this is a children's event. If a "
    "runner has no costume ('i' is false), the lines must be warm neutral encouragement that "
    "never mentions or questions the absence of a costume. "
    "Return a single JSON object, nothing else, with SHORT keys only: "
    '{"c": str, "k": str, "f": float, "i": bool, "n": int, "d": str, "l": [str, str]}. '
    '"c"=costume (short noun phrase, or "none"). '
    '"k"=category, exactly one of: ' + ", ".join(CATEGORIES) + ". "
    '"f"=confidence 0.0-1.0. "i"=is_costume (false for plain clothes). '
    '"n"=group_size (people in photo, 1 or more). "d"=at most 6 words of one visual detail. '
    f'"l"=exactly {NUM_CANDIDATES} candidate heckle lines (max {MAX_WORDS} words each, '
    "theatrical and haughty, naming the costume)."
)


def judge_and_quip_messages(image_data_urls: list[str]) -> list[dict]:
    """Chat messages for the single-call judge+quip fallback. `image_data_urls`: 1-3 crops,
    same convention as `judge_messages`."""
    content: list[dict] = [
        {"type": "text", "text": "Judge the costume and write the lines. JSON only, no prose."}
    ]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": JUDGE_AND_QUIP_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


JUDGE_AND_QUIP_JSON_SCHEMA: dict = {
    "name": "verdict_and_lines",
    "schema": {
        "type": "object",
        "properties": {
            "c": {"type": "string"},
            "k": {"type": "string", "enum": list(CATEGORIES)},
            "f": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "i": {"type": "boolean"},
            "n": {"type": "integer", "minimum": 1},
            "d": {"type": "string"},
            "l": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": NUM_CANDIDATES,
                "maxItems": NUM_CANDIDATES,
            },
        },
        "required": ["c", "k", "f", "i", "n", "d", "l"],
        "additionalProperties": False,
    },
}


def validate_judge_and_quip(obj: object) -> tuple[bool, str]:
    """Structural validation of a combined judge+quip response: the terse Verdict shape plus
    an `l` array of NUM_CANDIDATES non-empty lines."""
    ok, reason = validate_verdict(obj)
    if not ok:
        return ok, reason
    lines = obj.get("l")
    if not isinstance(lines, list) or len(lines) != NUM_CANDIDATES:
        return False, f"'l' is not a {NUM_CANDIDATES}-item list"
    for line in lines:
        if not isinstance(line, str) or not line.strip():
            return False, "a line in 'l' is not a non-empty string"
    return True, ""


def line_word_count(line: str) -> int:
    return len(line.split())


_STOPWORDS = {"the", "and", "with", "none", "unknown"}


def costume_noun_in_line(costume: str, line: str) -> bool:
    """Spec §4.3: quip output 'must contain the costume noun'.

    Lenient check: any content word (length > 2, not a stopword) from `costume` appears
    case-insensitively in `line`. Returns True (vacuously) if `costume` has no content words.
    """
    words = [w.strip(".,!?'\"").lower() for w in costume.split()]
    words = [w for w in words if len(w) > 2 and w not in _STOPWORDS]
    if not words:
        return True
    lower_line = line.lower()
    return any(w in lower_line for w in words)
