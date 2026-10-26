"""judge: VLM costume classifier. Spec §4.3, §4.4 ("Verdict" contract).

`Judge(cfg)` makes ONE chat-completion call per `Event` to the OpenAI-compatible VLM
server at `cfg.judge.endpoint`, JPEG-encoding the event's best crop(s) as `data:` URLs and
asking for JSON-schema-constrained output (`response_format: {"type": "json_schema", ...}`,
`prompts.JUDGE_JSON_SCHEMA` -- vLLM structured output, spec §5.1/§5.2) so a syntactically
valid response always parses.

Any error, timeout, or structurally invalid response returns `None` and logs `judge.fail`
with the reason -- the orchestrator then falls back to the library's `mystery` line (spec
§4.4 FALLBACK). A *low-confidence* response is NOT an error: it is a successful verdict
whose `category` is routed to `mystery` before it's returned (hub amendment to 02-02 Task 1,
WWMD W1 -- a wrong-category library line is the "bad line worse than no line" ADR 0001
warns about; a generic `mystery` line fits everyone). The threshold is
`judge.min_confidence`, calibrated against the silver crops by `scripts/calibrate_judge.py`.
"""

from __future__ import annotations

import io
import json
import logging

import httpx
import numpy as np
from PIL import Image

from scarebox.config import Config
from scarebox.events import Event, Verdict
from scarebox.prompts import (
    JUDGE_JSON_SCHEMA,
    LEAN_JUDGE_JSON_SCHEMA,
    judge_messages,
    lean_judge_messages,
    validate_lean_verdict,
    validate_verdict,
)

logger = logging.getLogger(__name__)

# Spec §4.3: a group sends the 2 or 3 largest crops in one request; a single runner sends 1.
_MAX_CROPS_PER_CALL = 3


def _encode_data_url(crop: np.ndarray) -> str:
    """BGR ndarray -> a `data:image/jpeg;base64,...` URL (PIL expects RGB)."""
    rgb = crop[:, :, ::-1] if crop.ndim == 3 and crop.shape[2] == 3 else crop
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="JPEG", quality=85)
    import base64

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class Judge:
    """`client` is injectable so tests can pass an `httpx.AsyncClient` wired to an
    `httpx.MockTransport` instead of hitting a real server."""

    def __init__(self, cfg: Config, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg.judge
        self._client = client if client is not None else httpx.AsyncClient(base_url=self.cfg.endpoint)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def judge(self, event: Event) -> Verdict | None:
        if not event.crops:
            logger.warning("judge.fail event=%s reason=no_crops", event.id)
            return None

        try:
            image_urls = [_encode_data_url(c) for c in event.crops[:_MAX_CROPS_PER_CALL]]
        except Exception as exc:  # noqa: BLE001 -- any encode failure is a judge failure
            logger.warning("judge.fail event=%s reason=encode_error:%s", event.id, exc)
            return None

        lean = self.cfg.schema == "lean"
        if lean:
            messages = lean_judge_messages(image_urls)
            json_schema = LEAN_JUDGE_JSON_SCHEMA
        else:
            messages = judge_messages(image_urls)
            json_schema = JUDGE_JSON_SCHEMA

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": json_schema},
            "temperature": 0.0,
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
            logger.warning("judge.fail event=%s reason=timeout", event.id)
            return None
        except Exception as exc:  # noqa: BLE001 -- network error, bad JSON, bad shape, ...
            logger.warning("judge.fail event=%s reason=%s", event.id, exc)
            return None

        if lean:
            ok, reason = validate_lean_verdict(obj)
            if not ok:
                logger.warning("judge.fail event=%s reason=invalid_verdict:%s", event.id, reason)
                return None
            category = obj["k"]
            costume = obj["c"]
            details = obj["d"]
            # Lean schema (03-03 lever 2) doesn't ask the model for confidence, is_costume or
            # group_size -- confidence is a fixed 1.0 (min_confidence is pinned at 0.0 in both
            # shipped configs per the 02-02 calibration finding that confidence was
            # uncorrelated with correctness anyway, so this is a no-op either way); is_costume
            # is derived from category; group_size comes from `perceive` (the event itself),
            # not re-guessed from pixels.
            confidence = 1.0
            # A group (or any category) whose costume text is "none" is not costumed either: 03-03
            # caught a live "A group of bare-faced runners?" line because is_costume came out True
            # for category "group" with costume "none", so NO_COSTUME_RULE was never applied.
            is_costume = category != "no_costume" and costume.strip().lower() not in {
                "", "none", "no costume", "n/a", "unknown"
            }
            group_size = event.group_size
            # Optional on the wire: a missing or odd rating just means "not rated" (0), never
            # a discarded verdict.
            s = obj.get("s")
            standout = s if isinstance(s, int) and not isinstance(s, bool) and 1 <= s <= 3 else 0
            return Verdict(
                costume=costume,
                category=category,
                confidence=confidence,
                is_costume=is_costume,
                group_size=group_size,
                details=details,
                standout=standout,
            )

        # Clamp confidence before structural validation: an out-of-range confidence is a
        # minor model quirk to tolerate, not a reason to discard an otherwise-valid verdict.
        # `f` is the terse wire key for confidence (03-02 -- see prompts.JUDGE_JSON_SCHEMA).
        if isinstance(obj, dict) and isinstance(obj.get("f"), (int, float)):
            obj["f"] = max(0.0, min(1.0, float(obj["f"])))

        ok, reason = validate_verdict(obj)
        if not ok:
            logger.warning("judge.fail event=%s reason=invalid_verdict:%s", event.id, reason)
            return None

        # obj uses the terse wire keys (c/k/f/i/n/d); map back onto the long-named Verdict
        # dataclass here so nothing downstream of this function ever sees the short keys.
        confidence = float(obj["f"])
        category = obj["k"]
        costume = obj["c"]
        if confidence < self.cfg.min_confidence:
            # Hub amendment: don't trust the category, route to the safe generic line.
            costume = ""
            category = "mystery"

        return Verdict(
            costume=costume,
            category=category,
            confidence=confidence,
            is_costume=obj["i"],
            group_size=obj["n"],
            details=obj["d"],
        )
