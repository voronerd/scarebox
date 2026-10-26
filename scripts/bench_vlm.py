#!/usr/bin/env python3
"""Bench a VLM's judge+quip latency and accuracy against the 01-02 silver-labelled crops.

Runs against an OpenAI-compatible chat completions endpoint (vLLM on gbox1, 127.0.0.1:8100) --
see deploy/bench/compose.vlm.yaml. Pillow is imported lazily (only needed to actually run a
bench, not for --help), since it is not installed in the CPU-only dev venv.

Usage:
    scripts/bench_vlm.py --endpoint http://127.0.0.1:8100/v1 --model qwen3-vl-4b \
        --crops testdata/crops --labels testdata/crops/labels.jsonl \
        --sizes 512,384 --out runs/vlm/qwen3-vl-4b.jsonl
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

from scarebox.prompts import (  # noqa: E402
    JUDGE_JSON_SCHEMA,
    QUIP_JSON_SCHEMA,
    judge_messages,
    quip_messages,
    validate_quip,
    validate_verdict,
)


class BenchHTTPError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Bench a VLM's judge+quip latency and category accuracy against silver-labelled "
            "crops (01-03 Task 2)."
        )
    )
    p.add_argument("--endpoint", default="http://127.0.0.1:8100/v1",
                    help="OpenAI-compatible base URL (default: %(default)s)")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--crops", default="testdata/crops", help="directory of crop jpgs")
    p.add_argument("--labels", default="testdata/crops/labels.jsonl", help="silver labels jsonl")
    p.add_argument("--sizes", default="512,384", help="comma-separated long-side px sizes to bench")
    p.add_argument("--out", required=True, help="output jsonl path")
    p.add_argument("--warmup", type=int, default=3, help="warm-up calls discarded per size")
    p.add_argument("--limit", type=int, default=None, help="cap number of labelled crops (smoke test)")
    p.add_argument("--timeout", type=float, default=60.0, help="per-HTTP-call timeout, seconds")
    p.add_argument("--no-guided", action="store_true",
                    help="skip guided_json structured output from the start")
    return p.parse_args(argv)


def load_labels(path: str | Path) -> list[dict]:
    """Load silver labels, dropping records marked `category: unusable` (01-02)."""
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("category") == "unusable":
                continue
            records.append(rec)
    return records


def load_and_encode(path: Path, long_side: int) -> str:
    """Resize (long side, downscale-only) and base64-JPEG-encode a crop as a `data:` URL."""
    from PIL import Image  # lazy: not needed for --help

    with Image.open(path) as img:
        img.load()
        w, h = img.size
        m = max(w, h)
        if m > long_side:
            scale = long_side / m
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def extract_json(text: str) -> dict | None:
    """Best-effort JSON object extraction: strips code fences, then brace-matches."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _chat_url(endpoint: str) -> str:
    return endpoint.rstrip("/") + "/chat/completions"


def call_judge_stream(
    client: httpx.Client, endpoint: str, model: str, image_url: str, timeout: float, guided: bool
) -> dict:
    """Streaming judge call. Returns ttft_ms, judge_ms, raw text, token usage (if offered)."""
    body = {
        "model": model,
        "messages": judge_messages([image_url]),
        "max_tokens": 250,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if guided:
        body["guided_json"] = JUDGE_JSON_SCHEMA["schema"]
    t0 = time.monotonic()
    ttft_ms = None
    parts: list[str] = []
    usage: dict = {}
    with client.stream("POST", _chat_url(endpoint), json=body, timeout=timeout) as resp:
        if resp.status_code != 200:
            resp.read()
            raise BenchHTTPError(resp.status_code, resp.text)
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    if ttft_ms is None:
                        ttft_ms = (time.monotonic() - t0) * 1000.0
                    parts.append(content)
    judge_ms = (time.monotonic() - t0) * 1000.0
    return {
        "ttft_ms": ttft_ms,
        "judge_ms": judge_ms,
        "text": "".join(parts),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def call_quip(
    client: httpx.Client, endpoint: str, model: str, verdict: dict, timeout: float, guided: bool
) -> dict:
    """Non-streaming, text-only quip call. Returns quip_ms, raw text, token usage."""
    body = {
        "model": model,
        "messages": quip_messages(verdict),
        "max_tokens": 150,
        "temperature": 0.9,
    }
    if guided:
        body["guided_json"] = QUIP_JSON_SCHEMA["schema"]
    t0 = time.monotonic()
    resp = client.post(_chat_url(endpoint), json=body, timeout=timeout)
    quip_ms = (time.monotonic() - t0) * 1000.0
    if resp.status_code != 200:
        raise BenchHTTPError(resp.status_code, resp.text)
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {}) or {}
    return {
        "quip_ms": quip_ms,
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def run_one(ctx: dict, client: httpx.Client, args: argparse.Namespace, rec: dict, size: int) -> dict:
    """Run one judge+quip pair for one labelled crop at one size. Never raises."""
    result: dict = {
        "file": rec["file"],
        "label_category": rec.get("category"),
        "label_costume": rec.get("costume"),
        "label_is_costume": rec.get("is_costume"),
        "size": size,
        "ttft_ms": None,
        "judge_ms": None,
        "quip_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "quip_prompt_tokens": None,
        "quip_completion_tokens": None,
        "verdict": None,
        "lines": None,
        "judge_json_valid": False,
        "quip_json_valid": False,
        "guided": ctx["guided"],
        "error": None,
    }

    crop_path = Path(args.crops) / rec["file"]
    if not crop_path.exists():
        result["error"] = f"crop not found: {crop_path}"
        return result

    try:
        image_url = load_and_encode(crop_path, size)
    except Exception as exc:  # noqa: BLE001 -- record and move on, never crash the bench
        result["error"] = f"image load failed: {exc}"
        return result

    try:
        jr = call_judge_stream(client, args.endpoint, args.model, image_url, args.timeout, ctx["guided"])
    except BenchHTTPError as exc:
        if ctx["guided"]:
            # Soft downgrade: this image/model may not support guided_json. Try once more plain.
            ctx["guided"] = False
            result["guided"] = False
            try:
                jr = call_judge_stream(client, args.endpoint, args.model, image_url, args.timeout, False)
            except Exception as exc2:  # noqa: BLE001
                result["error"] = f"judge failed (after guided downgrade): {exc2}"
                return result
        else:
            result["error"] = f"judge failed: {exc}"
            return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"judge failed: {exc}"
        return result

    result["ttft_ms"] = jr["ttft_ms"]
    result["judge_ms"] = jr["judge_ms"]
    result["prompt_tokens"] = jr["prompt_tokens"]
    result["completion_tokens"] = jr["completion_tokens"]

    verdict = extract_json(jr["text"])
    result["verdict"] = verdict
    if verdict is None:
        result["error"] = "verdict not parseable JSON"
        return result
    ok, reason = validate_verdict(verdict)
    result["judge_json_valid"] = ok
    if not ok:
        result["error"] = f"verdict invalid: {reason}"

    try:
        qr = call_quip(client, args.endpoint, args.model, verdict, args.timeout, ctx["guided"])
    except Exception as exc:  # noqa: BLE001
        prefix = (result["error"] + "; ") if result["error"] else ""
        result["error"] = f"{prefix}quip failed: {exc}"
        return result

    result["quip_ms"] = qr["quip_ms"]
    result["quip_prompt_tokens"] = qr["prompt_tokens"]
    result["quip_completion_tokens"] = qr["completion_tokens"]

    lines_obj = extract_json(qr["text"])
    result["lines"] = lines_obj.get("lines") if isinstance(lines_obj, dict) else None
    ok, reason = validate_quip(lines_obj if isinstance(lines_obj, dict) else {})
    result["quip_json_valid"] = ok
    if not ok and not result["error"]:
        result["error"] = f"quip invalid: {reason}"
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels = load_labels(args.labels)
    if args.limit:
        labels = labels[: args.limit]
    if not labels:
        print("no usable labelled crops found", file=sys.stderr)
        return 1
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = {"guided": not args.no_guided}

    with httpx.Client() as client:
        warm_crop = labels[0]
        for size in sizes:
            for _ in range(max(0, args.warmup)):
                run_one(ctx, client, args, warm_crop, size)

        count = 0
        with out_path.open("w") as fh:
            for size in sizes:
                for rec in labels:
                    result = run_one(ctx, client, args, rec, size)
                    fh.write(json.dumps(result) + "\n")
                    fh.flush()
                    count += 1
                    if result["error"]:
                        print(f"[{args.model} @ {size}] {rec['file']}: {result['error']}", file=sys.stderr)

    print(f"wrote {count} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
