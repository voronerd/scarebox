#!/usr/bin/env python3.12
"""Chatterbox TTS live-line server for the scarebox pipeline.

Serves warm Chatterbox renders over a tiny stdlib HTTP API so the scarebox app
process can get PCM audio for a judged line without paying model-load or
reference-embedding cost per call. One GPU, one model, one lock -- calls are
serialised, not batched (this box has one GPU and the deadline is per-call
latency, not throughput).

Run inside scarebox/tts-chatterbox:bake (see deploy/Dockerfile.chatterbox),
with /models mounted read-only (CB_MODEL) and the repo bind-mounted at /app
so CB_REF resolves against library/voice/reference_excited.wav. Must start
OFFLINE -- nothing here should touch the network after the image is built;
see runs/chatterbox-live/ notes for how that was verified.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

CB_PORT = int(os.environ.get("CB_PORT", "8102"))
CB_MODEL = os.environ.get("CB_MODEL", "/models/chatterbox")
CB_REF = os.environ.get("CB_REF", "/app/library/voice/reference_excited.wav")
CB_EXAGGERATION = float(os.environ.get("CB_EXAGGERATION", "1.3"))
CB_CFG_WEIGHT = float(os.environ.get("CB_CFG_WEIGHT", "0.3"))
# Speed tweaks Martin kept after a blind A/B (2026-10-16, "couldn't tell the difference"):
# T3 (the speech-token decoder, ~75% of render time) in bfloat16 is ~26% faster; S3Gen's
# flow matching at 5 steps instead of 10 is another ~7%. CB_T3_BF16=0 / CB_CFM_STEPS=10 go
# back to stock. The library is rendered through this same server, so live and library match.
CB_T3_BF16 = os.environ.get("CB_T3_BF16", "1") == "1"
CB_CFM_STEPS = int(os.environ.get("CB_CFM_STEPS", "5"))

MAX_TEXT_CHARS = 300

# Two throwaway warm-up lines -- absorbs first-call CUDA kernel compilation /
# cache-fill cost before /health reports ready.
WARMUP_LINES = [
    "Nearly there, runner! Get a move on!",
    "On your marks. Get set. Go, go, go!",
]

_model = None
_sample_rate = None
_gpu_lock = threading.Lock()
_warm = threading.Event()


def _load_and_warm() -> None:
    global _model, _sample_rate
    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[chatterbox-live] device={device}", flush=True)

    t0 = time.time()
    _model = ChatterboxTTS.from_local(CB_MODEL, device=device)
    print(f"[chatterbox-live] model load: {time.time() - t0:.1f}s", flush=True)

    if CB_T3_BF16:
        _model.t3 = _model.t3.to(dtype=torch.bfloat16)
    t0 = time.time()
    _model.prepare_conditionals(CB_REF, exaggeration=CB_EXAGGERATION)
    if CB_T3_BF16:
        _model.conds.t3 = _model.conds.t3.to(dtype=torch.bfloat16)
    print(f"[chatterbox-live] prepare_conditionals: {time.time() - t0:.1f}s", flush=True)
    print(f"[chatterbox-live] t3_bf16={CB_T3_BF16} cfm_steps={CB_CFM_STEPS}", flush=True)
    _sample_rate = _model.sr

    for line in WARMUP_LINES:
        t0 = time.time()
        _render(line)
        print(
            f"[chatterbox-live] warmup render ({len(line)} chars): "
            f"{(time.time() - t0) * 1000:.0f}ms",
            flush=True,
        )

    _warm.set()
    print("[chatterbox-live] WARM", flush=True)


def _generate(text: str):
    """chatterbox 0.1.7's ChatterboxTTS.generate(), inlined so two things the stock method
    can't do work: S3Gen gets `n_cfm_timesteps`, and a rebuilt T3 condition keeps T3's dtype
    (stock rebuilds it in float32 whenever the stored exaggeration differs, and a bf16-rounded
    1.3 always differs). Sampling settings are the stock defaults. Ported from the A/B
    harness (runs/cb-speed/cb_bench.py on gbox1) that made the clips Martin approved."""
    import torch.nn.functional as F
    from chatterbox.models.s3tokenizer import drop_invalid_tokens
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.tts import punc_norm

    m = _model
    if CB_EXAGGERATION != m.conds.t3.emotion_adv[0, 0, 0]:
        cond = m.conds.t3
        m.conds.t3 = T3Cond(
            speaker_emb=cond.speaker_emb,
            cond_prompt_speech_tokens=cond.cond_prompt_speech_tokens,
            emotion_adv=CB_EXAGGERATION * torch.ones(1, 1, 1),
        ).to(device=m.device, dtype=next(m.t3.parameters()).dtype)
    text_tokens = m.tokenizer.text_to_tokens(punc_norm(text)).to(m.device)
    if CB_CFG_WEIGHT > 0.0:
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    text_tokens = F.pad(text_tokens, (1, 0), value=m.t3.hp.start_text_token)
    text_tokens = F.pad(text_tokens, (0, 1), value=m.t3.hp.stop_text_token)
    speech_tokens = m.t3.inference(
        t3_cond=m.conds.t3,
        text_tokens=text_tokens,
        max_new_tokens=1000,
        temperature=0.8,
        cfg_weight=CB_CFG_WEIGHT,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
    )
    speech_tokens = drop_invalid_tokens(speech_tokens[0])
    speech_tokens = speech_tokens[speech_tokens < 6561].to(m.device)
    wav, _ = m.s3gen.inference(
        speech_tokens=speech_tokens, ref_dict=m.conds.gen, n_cfm_timesteps=CB_CFM_STEPS
    )
    wav = wav.squeeze(0).detach().cpu().numpy()
    return m.watermarker.apply_watermark(wav, sample_rate=m.sr)


def _render(text: str) -> tuple:
    """Render text with the precomputed conditionals. Returns (float32 mono PCM at
    _model.sr, render_ms)."""
    t0 = time.time()
    with torch.inference_mode():
        wav = _generate(text)
    render_ms = (time.time() - t0) * 1000.0
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return wav, render_ms


class Handler(BaseHTTPRequestHandler):
    server_version = "chatterbox-live/1.0"

    def log_message(self, fmt, *args):  # noqa: A003 -- stdlib signature
        print(f"[chatterbox-live] {self.address_string()} {fmt % args}", flush=True)

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if _warm.is_set():
                self._send_json(200, {"ok": True})
            else:
                self._send_json(503, {"ok": False, "reason": "warming"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/tts":
            self._send_json(404, {"error": "not found"})
            return
        if not _warm.is_set():
            self._send_json(503, {"error": "warming"})
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid json body"})
            return

        text = (body.get("text") or "").strip()
        if not text:
            self._send_json(400, {"error": "text is empty"})
            return
        if len(text) > MAX_TEXT_CHARS:
            self._send_json(400, {"error": f"text exceeds {MAX_TEXT_CHARS} chars"})
            return

        with _gpu_lock:
            try:
                wav, render_ms = _render(text)
            except Exception as exc:  # noqa: BLE001 -- surface render failure as 500, never hang
                print(f"[chatterbox-live] render error: {exc!r}", flush=True)
                self._send_json(500, {"error": "render failed", "detail": str(exc)})
                return

        pcm = wav.astype("<f4").tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(pcm)))
        self.send_header("X-Sample-Rate", str(_sample_rate))
        self.send_header("X-Render-Ms", f"{render_ms:.1f}")
        self.end_headers()
        self.wfile.write(pcm)


def main() -> None:
    _load_and_warm()
    httpd = ThreadingHTTPServer(("127.0.0.1", CB_PORT), Handler)
    print(f"[chatterbox-live] listening on 127.0.0.1:{CB_PORT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
