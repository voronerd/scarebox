#!/usr/bin/env python3
"""01-04 Task 2: TTS bake-off. Runs INSIDE the `scarebox/tts-bench:latest` container built from
`deploy/bench/Dockerfile.tts` (see that file's header for the `docker build`/`docker run`
invocation -- the repo is mounted live at /workspace, weights at /models:ro).

Generates the Kokoro reference clip, then times every candidate engine/voice on the same ten
lines: model load time (once per engine/voice), time-to-first-audio (streaming where the
library supports it, else full synth time), and real-time factor. Writes raw + BarkerFX WAVs
under runs/tts/<engine>_<voice>/, and runs/evidence/tts-bakeoff.json with the aggregated stats.

Each engine is wrapped in try/except at import, load and generate time -- a failure anywhere
is recorded (not raised), so one broken candidate never takes down the others (plan rule).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scarebox.fx import BarkerFX  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODELS = Path("/models")
RUNS = REPO / "runs" / "tts"
EVIDENCE = REPO / "runs" / "evidence" / "tts-bakeoff.json"
LIBRARY_VOICE = REPO / "library" / "voice"

TARGET_SR = 24000

REFERENCE_TEXT = (
    "I say, what an absolutely splendid afternoon for a perambulation. "
    "One does so adore a pageant."
)

# Ten lines: a mix of short exclamations and full ~15-word Sir Reginald lines (spec Sec5.3).
TEST_LINES = [
    "A pirate! How frightfully nautical.",
    "Good heavens, a superhero -- do mind the cape on the refreshment table, won't you.",
    "A dinosaur, extinct for sixty-five million years and still better dressed than most.",
    "No costume at all? Bold. Audacious. Faintly embarrassing for the rest of us.",
    "A robot! I do hope you've brought spare batteries for the encore.",
    "Why, a ghost -- how delightfully translucent of you to attend at all.",
    "A princess, resplendent, and not a single pumpkin carriage in sight. Shame.",
    "Steady on -- a zombie, shuffling with more grace than most adults manage sober.",
    "A mummy, wrapped to perfection, quite the triumph of amateur bandaging.",
    "And a dog in a costume! Sir, you have won the entire afternoon.",
]


def _p(values, pct):
    if not values:
        return None
    return float(np.percentile(values, pct))


def _write_pair(out_dir: Path, idx: int, audio: np.ndarray, sr: int, fx: BarkerFX):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{idx}.wav"
    fx_path = out_dir / f"{idx}_fx.wav"
    sf.write(raw_path, audio, sr)
    fx_audio = fx.process(np.asarray(audio, dtype=np.float32))
    sf.write(fx_path, fx_audio, sr)


def gpu_mem_gib() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024**3), 2)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------------------------
# Kokoro
# --------------------------------------------------------------------------------------------


def run_kokoro(result: dict):
    kokoro_dir = MODELS / "Kokoro-82M"
    try:
        import torch
        from kokoro import KModel, KPipeline
    except Exception as e:
        result["kokoro"] = {"install_blocked": f"import failed: {e}"}
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.monotonic()
    try:
        model = (
            KModel(config=str(kokoro_dir / "config.json"), model=str(kokoro_dir / "kokoro-v1_0.pth"))
            .to(device)
            .eval()
        )
        pipeline = KPipeline(lang_code="b", model=model)
    except Exception as e:
        result["kokoro"] = {"install_blocked": f"load failed: {e}\n{traceback.format_exc()}"}
        return None, None
    load_s = time.monotonic() - t0

    voices = ["bm_george", "bm_lewis", "bm_fable", "bm_daniel"]

    def synth(text: str, voice: str):
        voice_path = str(kokoro_dir / "voices" / f"{voice}.pt")
        t_start = time.monotonic()
        ttfa = None
        chunks = []
        for _gs, _ps, audio in pipeline(text, voice=voice_path):
            if ttfa is None:
                ttfa = time.monotonic() - t_start
            chunks.append(audio)
        total = time.monotonic() - t_start
        audio = np.concatenate([np.asarray(c) for c in chunks]) if chunks else np.zeros(0)
        return ttfa if ttfa is not None else total, total, audio

    return load_s, (synth, voices)


# --------------------------------------------------------------------------------------------
# Piper (pinned piper-tts==1.2.0, the last MIT release -- see Dockerfile.tts header)
# --------------------------------------------------------------------------------------------


def run_piper(result: dict):
    try:
        from piper import PiperVoice
    except Exception as e:
        result["piper"] = {"install_blocked": f"import failed: {e}"}
        return None, None

    # Only en_GB male voice staged that both (a) clears a permissive licence check on its
    # MODEL_CARD and (b) is single-speaker (so "which speaker is male" isn't ambiguous):
    #   - alan: MODEL_CARD says "License: See URL" -> the linked LICENSE file says
    #     "Copyright 2022 Mycroft AI All Rights Reserved" -- NOT permissive. Dropped.
    #   - aru (12 speakers) / vctk (109 speakers): CC-BY-4.0, permissive, but voices.json's
    #     speaker_id_map has no gender field and the VCTK/ARU corpora's per-speaker gender
    #     lists aren't staged locally -- resolving "which of 12/109 IDs are male" is out of
    #     scope for this budgeted bake-off. Noted as a deviation, not measured.
    #   - semaine: CC-BY-NC-SA 4.0 (NonCommercial) -- excluded on licence.
    #   - alba / cori / jenny_dioco / southern_english_female: female voices, not candidates.
    #   - northern_english_male (medium): CC-BY-SA 4.0, single speaker, male. KEPT.
    voice_dirs = {
        "northern_english_male": MODELS
        / "piper-voices/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx",
    }
    loaded = {}
    load_s = {}
    for name, onnx_path in voice_dirs.items():
        t0 = time.monotonic()
        try:
            loaded[name] = PiperVoice.load(str(onnx_path))
            load_s[name] = time.monotonic() - t0
        except Exception as e:
            result.setdefault("piper", {})[name] = {"install_blocked": f"load failed: {e}"}

    def synth(text: str, voice: str):
        pv = loaded[voice]
        t_start = time.monotonic()
        ttfa = None
        pcm_chunks = []
        sr = TARGET_SR
        for chunk in pv.synthesize(text):
            if ttfa is None:
                ttfa = time.monotonic() - t_start
            sr = getattr(chunk, "sample_rate", TARGET_SR)
            pcm_chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        total = time.monotonic() - t_start
        if pcm_chunks:
            pcm = np.concatenate(pcm_chunks)
            audio = (pcm.astype(np.float32) / 32768.0)
        else:
            audio = np.zeros(0, dtype=np.float32)
            sr = TARGET_SR
        return (ttfa if ttfa is not None else total), total, audio, sr

    return load_s, (synth, list(loaded.keys()))


# --------------------------------------------------------------------------------------------
# Chatterbox Turbo
# --------------------------------------------------------------------------------------------


def run_chatterbox(result: dict, reference_wav: Path):
    try:
        import torch
        from chatterbox.tts import ChatterboxTTS
    except Exception as e:
        result["chatterbox"] = {"install_blocked": f"import failed: {e}"}
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = MODELS / "chatterbox-turbo"
    t0 = time.monotonic()
    model = None
    for attempt in ("from_local", "from_pretrained"):
        try:
            if attempt == "from_local" and hasattr(ChatterboxTTS, "from_local"):
                model = ChatterboxTTS.from_local(str(ckpt_dir), device=device)
            elif attempt == "from_pretrained":
                model = ChatterboxTTS.from_pretrained(device=device)
            if model is not None:
                break
        except Exception as e:
            result.setdefault("chatterbox", {})[f"{attempt}_error"] = (
                f"{e}\n{traceback.format_exc()[-1500:]}"
            )
    if model is None:
        result.setdefault("chatterbox", {})["install_blocked"] = "both from_local/from_pretrained failed"
        return None, None
    load_s = time.monotonic() - t0

    exaggerations = [0.5, 0.8]

    def synth(text: str, exaggeration: float):
        t_start = time.monotonic()
        wav = model.generate(
            text, audio_prompt_path=str(reference_wav), exaggeration=exaggeration, cfg_weight=0.5
        )
        total = time.monotonic() - t_start
        audio = wav.squeeze().detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        sr = getattr(model, "sr", TARGET_SR)
        return total, total, audio, sr  # non-streaming API: ttfa == full synth time

    return load_s, (synth, exaggerations)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    LIBRARY_VOICE.mkdir(parents=True, exist_ok=True)
    result: dict = {"voices": [], "errors": {}}

    # --- Reference clip (Kokoro bm_george) -- must happen before chatterbox needs it. ---
    load_s, kokoro_bits = run_kokoro(result)
    reference_wav = LIBRARY_VOICE / "reference.wav"
    if kokoro_bits is not None:
        synth, voices = kokoro_bits
        try:
            ttfa, total, audio = synth(REFERENCE_TEXT, "bm_george")
            sf.write(reference_wav, audio, TARGET_SR)
            dur = len(audio) / TARGET_SR
            print(f"reference.wav written: {dur:.1f}s audio, synth {total:.2f}s", file=sys.stderr)
        except Exception as e:
            result["errors"]["reference_clip"] = f"{e}\n{traceback.format_exc()[-1000:]}"

        fx = BarkerFX(TARGET_SR, "barker")
        for voice in voices:
            ttfas, rtfs = [], []
            voice_dir = RUNS / f"kokoro_{voice}"
            for i, line in enumerate(TEST_LINES):
                try:
                    ttfa, total, audio = synth(line, voice)
                except Exception as e:
                    result["errors"][f"kokoro_{voice}_{i}"] = str(e)
                    continue
                dur = len(audio) / TARGET_SR
                rtf = dur / total if total > 0 else None
                ttfas.append(ttfa * 1000)
                if rtf is not None:
                    rtfs.append(rtf)
                _write_pair(voice_dir, i, audio, TARGET_SR, fx)
            result["voices"].append(
                {
                    "engine": "kokoro",
                    "voice": voice,
                    "load_s": load_s,
                    "n_ok": len(ttfas),
                    "ttfa_p50_ms": _p(ttfas, 50),
                    "ttfa_p95_ms": _p(ttfas, 95),
                    "rtf_p50": _p(rtfs, 50),
                    "gpu_mem_gib": gpu_mem_gib(),
                }
            )
    else:
        result["errors"]["reference_clip"] = "kokoro unavailable, reference.wav NOT generated"

    if not reference_wav.exists():
        result["errors"]["chatterbox"] = "skipped: no reference.wav (kokoro failed)"
    else:
        # --- Piper ---
        load_s, piper_bits = run_piper(result)
        if piper_bits is not None:
            synth, voices = piper_bits
            fx = BarkerFX(TARGET_SR, "barker")
            for voice in voices:
                ttfas, rtfs = [], []
                voice_dir = RUNS / f"piper_{voice}"
                this_load_s = load_s.get(voice) if isinstance(load_s, dict) else load_s
                for i, line in enumerate(TEST_LINES):
                    try:
                        ttfa, total, audio, sr = synth(line, voice)
                    except Exception as e:
                        result["errors"][f"piper_{voice}_{i}"] = str(e)
                        continue
                    dur = len(audio) / sr if sr else 0
                    rtf = dur / total if total > 0 else None
                    ttfas.append(ttfa * 1000)
                    if rtf is not None:
                        rtfs.append(rtf)
                    fx_sr = BarkerFX(sr, "barker") if sr != TARGET_SR else fx
                    _write_pair(voice_dir, i, audio, sr, fx_sr)
                result["voices"].append(
                    {
                        "engine": "piper",
                        "voice": voice,
                        "load_s": this_load_s,
                        "n_ok": len(ttfas),
                        "ttfa_p50_ms": _p(ttfas, 50),
                        "ttfa_p95_ms": _p(ttfas, 95),
                        "rtf_p50": _p(rtfs, 50),
                        "gpu_mem_gib": gpu_mem_gib(),
                    }
                )

        # --- Chatterbox Turbo ---
        load_s, cb_bits = run_chatterbox(result, reference_wav)
        if cb_bits is not None:
            synth, exaggerations = cb_bits
            fx = BarkerFX(TARGET_SR, "barker")
            for exag in exaggerations:
                ttfas, rtfs = [], []
                voice_name = f"exag{exag}"
                voice_dir = RUNS / f"chatterbox_{voice_name}"
                for i, line in enumerate(TEST_LINES):
                    try:
                        ttfa, total, audio, sr = synth(line, exag)
                    except Exception as e:
                        result["errors"][f"chatterbox_{voice_name}_{i}"] = str(e)
                        continue
                    dur = len(audio) / sr if sr else 0
                    rtf = dur / total if total > 0 else None
                    ttfas.append(ttfa * 1000)
                    if rtf is not None:
                        rtfs.append(rtf)
                    fx_sr = BarkerFX(sr, "barker") if sr != TARGET_SR else fx
                    _write_pair(voice_dir, i, audio, sr, fx_sr)
                result["voices"].append(
                    {
                        "engine": "chatterbox",
                        "voice": voice_name,
                        "load_s": load_s,
                        "n_ok": len(ttfas),
                        "ttfa_p50_ms": _p(ttfas, 50),
                        "ttfa_p95_ms": _p(ttfas, 95),
                        "rtf_p50": _p(rtfs, 50),
                        "gpu_mem_gib": gpu_mem_gib(),
                    }
                )

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"voices": [(v["engine"], v["voice"], v.get("ttfa_p95_ms")) for v in result["voices"]]}))
    print("errors:", json.dumps(result["errors"], indent=2)[:4000], file=sys.stderr)


if __name__ == "__main__":
    main()
