# Models and licences

Every model, weight file and voice used by scarebox, with its licence. Only permissive licences
(Apache-2.0, MIT, BSD, CC0) are allowed (ADR 0001 §1d). Status: `candidate` (staged for the
Phase 1 bake-off), `selected` (in use), `rejected` (with reason).

| Role | Model | Source | Revision | Licence | Status |
|---|---|---|---|---|---|
| Vision (judge + quip) | Qwen3-VL-4B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct | ebb281ec70b0 | Apache-2.0 | rejected: matches the FP8 build's accuracy (0.57 category acc) at double the latency (judge+quip p95 6036ms vs 3083ms) (Phase 1 VLM bake-off) |
| Vision (judge + quip) | Qwen3-VL-8B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct | 0c351dd01ed8 | Apache-2.0 | rejected: skipped unmeasured (BF16) -- predicted to decode about half the 4B's rate for no accuracy gain, confirmed by the 8B-FP8 row below |
| Vision (judge + quip) | **Qwen3-VL-4B-Instruct-FP8** | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8 | fefbb44cbcce | Apache-2.0 | **selected**: judge p50/p95 1292/1513ms, judge+quip p95 3083ms, decode 44.8 tok/s, category acc 0.57 (ties the BF16 build, halves its latency). Still ~1.5x over the judge+quip budget allocation -- flags "single-call fallback needed" (spec §4.1 one-call option re-activated for 03-02) |
| Vision (judge + quip) | Qwen3-VL-8B-Instruct-FP8 | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8 | 9cdc6310a8cb | Apache-2.0 | rejected: same category acc (0.57) as 4B-FP8 at roughly half the speed (judge+quip p95 5676ms) |
| Vision (judge + quip) | Qwen3-VL-2B-Instruct-FP8 | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8 | 46485250d885 | Apache-2.0 | rejected: accuracy collapse (0.15 category acc, 61% quip-schema failure) -- a visual-grounding problem with this model/quant, not a usable speed trade |
| Vision (speed floor) | SmolVLM2-2.2B-Instruct | https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct | 482adb537c02 | Apache-2.0 | rejected: not run -- the vLLM image's SmolVLM processor import fails on a missing `num2words`; not worth a custom image given the 2B-FP8 accuracy collapse above |
| Vision | Gemma-4-26B-A4B (Q8_0 GGUF already on the dev host) | Google | — | Gemma terms (unconfirmed) | candidate only if the licence checks out as permissive |
| Guard (censor) | granite-guardian-3.2-3b-a800m | https://huggingface.co/ibm-granite/granite-guardian-3.2-3b-a800m | 3de033d89b49 | Apache-2.0 | candidate |
| TTS | Chatterbox Turbo | https://huggingface.co/ResembleAI/chatterbox-turbo | 749d1c1a46eb | MIT | rejected: install-blocked on gbox1 (aarch64) -- `torchaudio` pulled in transitively at a mismatched CUDA ABI (13.0 vs the base image's 13.4) (Phase 1 TTS bake-off) |
| TTS | Kokoro-82M (voices bm_george, bm_lewis, bm_fable, bm_daniel) | https://huggingface.co/hexgrad/Kokoro-82M | f3ff3571791e | Apache-2.0 | selected (provisional, gate G1 pending): `bm_george`, arbitrary tie-break -- time-to-first-audio p50 60ms / p95 83ms, ~89x real-time; all four voices measure identically on speed |
| TTS | Piper en_GB voices | ~~https://huggingface.co/rhasspy/piper-voices~~ (archived 2025-10-06) | c10ece1aade4 | **Licence changed since staging**: `pip install piper-tts` now installs OHF-Voice/piper1-gpl, **GPL-3.0-or-later** (relicensed 2025-07-10, piper-tts>=1.3.0) -- banned by ADR 0001 Sec1d. The last MIT release (`piper-tts==1.2.0`) has no aarch64 wheel for its `piper-phonemize` dependency. | rejected: GPL on the version that runs here, no aarch64 build on the version that's MIT (Phase 1 TTS bake-off) |
| Detector | YOLOX-s | https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0 | 0.1.1rc0 | Apache-2.0 | selected (unmeasured -- 02-01 benches detect+track latency; revisit if it misses the under-50ms target) |
| Tracker | ByteTrack (Roboflow `trackers`) | https://github.com/roboflow/trackers | pin in Phase 2 | Apache-2.0 | candidate |

**sha256:** not separately computed for the picks above. The Revision column is already
a content-addressed git commit hash from the source repo (HuggingFace revision or GitHub tag),
which pins the same content a file-level sha256 would. Adding a sha256 of the downloaded weight
blobs on gbox1 is a follow-up if Martin wants belt-and-suspenders verification, not done here.

**Excluded on licence:** Ultralytics YOLO (AGPL-3.0), Coqui XTTS-v2 (CPML, non-commercial),
F5-TTS (CC-BY-NC-4.0), Spotify pedalboard (GPL-3.0; the FX chain is numpy/scipy instead),
`piper-tts>=1.3.0`/OHF-Voice piper1-gpl (GPL-3.0-or-later, relicensed 2025-07-10 -- the MIT
`piper-tts==1.2.0` that predates it has no aarch64 wheel for `piper-phonemize`, so Piper is
excluded on this hardware regardless of which version is picked).

**Voice ethics:** the character voice is cloned from a synthetic reference clip generated with
Kokoro. No real person's voice is used.

**Runtime extra (not a staged weight):** `misaki[en]` (Kokoro's English G2P) auto-downloads
`en_core_web_sm` (spaCy's small English model, MIT, ~13MB) at bench time if it isn't cached --
allowed once since gbox1 is online (01-04 KNOWN FACTS). The bench container runs `--rm`, so
this re-downloads every run instead of persisting in a mounted cache volume; a one-line
follow-up (mount a pip/spacy cache dir), not a blocker.

## Container images

Pinned in `deploy/images.env`. Both are NVIDIA NGC arm64 images (NVIDIA's container licence;
pulled by the user, not redistributed by this repo).
