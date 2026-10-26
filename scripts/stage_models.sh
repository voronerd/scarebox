#!/usr/bin/env bash
# Download candidate model weights into ~/models on the execution host (run while online).
# Idempotent: hf download skips files already present. Licences are recorded in docs/MODELS.md.
set -euo pipefail

MODELS="${SCAREBOX_MODELS:-$HOME/models}"
HF="${HF_BIN:-$HOME/.venv/bin/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$MODELS"

dl() {  # repo [extra hf args...]
    local repo="$1"; shift
    echo "== $repo"
    "$HF" download "$repo" --local-dir "$MODELS/$(basename "$repo")" "$@" >/dev/null
    du -sh "$MODELS/$(basename "$repo")"
}

# vision-language candidates (Apache-2.0)
dl Qwen/Qwen3-VL-4B-Instruct
dl Qwen/Qwen3-VL-8B-Instruct
dl HuggingFaceTB/SmolVLM2-2.2B-Instruct
# FP8 variants: GB10 decode is memory-bandwidth bound, so half the bytes ~ double the tokens/s
dl Qwen/Qwen3-VL-4B-Instruct-FP8
dl Qwen/Qwen3-VL-8B-Instruct-FP8
dl Qwen/Qwen3-VL-2B-Instruct-FP8
# guard model (Apache-2.0)
dl ibm-granite/granite-guardian-3.2-3b-a800m
# TTS (Apache-2.0 / MIT)
dl hexgrad/Kokoro-82M
dl ResembleAI/chatterbox-turbo
dl rhasspy/piper-voices --include "en/en_GB/*" --include "voices.json"
# detector weights (Apache-2.0, Megvii YOLOX release)
mkdir -p "$MODELS/yolox"
[ -s "$MODELS/yolox/yolox_s.pth" ] || curl -fsSL -o "$MODELS/yolox/yolox_s.pth" \
    https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
du -sh "$MODELS/yolox"
echo "== done"; du -sh "$MODELS"
