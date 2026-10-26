# The scarebox app container: ingest + perceive + judge/quip/censor clients (HTTP to the
# model-server containers) + voice (03-02: TTS runs IN-PROCESS here, on the GPU -- not a
# separate server). Built FROM the pinned PyTorch base (deploy/images.env IMAGE_TORCH) --
# never reinstall torch/torchvision/torchaudio from PyPI on this aarch64 host; the base
# image's CUDA-matched build is the only one that works (02-01 KNOWN FACTS, confirmed again
# by 01-04's torchaudio ABI mismatch).
#
# TTS packages (03-02): installed EXACTLY as they built in 01-04's bench image
# (deploy/bench/Dockerfile.tts) -- Kokoro + misaki[en] `--no-deps` first (kokoro's own
# requires are already satisfied by the base image), then its real deps explicitly, then the
# en_core_web_sm spaCy wheel baked in at BUILD time (OFFLINE TRAP, found in the 01-04 TTS spike: misaki's
# English G2P pip-installs this model at first use otherwise, which fails with no internet
# at race time). Chatterbox and Piper are NOT included here -- both install-blocked/GPL per
# docs/MODELS.md; Kokoro is the only engine `voice.py` supports.
# espeak-ng: system binary misaki[en] needs for phonemization.
#
# Build (from repo root, on gbox1):
#   docker build -f deploy/Dockerfile.app \
#     --build-arg IMAGE_TORCH=$(grep ^IMAGE_TORCH deploy/images.env | cut -d= -f2) \
#     -t scarebox/app:dev .
ARG IMAGE_TORCH
FROM ${IMAGE_TORCH}

WORKDIR /app

# opencv-python-headless (not opencv-python: avoid pulling in GUI deps nobody needs in a
# headless container) and av (PyAV; its aarch64 wheel ships its own FFmpeg libs, so no
# system ffmpeg package is needed for decode -- 02-01 KNOWN FACTS) are the two heavy new
# pieces. supervision brings sv.Detections + sv.ByteTrack (both Apache-2.0/MIT); the
# Roboflow `trackers` package itself is NOT installed here -- by
# design: supervision's built-in ByteTrack covers the same algorithm and avoids a
# second, unverified tracker API on a tight build-and-measure budget. `yolox` (the pip
# package) is also NOT installed -- its C++ NMS extension doesn't reliably build on aarch64,
# so only the model definition is vendored (scarebox/vendor/yolox/), using
# torchvision.ops.batched_nms (confirmed present in this base image) instead of a custom NMS.
#
# Both `trackers` and `supervision` are --no-deps here because their default installs would
# otherwise pull a second, non-headless `opencv-python` (conflicting with
# opencv-python-headless) -- their actual runtime deps (beyond torch/numpy/scipy, already in
# the base image) are installed explicitly on the line below.
# alsa-utils (04-01, hub amendment): ships `aplay`, the CLI `stage.AlsaSink` shells out to.
# Without it the image has no way to make a sound AT ALL -- `stage._make_sink` silently fell
# back to WavSink when `aplay` was missing (02-02 bug), which on race day means a box that
# runs perfectly and never says a word. `which aplay` must resolve inside this image now.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends espeak-ng alsa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        opencv-python-headless \
        av \
        scipy \
        soundfile \
        httpx \
        pyyaml \
        pillow \
        rich \
        tqdm \
        pydeprecate \
        defusedxml \
        matplotlib \
        requests \
    && pip install --no-cache-dir --no-deps supervision

# Kokoro + misaki[en] (03-02, mirrors deploy/bench/Dockerfile.tts exactly -- see that file's
# comments for why each package is --no-deps / explicit).
RUN pip install --no-cache-dir --no-deps kokoro \
    && pip install --no-cache-dir huggingface-hub loguru "misaki[en]" transformers

# OFFLINE TRAP: bake en_core_web_sm in at build time so misaki's G2P never needs network at
# runtime (`--network none` smoke-tested by tests/test_voice.py's container verify step).
RUN pip install --no-cache-dir \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl \
    && python -c "import spacy; spacy.load('en_core_web_sm')"

COPY pyproject.toml README.md /app/
COPY scarebox /app/scarebox
COPY tests /app/tests
COPY config /app/config

RUN pip install --no-cache-dir -e /app[dev]

# No CMD yet: `scarebox run` (the orchestrator entrypoint) doesn't exist until a later phase.
# `docker run scarebox/app:dev <cmd>` (e.g. pytest) is how this image is used for now.
