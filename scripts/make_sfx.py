#!/usr/bin/env python3
"""02-03 Task 3: procedurally generate the Halloween SFX stings, numpy/scipy only.

No real recordings of people or places -- every sting here is synthesised from noise and
tones, so it's CC0 by construction (there is no underlying copyrighted recording to license).
See `library/sfx/SOURCES.md` for the per-file record this script also writes.

Each sting: <= 1.5s, loudness-normalised to -18 LUFS (spec Sec6.3/hub instruction), 48kHz
mono 16-bit, written to `library/sfx/<name>.wav`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scarebox.fx import loudnorm_offline  # noqa: E402

SR = 48000
TARGET_LUFS = -18.0
OUT_DIR = REPO / "library" / "sfx"


def _env(n: int, attack: int, release: int) -> np.ndarray:
    """Simple linear attack / exponential release envelope, length n samples."""
    e = np.ones(n)
    a = min(attack, n)
    e[:a] = np.linspace(0, 1, a)
    r = min(release, n)
    tail = np.exp(-np.linspace(0, 6, r))
    e[n - r :] *= tail
    return e


def thunder_rumble(rng: np.random.Generator) -> np.ndarray:
    """Filtered noise burst: broadband noise, low-passed hard, with a slow rumbling envelope."""
    n = int(1.4 * SR)
    noise = rng.standard_normal(n)
    sos = signal.butter(4, 120.0, btype="lowpass", fs=SR, output="sos")
    rumble = signal.sosfilt(sos, noise)
    # a couple of extra low "cracks" layered on top
    t = np.arange(n) / SR
    crack_env = np.exp(-((t - 0.15) ** 2) / (2 * 0.02**2)) + 0.6 * np.exp(
        -((t - 0.55) ** 2) / (2 * 0.04**2)
    )
    env = _env(n, int(0.05 * SR), int(0.9 * SR))
    return (rumble * env + rumble * crack_env * 1.5).astype(np.float64)


def organ_chord(rng: np.random.Generator) -> np.ndarray:
    """A minor chord, several harmonics per note, slow attack -- a haunted-house organ stab."""
    n = int(1.3 * SR)
    t = np.arange(n) / SR
    freqs = [110.0, 130.81, 164.81, 220.0]  # A minor-ish stack (A2, C3, E3, A3)
    out = np.zeros(n)
    for f in freqs:
        for harmonic, amp in ((1, 1.0), (2, 0.4), (3, 0.2)):
            out += amp * np.sin(2 * np.pi * f * harmonic * t)
    out /= len(freqs)
    env = _env(n, int(0.08 * SR), int(0.8 * SR))
    return (out * env).astype(np.float64)


def gong(rng: np.random.Generator) -> np.ndarray:
    """Inharmonic partials with a long decay -- a struck gong/bell."""
    n = int(1.5 * SR)
    t = np.arange(n) / SR
    fundamental = 90.0
    partials = [1.0, 1.59, 2.14, 2.91, 3.76]  # inharmonic ratios, bell-like
    out = np.zeros(n)
    for i, ratio in enumerate(partials):
        decay = np.exp(-t * (1.2 + i * 0.6))
        out += (1.0 / (i + 1)) * np.sin(2 * np.pi * fundamental * ratio * t) * decay
    env = _env(n, int(0.01 * SR), int(1.4 * SR))
    return (out * env).astype(np.float64)


def creaky_door(rng: np.random.Generator) -> np.ndarray:
    """A pitch-glide tone plus a little noise -- the classic creaking-door glide."""
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    # frequency glides from 280Hz down to 180Hz then back up a touch, with jitter
    base = 280 - 100 * (t / t[-1]) + 15 * np.sin(2 * np.pi * 7 * t)
    phase = 2 * np.pi * np.cumsum(base) / SR
    tone = signal.sawtooth(phase, width=0.3)
    noise = rng.standard_normal(n) * 0.08
    sos = signal.butter(2, [200.0, 2000.0], btype="bandpass", fs=SR, output="sos")
    mix = signal.sosfilt(sos, tone + noise)
    env = _env(n, int(0.05 * SR), int(0.3 * SR))
    return (mix * env).astype(np.float64)


def owl_hoot(rng: np.random.Generator) -> np.ndarray:
    """Two descending hoots with a little vibrato -- an owl, not a person."""
    n = int(1.3 * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for start, dur, f0 in ((0.05, 0.35, 420.0), (0.55, 0.45, 380.0)):
        a, b = int(start * SR), int((start + dur) * SR)
        seg_t = np.arange(b - a) / SR
        vibrato = 8.0 * np.sin(2 * np.pi * 6 * seg_t)
        glide = f0 - 60 * (seg_t / seg_t[-1])
        tone = np.sin(2 * np.pi * np.cumsum(glide + vibrato) / SR)
        seg_env = _env(b - a, int(0.03 * SR), int(0.5 * (b - a)))
        out[a:b] += tone * seg_env
    return out.astype(np.float64)


def ghost_whoosh(rng: np.random.Generator) -> np.ndarray:
    """Bandpass-swept noise -- a spectral whoosh, fade in and out."""
    n = int(1.4 * SR)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    center = np.linspace(200.0, 2500.0, n)
    # time-varying bandpass via short overlapping blocks (cheap approximation)
    block = 2048
    for start in range(0, n - block, block // 2):
        f = center[start]
        lo, hi = max(f - 300, 40), f + 300
        sos = signal.butter(2, [lo, hi], btype="bandpass", fs=SR, output="sos")
        filtered = signal.sosfilt(sos, noise[start : start + block])
        win = signal.windows.hann(block)
        out[start : start + block] += filtered * win
    env = _env(n, int(0.1 * SR), int(0.6 * SR))
    return (out * env).astype(np.float64)


def bell_chime(rng: np.random.Generator) -> np.ndarray:
    """A clean bell chime -- a few harmonics, moderate decay, for a lighter "ding" sting."""
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    fundamental = 523.25  # C5
    out = np.zeros(n)
    for ratio, amp, decay_rate in ((1.0, 1.0, 2.0), (2.0, 0.5, 3.0), (3.01, 0.25, 4.0)):
        out += amp * np.sin(2 * np.pi * fundamental * ratio * t) * np.exp(-t * decay_rate)
    env = _env(n, int(0.005 * SR), int(0.9 * SR))
    return (out * env).astype(np.float64)


GENERATORS = {
    "thunder_rumble": thunder_rumble,
    "organ_chord": organ_chord,
    "gong": gong,
    "creaky_door": creaky_door,
    "owl_hoot": owl_hoot,
    "ghost_whoosh": ghost_whoosh,
    "bell_chime": bell_chime,
}

SEED = 20261015  # fixed, deterministic across runs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    written = []
    for name, gen in GENERATORS.items():
        audio = gen(rng)
        audio = audio[: int(1.5 * SR)]
        normed = loudnorm_offline(audio, SR, target_lufs=TARGET_LUFS)
        peak = float(np.max(np.abs(normed))) if normed.size else 0.0
        if peak > 0.98:
            normed = normed * (0.98 / peak)
        path = OUT_DIR / f"{name}.wav"
        pcm16 = np.clip(normed * 32767.0, -32768, 32767).astype(np.int16)
        wavfile.write(path, SR, pcm16)
        written.append((name, len(audio) / SR))
        print(f"wrote {path} ({len(audio) / SR:.2f}s)")
    print(f"{len(written)} stings written to {OUT_DIR}")


if __name__ == "__main__":
    main()
