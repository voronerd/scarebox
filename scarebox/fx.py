"""Barker FX chain: numpy + scipy only (ADR 0001 Sec1d: the Spotify audio-FX package is
GPL-3.0 licensed and is not allowed here -- see docs/MODELS.md "Excluded on licence").

``BarkerFX`` is the "carnival barker" voice effect from spec Sec4.5: a bandpass-ish colour,
a soft saturator, a presence boost, a compressor and a short synthetic reverb, followed by a
fixed output gain and a hard limiter. It is stateful across calls to ``process()`` so audio can
be streamed in small chunks (e.g. 20ms) and still match processing the whole clip in one call,
within floating-point tolerance -- every filter stage carries its own ``zi``/envelope state
between calls instead of resetting it.

``loudnorm_offline`` is a separate, non-streaming helper for one-shot library renders: it
measures integrated loudness with a from-scratch ITU-R BS.1770 K-weighting + gating
implementation (no external loudness library), then applies a single gain to hit the target.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# Hard limiter ceiling used by every preset.
_LIMITER_DBFS = -1.0
_LIMITER_LINEAR = 10.0 ** (_LIMITER_DBFS / 20.0)

# Fixed seed for the synthetic reverb impulse response -- deterministic across runs/processes
# so streaming-vs-block tests (and repeated bench runs) are reproducible.
_REVERB_SEED = 20261015
_REVERB_TAIL_S = 0.25
_REVERB_DECAY_S = 0.08
_REVERB_WET = 0.15

_PRESETS = {
    # High-pass/low-pass corners (Hz), saturator drive g, and whether the rest of the chain
    # (presence EQ, compressor, reverb) is engaged.
    "barker": dict(hp=300.0, lp=5000.0, drive=1.5, chain=True),
    "clean": dict(hp=None, lp=None, drive=None, chain=False),
    "megaphone": dict(hp=500.0, lp=3500.0, drive=3.0, chain=True),
    # 2026-10-16 (Martin: "up the energy"): hotter than barker -- more saturation and a wider
    # band so a goading line sounds shouted through a PA rather than read out.
    "barker_hot": dict(hp=250.0, lp=6000.0, drive=2.4, chain=True),
}


def _butter_sos(cutoff: float, sr: int, btype: str, order: int = 4) -> np.ndarray:
    return signal.butter(order, cutoff, btype=btype, fs=sr, output="sos")


def _rbj_peaking_sos(freq: float, sr: int, gain_db: float, q: float) -> np.ndarray:
    """RBJ Audio EQ Cookbook peaking EQ, as a single second-order section."""
    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    b0 = 1.0 + alpha * a_amp
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_amp
    a0 = 1.0 + alpha / a_amp
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_amp

    sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
    return sos


def _make_reverb_ir(sr: int) -> np.ndarray:
    """Synthetic exponentially-decaying noise IR -- fixed seed, so it's the same every call."""
    rng = np.random.default_rng(_REVERB_SEED)
    n = max(int(_REVERB_TAIL_S * sr), 1)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-t / _REVERB_DECAY_S)
    energy = np.sqrt(np.sum(ir**2))
    if energy > 0:
        ir = ir / energy
    return ir.astype(np.float64)


class BarkerFX:
    """Stateful FX chain. ``process()`` can be called once on a whole clip or repeatedly on
    small chunks -- the filter/envelope/reverb state carries over either way."""

    def __init__(self, sr: int, preset: str = "barker"):
        if preset not in _PRESETS:
            raise ValueError(f"unknown preset {preset!r}, choices: {sorted(_PRESETS)}")
        self.sr = int(sr)
        self.preset = preset
        cfg = _PRESETS[preset]
        self._drive_g = cfg["drive"]
        self._chain = cfg["chain"]

        self._hp_sos = _butter_sos(cfg["hp"], self.sr, "highpass") if cfg["hp"] else None
        self._hp_zi = (
            np.zeros((self._hp_sos.shape[0], 2)) if self._hp_sos is not None else None
        )
        self._lp_sos = _butter_sos(cfg["lp"], self.sr, "lowpass") if cfg["lp"] else None
        self._lp_zi = (
            np.zeros((self._lp_sos.shape[0], 2)) if self._lp_sos is not None else None
        )

        if self._chain:
            self._eq_sos = _rbj_peaking_sos(2000.0, self.sr, gain_db=4.0, q=1.0)
            self._eq_zi = np.zeros((self._eq_sos.shape[0], 2))

            # Feed-forward compressor: -18 dBFS threshold, 3:1 ratio, 5ms attack, 80ms release.
            self._comp_thresh = 10.0 ** (-18.0 / 20.0)
            self._comp_ratio = 3.0
            self._comp_att = np.exp(-1.0 / (self.sr * 0.005))
            self._comp_rel = np.exp(-1.0 / (self.sr * 0.080))
            self._comp_env = 0.0

            self._reverb_ir = _make_reverb_ir(self.sr)
            self._reverb_zi = np.zeros(max(len(self._reverb_ir) - 1, 0))
        else:
            self._eq_sos = None
            self._eq_zi = None
            self._reverb_ir = None
            self._reverb_zi = None

        self._output_gain = 1.0

    def _drive(self, x: np.ndarray) -> np.ndarray:
        g = self._drive_g
        return np.tanh(g * x) / np.tanh(g)

    def _compress(self, x: np.ndarray) -> np.ndarray:
        env = self._comp_env
        att, rel = self._comp_att, self._comp_rel
        thresh, ratio = self._comp_thresh, self._comp_ratio
        out = np.empty_like(x, dtype=np.float64)
        for i in range(x.shape[0]):
            level = abs(x[i])
            if level > env:
                env = att * env + (1.0 - att) * level
            else:
                env = rel * env + (1.0 - rel) * level
            if env > thresh and env > 0:
                over_db = 20.0 * np.log10(env / thresh)
                gain_db = over_db * (1.0 / ratio - 1.0)
                gain = 10.0 ** (gain_db / 20.0)
            else:
                gain = 1.0
            out[i] = x[i] * gain
        self._comp_env = env
        return out

    def _reverb(self, x: np.ndarray) -> np.ndarray:
        wet, zf = signal.lfilter(self._reverb_ir, [1.0], x, zi=self._reverb_zi)
        self._reverb_zi = zf
        return (1.0 - _REVERB_WET) * x + _REVERB_WET * wet

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """chunk: float32 mono ndarray. Returns a float32 ndarray of the same length."""
        x = np.asarray(chunk, dtype=np.float64)

        if self._hp_sos is not None:
            x, self._hp_zi = signal.sosfilt(self._hp_sos, x, zi=self._hp_zi)
        if self._lp_sos is not None:
            x, self._lp_zi = signal.sosfilt(self._lp_sos, x, zi=self._lp_zi)

        if self._drive_g is not None:
            x = self._drive(x)

        if self._chain:
            x, self._eq_zi = signal.sosfilt(self._eq_sos, x, zi=self._eq_zi)
            x = self._compress(x)
            x = self._reverb(x)

        x = x * self._output_gain
        x = np.clip(x, -_LIMITER_LINEAR, _LIMITER_LINEAR)
        return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Offline loudness normaliser (ITU-R BS.1770 K-weighting + gating), numpy-only.
# ---------------------------------------------------------------------------


def _k_weighting_sos(sr: int) -> np.ndarray:
    """Two-stage K-weighting filter (pre-filter shelf + RLB high-pass), designed from the
    analog prototype parameters so it generalises to any sample rate, not just 48kHz."""
    # Stage 1: high-frequency shelf boost.
    f0, g_db, q = 1681.9744509555319, 3.99984385397, 0.7071752369554193
    shelf_sos = _rbj_peaking_sos(f0, sr, g_db, q)  # peaking approximation of the shelf is close
    # enough for a self-consistent loudness measurement (not claimed bit-exact to the ITU tables).

    # Stage 2: RLB weighting curve, approximated as a 2nd-order high-pass near its corner.
    rlb_sos = _butter_sos(38.0, sr, "highpass", order=2)

    return np.vstack([shelf_sos, rlb_sos])


def measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Integrated loudness (LUFS) via K-weighting + BS.1770 gating. No external library."""
    x = np.asarray(audio, dtype=np.float64)
    weighted = signal.sosfilt(_k_weighting_sos(sr), x)

    block_s, hop_s = 0.400, 0.100  # 400ms blocks, 75% overlap
    block_n, hop_n = int(block_s * sr), int(hop_s * sr)
    if len(weighted) < block_n:
        ms = float(np.mean(weighted**2)) + 1e-12
        return -0.691 + 10.0 * np.log10(ms)

    starts = range(0, len(weighted) - block_n + 1, hop_n)
    block_ms = np.array([np.mean(weighted[s : s + block_n] ** 2) for s in starts])
    block_loudness = -0.691 + 10.0 * np.log10(block_ms + 1e-12)

    # Absolute gate at -70 LUFS.
    gated = block_ms[block_loudness > -70.0]
    if len(gated) == 0:
        return -70.0
    ungated_loudness = -0.691 + 10.0 * np.log10(np.mean(gated) + 1e-12)

    # Relative gate at (ungated loudness - 10 LU).
    rel_threshold = ungated_loudness - 10.0
    final = gated[(-0.691 + 10.0 * np.log10(gated + 1e-12)) > rel_threshold]
    if len(final) == 0:
        final = gated
    return float(-0.691 + 10.0 * np.log10(np.mean(final) + 1e-12))


def loudnorm_offline(audio: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    """One-shot (non-streaming) loudness normalisation for library renders."""
    x = np.asarray(audio, dtype=np.float64)
    measured = measure_lufs(x, sr)
    gain_db = target_lufs - measured
    out = x * (10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > _LIMITER_LINEAR:
        out = out * (_LIMITER_LINEAR / peak)
    return out.astype(np.float32)
