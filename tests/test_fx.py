import numpy as np
import pytest

from scarebox.fx import BarkerFX, loudnorm_offline, measure_lufs

SR = 24000


def _noise(seconds: float, seed: int = 7, amp: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * SR))).astype(np.float32)


@pytest.mark.parametrize("preset", ["barker", "clean", "megaphone"])
def test_streaming_matches_block(preset):
    x = _noise(2.0)

    block = BarkerFX(SR, preset)
    block_out = block.process(x)

    chunked = BarkerFX(SR, preset)
    chunk_n = int(0.020 * SR)  # 20ms chunks
    pieces = []
    for i in range(0, len(x), chunk_n):
        pieces.append(chunked.process(x[i : i + chunk_n]))
    chunk_out = np.concatenate(pieces)

    assert chunk_out.shape == block_out.shape
    assert np.allclose(block_out, chunk_out, atol=1e-4)


@pytest.mark.parametrize("preset", ["barker", "clean", "megaphone"])
def test_output_never_exceeds_limiter(preset):
    # A loud burst well above every stage's headroom.
    x = _noise(1.0, amp=3.0)
    fx = BarkerFX(SR, preset)
    out = fx.process(x)
    limiter_linear = 10.0 ** (-1.0 / 20.0)
    assert np.max(np.abs(out)) <= limiter_linear + 1e-6


def test_loudnorm_hits_target():
    t = np.arange(int(3.0 * SR)) / SR
    tone = (0.1 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

    before = measure_lufs(tone, SR)
    normed = loudnorm_offline(tone, SR, target_lufs=-14.0)
    after = measure_lufs(normed, SR)

    assert before < -14.0  # sanity: the quiet tone needed boosting, not cutting
    assert abs(after - (-14.0)) <= 0.5
    assert np.max(np.abs(normed)) <= 10.0 ** (-1.0 / 20.0) + 1e-6
