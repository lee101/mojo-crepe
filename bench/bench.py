"""Correctness-gated benchmark for mojo-crepe.

Every case checks agreement with the NumPy reference in `tests/upstream_ref.py`
before timing anything, so a kernel that drifts shows up as a correctness
failure rather than as a suspiciously good number. The reference timings are
the fastest reasonable NumPy formulations, not Python loops that NumPy would
never use: the per-row local-cents average is written as a masked gather plus
a weighted reduction, and the Viterbi decode as a per-frame broadcast max.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))

import mojo_crepe as mc  # noqa: E402
import upstream_ref as ref  # noqa: E402

MAPPING = ref.cents_mapping()
OFFSETS = np.arange(-4, 5)


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def numpy_local_cents(act: np.ndarray, centers=None) -> np.ndarray:
    """The 9-bin weighted cents average, fully vectorised over rows."""
    if centers is None:
        centers = np.argmax(act, axis=1)
    idx = centers[:, None] + OFFSETS[None, :]
    inside = (idx >= 0) & (idx < act.shape[1])
    clipped = np.clip(idx, 0, act.shape[1] - 1)
    window = np.take_along_axis(act, clipped, axis=1)
    weight = np.where(inside, MAPPING[clipped], 0.0)
    return (window * weight).sum(axis=1) / (window * inside).sum(axis=1)


def numpy_predict_tail(act: np.ndarray):
    confidence = ref.ref_row_max(act)
    path = ref.ref_viterbi_path(np.argmax(act, axis=1))
    cents = numpy_local_cents(act, path)
    frequency = ref.ref_cents_to_hz(cents)
    return np.arange(confidence.shape[0]) * 0.01, frequency, confidence, path


def bench_frame_normalize(seconds: int = 30):
    n = mc.MODEL_SRATE * seconds
    rng = np.random.default_rng(0)
    t = np.arange(n) / mc.MODEL_SRATE
    audio = (
        0.6 * np.sin(2 * np.pi * 220.0 * t)
        + 0.2 * np.sin(2 * np.pi * 1100.0 * t)
        + 0.01 * rng.standard_normal(n)
    )
    hop = mc.MODEL_SRATE // 100
    windows = np.lib.stride_tricks.sliding_window_view(audio, mc.FRAME_LEN)[::hop]

    got = mc.frame_normalize(audio)
    expect = ref.ref_frame_normalize(windows)
    assert got.shape == expect.shape
    assert np.allclose(got, expect, rtol=1e-12, atol=1e-12), "frame mismatch"
    assert np.allclose(got.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(got.std(axis=1), 1.0, rtol=1e-12)

    numpy_time = _time(lambda: ref.ref_frame_normalize(windows), 3)
    mojo_time = _time(lambda: mc.frame_normalize(audio), 3)
    return f"frame-normalize {seconds}s audio", numpy_time, mojo_time


def bench_local_cents(rows: int = 3000):
    rng = np.random.default_rng(1)
    act = rng.random((rows, mc.N_BINS))
    got = mc.to_local_average_cents(act)
    assert np.allclose(
        got, numpy_local_cents(act), rtol=1e-11, atol=1e-9
    ), "local cents mismatch"
    numpy_time = _time(lambda: numpy_local_cents(act), 3)
    mojo_time = _time(lambda: mc.to_local_average_cents(act), 3)
    return f"local-average-cents {rows}x360", numpy_time, mojo_time


def bench_viterbi(frames: int = 1000):
    rng = np.random.default_rng(2)
    act = rng.random((frames, mc.N_BINS))
    obs = np.argmax(act, axis=1).astype(np.int32)
    got = mc.viterbi_path(act)
    assert got.tolist() == ref.ref_viterbi_path(obs).tolist(), "viterbi mismatch"
    numpy_time = _time(lambda: ref.ref_viterbi_path(obs), 1)
    mojo_time = _time(lambda: mc.viterbi_path(act), 3)
    return f"viterbi decode {frames} frames", numpy_time, mojo_time


def bench_cents_to_hz(n: int = 1 << 20):
    rng = np.random.default_rng(3)
    cents = rng.uniform(0.0, 9200.0, size=n)
    assert np.allclose(
        mc.cents_to_hz(cents), ref.ref_cents_to_hz(cents), rtol=1e-11
    ), "hz mismatch"
    numpy_time = _time(lambda: ref.ref_cents_to_hz(cents), 3)
    mojo_time = _time(lambda: mc.cents_to_hz(cents), 3)
    return f"cents-to-hz n={n}", numpy_time, mojo_time


def bench_predict_tail(frames: int = 2000):
    """Everything crepe.predict does once the activation matrix exists."""
    rng = np.random.default_rng(4)
    act = rng.random((frames, mc.N_BINS))
    time_, freq, conf, _ = mc.predict_from_activation(act, viterbi=True)
    expect_time, expect_freq, expect_conf, _ = numpy_predict_tail(act)
    assert np.array_equal(conf, expect_conf)
    assert np.array_equal(time_, expect_time)
    assert np.allclose(freq, expect_freq, rtol=1e-11), "predict tail mismatch"
    numpy_time = _time(lambda: numpy_predict_tail(act), 1)
    mojo_time = _time(lambda: mc.predict_from_activation(act, viterbi=True), 1)
    return f"predict tail {frames} frames", numpy_time, mojo_time


def main():
    print(f"{'case':<30}{'numpy reference':>18}{'mojo-crepe':>16}{'ratio':>10}")
    print("-" * 74)
    for fn in (
        bench_frame_normalize,
        bench_local_cents,
        bench_viterbi,
        bench_cents_to_hz,
        bench_predict_tail,
    ):
        label, baseline, got = fn()
        ratio = baseline / got if got else float("nan")
        print(
            f"{label:<30}{baseline * 1e3:>14.2f}ms{got * 1e3:>12.2f}ms"
            f"{ratio:>9.2f}x"
        )
    print()
    print("ratio > 1 means the Mojo kernel is faster. The reference column is")
    print("NumPy: crepe itself cannot run here at all, because crepe.core")
    print("imports keras, hmmlearn and resampy at module scope.")


if __name__ == "__main__":
    main()
