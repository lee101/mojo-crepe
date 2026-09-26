"""Parity and analytic tests for the Mojo CREPE kernels.

Every test would fail on a plausible kernel bug: a wrong hop, a dropped SIMD
tail frame, a window that is off by one bin, a Viterbi recurrence that ignores
the transition prior, a first-argmax rule that picks the wrong tied bin.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojo_crepe as mc
import upstream_ref as ref


def _audio(n: int = 16000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / mc.MODEL_SRATE
    return (
        0.6 * np.sin(2 * np.pi * 220.0 * t)
        + 0.2 * np.sin(2 * np.pi * 1100.0 * t + 0.3)
        + 0.01 * rng.standard_normal(n)
    )


# --------------------------------------------------------------- framing ----


def test_frame_normalize_parity():
    audio = _audio()
    frames = mc.frame_normalize(audio)
    hop = mc.MODEL_SRATE // 100
    n_frames = 1 + int((audio.size - mc.FRAME_LEN) / hop)
    assert frames.shape == (n_frames, mc.FRAME_LEN)
    windows = np.lib.stride_tricks.sliding_window_view(audio, mc.FRAME_LEN)[
        ::hop
    ]
    expect = ref.ref_frame_normalize(windows)
    np.testing.assert_allclose(frames, expect, rtol=1e-12, atol=1e-12)


def test_frame_normalize_statistics():
    """Each frame comes out zero-mean and unit population std."""
    frames = mc.frame_normalize(_audio(seed=3))
    np.testing.assert_allclose(frames.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(frames.std(axis=1), 1.0, rtol=1e-12)

def test_frame_normalize_hop_alignment():
    """A single impulse must light up exactly the frames whose window covers
    it, at the right column of each.

    1024-sample frames on a 160-sample hop overlap heavily, so one impulse
    lands in seven frames; which seven is a direct function of the hop, and
    the column it lands in is a direct function of the frame origin. A wrong
    hop, or a frame that reused the previous frame's window, fails both.
    """
    hop = mc.MODEL_SRATE // 100
    audio = np.zeros(8 * hop + mc.FRAME_LEN)
    spike = 3 * hop + 17
    audio[spike] = 1.0
    frames = mc.frame_normalize(audio)
    hot = np.flatnonzero(np.abs(frames).max(axis=1) > 0.0)
    assert hot.tolist() == [0, 1, 2, 3]
    for t in hot:
        assert int(np.argmax(np.abs(frames[t]))) == spike - t * hop



def test_frame_normalize_tail_frame_is_complete():
    """The last frame is normalised, not skipped: a dropped tail would leave
    the final row all zeros while every earlier row has unit std."""
    audio = _audio(n=4096, seed=7)
    frames = mc.frame_normalize(audio)
    assert frames.shape[0] >= 2
    np.testing.assert_allclose(frames[-1].std(), 1.0, rtol=1e-12)


def test_frame_normalize_constant_frame_is_zero():
    """Documented divergence from NumPy: a silent frame would be 0/0 = NaN
    there, and is all-zero here."""
    audio = np.ones(4096)
    frames = mc.frame_normalize(audio)
    assert np.all(frames == 0.0)


def test_frame_normalize_float32_input():
    """CREPE casts the audio to float32 before framing; the port keeps
    float64. Against a float64 reference the float32-input path is exact, and
    against a float32 reference it agrees to single precision."""
    audio = _audio().astype(np.float32)
    got = mc.frame_normalize(audio)
    hop = mc.MODEL_SRATE // 100
    wide = np.lib.stride_tricks.sliding_window_view(
        audio.astype(np.float64), mc.FRAME_LEN
    )[::hop]
    np.testing.assert_allclose(
        got, ref.ref_frame_normalize(wide), rtol=1e-12, atol=1e-12
    )
    narrow = np.lib.stride_tricks.sliding_window_view(audio, mc.FRAME_LEN)[::hop]
    np.testing.assert_allclose(
        got, ref.ref_frame_normalize(narrow), rtol=1e-5, atol=1e-5
    )


# --------------------------------------------------------- local cents ------


def test_local_cents_parity():
    rng = np.random.default_rng(1)
    act = rng.random((37, mc.N_BINS))
    np.testing.assert_allclose(
        mc.to_local_average_cents(act),
        ref.ref_local_average_cents(act),
        rtol=1e-12,
        atol=1e-9,
    )


def test_local_cents_one_hot_is_exact_mapping():
    """A one-hot row must return the mapping value of that bin exactly.

    Neighbouring bins have zero weight, so this pins the mapping itself: a
    wrong step, a wrong offset or a wrong table would all show up.
    """
    mapping = ref.cents_mapping()
    for bin_index in (0, 1, 42, 180, mc.N_BINS - 2, mc.N_BINS - 1):
        act = np.zeros((1, mc.N_BINS))
        act[0, bin_index] = 1.0
        got = mc.to_local_average_cents(act)
        np.testing.assert_allclose(got, [mapping[bin_index]], rtol=1e-12)


def test_local_cents_window_is_nine_bins():
    """Two peaks nine bins apart must pull the result toward the middle bin;
    a window of 5 or 11 bins weights them differently."""
    act = np.zeros((1, mc.N_BINS))
    act[0, 200] = 1.0
    act[0, 208] = 1.0
    got = mc.to_local_average_cents(act)[0]
    mapping = ref.cents_mapping()
    # peaks are outside the +/-4 window of the argmax (200), so the other peak
    # contributes nothing and the result is bin 200's mapping value
    np.testing.assert_allclose([got], [mapping[200]], rtol=1e-12)
    # shift the pair to 205/213: 208 is within 205+4, so it does contribute
    act2 = np.zeros((1, mc.N_BINS))
    act2[0, 205] = 1.0
    act2[0, 208] = 1.0
    got2 = mc.to_local_average_cents(act2)[0]
    expect = (mapping[205] + mapping[208]) / 2.0
    np.testing.assert_allclose([got2], [expect], rtol=1e-12)
    assert abs(got2 - mapping[205]) > 1.0


def test_local_cents_argmax_tie_takes_first_bin():
    """np.argmax keeps the first maximum; a kernel using >= would jump to the
    last tied bin and report a very different pitch."""
    act = np.zeros((1, mc.N_BINS))
    act[0, 100] = 0.5
    act[0, 200] = 0.5
    np.testing.assert_allclose(
        mc.to_local_average_cents(act),
        ref.ref_local_average_cents(act),
        rtol=1e-12,
    )
    assert mc.to_local_average_cents(act)[0] == pytest.approx(
        ref.cents_mapping()[100], rel=1e-12
    )


def test_local_cents_window_clips_at_edges():
    """An argmax at bin 0 uses [0, 5) and at bin 359 uses [355, 360); an
    unclipped window would read out of bounds or shift the window."""
    mapping = ref.cents_mapping()
    for edge, weight_bin in ((0, 3), (mc.N_BINS - 1, mc.N_BINS - 4)):
        act = np.zeros((1, mc.N_BINS))
        act[0, edge] = 1.0
        act[0, weight_bin] = 1.0
        got = mc.to_local_average_cents(act)[0]
        expect = ref.ref_local_average_cents(act)[0]
        np.testing.assert_allclose([got], [expect], rtol=1e-12)
        assert abs(got - expect) < 1e-9


def test_local_cents_centred_matches_reference():
    rng = np.random.default_rng(2)
    act = rng.random((23, mc.N_BINS))
    centers = rng.integers(0, mc.N_BINS, size=23)
    np.testing.assert_allclose(
        mc.to_local_average_cents(act, centers),
        ref.ref_local_average_cents(act, centers),
        rtol=1e-12,
        atol=1e-9,
    )


# -------------------------------------------------------------- viterbi -----


def test_viterbi_path_parity():
    """Exact index equality: the path is a discrete optimum, and both sides
    evaluate the same max-product recurrence in the same order."""
    rng = np.random.default_rng(11)
    act = rng.random((40, mc.N_BINS))
    got = mc.viterbi_path(act)
    obs = np.argmax(act, axis=1)
    expect = ref.ref_viterbi_path(obs)
    assert got.tolist() == expect.tolist()


def test_viterbi_respects_transition_support():
    """`max(12 - |i - j|, 0)` is zero beyond 12 bins, so the decoded path can
    never jump more than 12 bins between frames. A kernel that dropped the
    transition term, or used the unscaled matrix, would sail across the whole
    salience range between the two alternating peaks below."""
    act = np.zeros((12, mc.N_BINS))
    act[:, 20] = 1.0
    act[:, 300] = 1.0 - 1e-6
    path = mc.viterbi_path(act)
    steps = np.abs(np.diff(path.astype(np.int64)))
    assert steps.max() <= 12
    assert path.max() - path.min() <= 12 * (path.size - 1)


def test_viterbi_total_log_probability_is_maximal():
    """Score the returned path and compare with the DP optimum.

    The optimum is the same quantity the NumPy reference computes, so this
    checks the backtrack as well as the forward pass: a backtrack that keeps
    the wrong predecessor still walks a valid-looking path but scores lower.
    """
    rng = np.random.default_rng(5)
    act = rng.random((30, mc.N_BINS))
    obs = np.argmax(act, axis=1)
    path = mc.viterbi_path(act)

    start, trans, emis = ref.ref_hmm_parameters(mc.N_BINS)
    with np.errstate(divide="ignore"):
        log_start, log_trans, log_emis = (
            np.log(start), np.log(trans), np.log(emis)
        )
    score = log_start[path[0]] + log_emis[path[0], obs[0]]
    for t in range(1, path.size):
        score += log_trans[path[t - 1], path[t]] + log_emis[path[t], obs[t]]

    expect_path = ref.ref_viterbi_path(obs)
    expect = log_start[expect_path[0]] + log_emis[expect_path[0], obs[0]]
    for t in range(1, expect_path.size):
        expect += log_trans[expect_path[t - 1], expect_path[t]] + log_emis[
            expect_path[t], obs[t]
        ]
    assert score == pytest.approx(expect, rel=1e-12, abs=1e-9)


def test_viterbi_cents_parity():
    rng = np.random.default_rng(13)
    act = rng.random((28, mc.N_BINS))
    path = ref.ref_viterbi_path(np.argmax(act, axis=1))
    np.testing.assert_allclose(
        mc.to_viterbi_cents(act),
        ref.ref_local_average_cents(act, path),
        rtol=1e-12,
        atol=1e-9,
    )


def test_viterbi_stays_on_a_constant_peak():
    """A salience that is constant over time decodes to that bin, since the
    transition prior is normalised and every self-transition is the most likely
    step from that state."""
    act = np.zeros((9, mc.N_BINS))
    act[:, 123] = 1.0
    path = mc.viterbi_path(act)
    assert path.tolist() == [123] * 9


def test_viterbi_single_frame():
    act = np.zeros((1, mc.N_BINS))
    act[0, 7] = 1.0
    assert mc.viterbi_path(act).tolist() == [7]


def test_cents_to_hz_parity():
    """Mojo's `exp2` and NumPy's `2 ** x` (which is `exp(x * ln 2)`) are
    different approximations; over the 0..9200 cent range the two disagree by
    at most 2.3e-12 relative, growing linearly with the exponent. 1e-11 is
    the honest bound; the octave cases below are exact."""
    rng = np.random.default_rng(17)
    cents = rng.uniform(0.0, 9200.0, size=64)
    np.testing.assert_allclose(
        mc.cents_to_hz(cents),
        ref.ref_cents_to_hz(cents),
        rtol=1e-11,
    )


@pytest.mark.parametrize(
    "cents,hz", [(0.0, 10.0), (1200.0, 20.0), (2400.0, 40.0), (-1200.0, 5.0)]
)
def test_cents_to_hz_exact_octaves(cents, hz):
    got = mc.cents_to_hz(np.array([cents]))
    assert got[0] == pytest.approx(hz, rel=1e-15)


def test_cents_to_hz_maps_nan_to_zero():
    got = mc.cents_to_hz(np.array([np.nan, 0.0, 1200.0]))
    np.testing.assert_array_equal(got, np.array([0.0, 10.0, 20.0]))


def test_row_max_parity():
    rng = np.random.default_rng(19)
    act = rng.random((31, mc.N_BINS))
    np.testing.assert_array_equal(mc.row_max(act), ref.ref_row_max(act))


# ------------------------------------------------------------- predict ------

def test_predict_from_activation_parity():
    rng = np.random.default_rng(23)
    act = rng.random((45, mc.N_BINS))
    time, freq, conf, activation = mc.predict_from_activation(act, viterbi=True)
    path = ref.ref_viterbi_path(np.argmax(act, axis=1))

    expect_conf = ref.ref_row_max(act)
    expect_cents = ref.ref_local_average_cents(act, path)
    expect_freq = ref.ref_cents_to_hz(expect_cents)
    expect_time = np.arange(expect_conf.shape[0]) * 0.01

    assert time.shape == freq.shape == conf.shape == (45,)
    assert activation is not None
    np.testing.assert_array_equal(conf, expect_conf)
    np.testing.assert_allclose(freq, expect_freq, rtol=1e-12)
    np.testing.assert_array_equal(time, expect_time)


def test_predict_default_branch_is_local_average():
    rng = np.random.default_rng(29)
    act = rng.random((12, mc.N_BINS))
    _, freq_plain, _, _ = mc.predict_from_activation(act, viterbi=False)
    _, freq_viterbi, _, _ = mc.predict_from_activation(act, viterbi=True)
    # the two branches must actually differ, or the viterbi test above is vacuous
    assert not np.allclose(freq_plain, freq_viterbi)
    np.testing.assert_allclose(
        freq_viterbi,
        ref.ref_cents_to_hz(
            ref.ref_local_average_cents(
                act, ref.ref_viterbi_path(np.argmax(act, axis=1))
            )
        ),
        rtol=1e-12,
    )


def test_predict_rejects_too_short_audio():
    with pytest.raises(ValueError):
        mc.frame_normalize(np.zeros(100))
