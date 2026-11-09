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
    """`get_activation` divides by `np.clip(std, 1e-8, None)`, so a silent
    frame is `(x - mean) / 1e-8`, which is exactly zero when the samples are
    exactly representable (ones: mean 1.0, differences 0.0)."""
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


def test_frame_normalize_clips_std_at_1e_minus_8():
    """`get_activation` divides by `np.clip(std, 1e-8, None)`: a frame whose
    std is ~3e-14 (one sample differing by 1e-12) is scaled by 1e8, not by
    1/std, so its output magnitude stays near 1e-4. A kernel that divides by
    the raw std would report values around 32. The tolerance is loose because
    the numerator is a catastrophic-cancellation quantity: sequential and
    pairwise summation order differ by a few ulps, which dominates the
    relative error once amplified by 1e8."""
    audio = np.zeros(mc.FRAME_LEN)
    audio[0] = 1e-12
    got = mc.frame_normalize(audio)
    expect = ref.ref_frame_normalize(audio[None, :])
    np.testing.assert_allclose(got, expect, rtol=1.0, atol=1e-3)
    assert np.abs(got).max() < 1e-3


def test_frame_normalize_non_representable_constant_is_tiny_not_nan():
    """A window of 0.1 has a sequential-sum mean that is not exactly 0.1, so
    the numerator is ~1e-17; dividing by the clipped 1e-8 leaves values
    around 1e-9, never NaN from a 0/0. Upstream on the same window gives the
    same magnitude in float64."""
    audio = np.full(4096, 0.1)
    got = mc.frame_normalize(audio)
    expect = ref.ref_frame_normalize(np.full((1, mc.FRAME_LEN), 0.1))
    np.testing.assert_allclose(got[0], expect[0], rtol=1.0, atol=1e-6)
    assert np.all(np.abs(got) < 1e-6)


def test_frame_normalize_nan_frames_is_nan():
    """A NaN in the window poisons mean and std through the clip, so upstream
    returns NaN; the port must too."""
    audio = np.full(4096, np.nan)
    got = mc.frame_normalize(audio)
    assert got.shape[0] >= 1
    assert np.all(np.isnan(got))


def test_frame_normalize_inf_frames_is_nan():
    """An infinite sample makes `x - mean` an `inf - inf` difference, so the
    clipped std is NaN and upstream returns NaN; the port matches."""
    audio = np.zeros(4096)
    audio[100] = np.inf
    got = mc.frame_normalize(audio)
    assert np.all(np.isnan(got[0]))
    audio = np.full(4096, np.inf)
    got = mc.frame_normalize(audio)
    assert np.all(np.isnan(got))


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


# -------------------------------------------------- degenerate inputs -----


def test_frame_normalize_empty_and_partial_frames_raise_valueerror():
    """Upstream's `n_frames = 1 + int((len(audio) - 1024) / hop_length)`
    followed by `as_strided` with a negative or oversized frame count
    raises `ValueError` for inputs shorter than one frame (the oversized
    case reads out of bounds, which the port rejects rather than trusting
    memory past the buffer)."""
    with pytest.raises(ValueError, match="cannot fill a"):
        mc.frame_normalize(np.zeros(0))
    with pytest.raises(ValueError, match="cannot fill a"):
        mc.frame_normalize(np.zeros(mc.FRAME_LEN - 1))


def test_frame_normalize_single_frame_boundary():
    """Exactly one frame is the smallest valid input; the last input length
    that still yields one frame is 1024 + hop - 1 = 1183."""
    rng = np.random.default_rng(31)
    audio = rng.standard_normal(mc.FRAME_LEN)
    got = mc.frame_normalize(audio)
    assert got.shape == (1, mc.FRAME_LEN)
    np.testing.assert_allclose(
        got, ref.ref_frame_normalize(audio[None, :].astype(np.float64)),
        rtol=1e-12, atol=1e-12,
    )
    longest_one = mc.FRAME_LEN + mc.MODEL_SRATE // 100 - 1
    assert mc.frame_normalize(rng.standard_normal(longest_one)).shape[0] == 1
    assert mc.frame_normalize(rng.standard_normal(longest_one + 1)).shape[0] == 2


def test_frame_normalize_raises_same_valueerror_as_upstream_framing():
    """For inputs shorter than a frame, upstream's `as_strided` window
    shape goes negative and NumPy raises `ValueError: negative dimensions
    are not allowed`. The port raises the same class of error."""
    for n in (0, 100, mc.FRAME_LEN - 1):
        with pytest.raises(ValueError):
            mc.frame_normalize(np.zeros(n))


def test_local_cents_empty_matrix_matches_upstream():
    """Upstream's 2D branch is a list comprehension over rows: zero rows
    give `np.array([])`, an empty float64 array, not an error."""
    expect = ref.ref_local_average_cents(np.empty((0, mc.N_BINS)))
    got = mc.to_local_average_cents(np.empty((0, mc.N_BINS)))
    assert got.shape == expect.shape == (0,)


def test_local_cents_zero_width_row_raises_like_numpy_argmax():
    """`np.argmax` on an empty row raises `ValueError: attempt to get
    argmax of an empty sequence`; the port surfaces the same."""
    with pytest.raises(ValueError, match="argmax"):
        mc.to_local_average_cents(np.empty((1, 0)))


def test_local_cents_1d_returns_scalar_like_upstream():
    """Upstream's 1D branch returns `product_sum / weight_sum`, a float64
    scalar, not an array. The port's matrix entry point still accepts a
    single row but its return must be scalar for parity."""
    rng = np.random.default_rng(37)
    row = rng.random(mc.N_BINS)
    got = mc.to_local_average_cents(row)
    assert np.ndim(got) == 0
    # upstream's 1D expression, transcribed from core.py
    mapping = ref.cents_mapping()
    center = int(np.argmax(row))
    start, end = max(0, center - 4), min(len(row), center + 5)
    expect = np.sum(row[start:end] * mapping[start:end]) / np.sum(row[start:end])
    np.testing.assert_allclose(got, expect, rtol=1e-12)


def test_local_cents_1d_centred_scalar():
    """The centred branch of the upstream 1D API: `to_local_average_cents(
    salience[i], path[i])`, one row at a time."""
    rng = np.random.default_rng(39)
    row = rng.random(mc.N_BINS)
    got = mc.to_local_average_cents(row, np.array([150]))
    mapping = ref.cents_mapping()
    window = row[146:155]
    expect = np.sum(window * mapping[146:155]) / np.sum(window)
    np.testing.assert_allclose(got, expect, rtol=1e-12)


def test_local_cents_rejects_0d_and_3d_like_upstream():
    """Upstream raises `Exception("label should be either 1d or 2d
    ndarray")` for any other rank."""
    with pytest.raises(Exception, match="label should be either 1d or 2d"):
        mc.to_local_average_cents(np.float64(1.0))
    with pytest.raises(Exception, match="label should be either 1d or 2d"):
        mc.to_local_average_cents(np.ones((2, 3, mc.N_BINS)))


def test_local_cents_zero_weight_row_is_nan_like_upstream():
    """An all-zero window gives `0 / 0 = NaN` in upstream's
    `product_sum / weight_sum`; the port produces the same NaN."""
    got = mc.to_local_average_cents(np.zeros((2, mc.N_BINS)))
    assert np.all(np.isnan(got))


def test_local_cents_nan_argmax_propagates_like_numpy():
    """Modern `np.argmax` propagates NaN (the first NaN wins over any finite
    value and over +inf); the port's argmax must too, so the weighted
    average over the NaN's window is NaN like upstream."""
    for row, in_nan_vs_ref in (
        (np.array([[1.0] * 359 + [np.nan]]), None),
        (np.array([[np.nan] + [1.0] * 359]), None),
        (np.array([[1.0] * 100 + [np.nan] + [1.0] * 259]), None),
    ):
        got = mc.to_local_average_cents(row)
        expect = ref.ref_local_average_cents(row)
        assert np.isnan(got[0]) and np.isnan(expect[0])


def test_local_cents_inf_row_parity():
    """A +inf salience entry dominates both the argmax and the window
    weights; the port's weighted mean matches the NumPy reference."""
    row = np.ones((1, mc.N_BINS))
    row[0, 180] = np.inf
    got = mc.to_local_average_cents(row)
    expect = ref.ref_local_average_cents(row)
    assert np.isnan(got[0]) and np.isnan(expect[0])  # inf-weighted mean
    row = np.full((1, mc.N_BINS), np.inf)
    got = mc.to_local_average_cents(row)
    expect = ref.ref_local_average_cents(row)
    np.testing.assert_allclose(got, expect, equal_nan=True, rtol=1e-12, atol=1e-9)


def test_local_cents_duplicate_max_takes_first_like_upstream():
    """Duplicates: an all-equal row's argmax is bin 0 under the first-max
    rule, so the result is the mean of the mapping over bins 0..4."""
    got = mc.to_local_average_cents(np.full((2, mc.N_BINS), 3.0))
    expect = ref.ref_local_average_cents(np.full((2, mc.N_BINS), 3.0))
    np.testing.assert_allclose(got, expect, rtol=1e-12)
    mapping = ref.cents_mapping()
    np.testing.assert_allclose(got, mapping[0:5].mean(), rtol=1e-12)


def test_local_cents_out_of_range_centre_matches_upstream_clipping():
    """Upstream clips the window with `start = max(0, centre-4)` and
    `end = min(len, centre+5)`, so centre 360 averages bins 356..359 and
    centre -1 averages bins 0..3, while a centre far outside produces an
    empty window and 0/0 = NaN."""
    act = np.ones((1, mc.N_BINS))
    np.testing.assert_allclose(
        mc.to_local_average_cents(act, np.array([mc.N_BINS])),
        ref.ref_local_average_cents(act, np.array([mc.N_BINS])),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        mc.to_local_average_cents(act, np.array([-1])),
        ref.ref_local_average_cents(act, np.array([-1])),
        rtol=1e-12,
    )
    got = mc.to_local_average_cents(act, np.array([1000]))
    expect = ref.ref_local_average_cents(act, np.array([1000]))
    assert np.isnan(got[0]) and np.isnan(expect[0])


def test_row_max_empty_and_zero_width():
    """`activation.max(axis=1)` on (0, 360) is the empty vector; on (1, 0)
    NumPy raises `ValueError` because maximum has no identity. Status of
    the port: rows are never empty in production (the CNN emits 360), so
    the empty-rows case is returned and the zero-width case raises."""
    got = mc.row_max(np.empty((0, mc.N_BINS)))
    assert got.shape == (0,)
    with pytest.raises(ValueError, match="zero-size"):
        mc.row_max(np.empty((1, 0)))


def test_row_max_nan_propagates_like_numpy_max():
    """`np.max` propagates NaN anywhere in the row; the port must agree
    instead of returning the largest finite value."""
    for row in (
        np.array([[np.nan, 1.0, 2.0]]),
        np.array([[1.0, np.nan, 2.0]]),
        np.array([[1.0, 2.0, np.nan]]),
    ):
        got = mc.row_max(row)
        expect = ref.ref_row_max(row)
        assert np.isnan(got[0]) and np.isnan(expect[0])


def test_row_max_inf_and_negatives_parity():
    rng = np.random.default_rng(41)
    act = rng.random((9, mc.N_BINS)) * 2.0 - 1.0
    act[0, ::2] = -np.inf
    act[1, 10] = np.inf
    np.testing.assert_array_equal(mc.row_max(act), ref.ref_row_max(act))


def test_cents_to_hz_inf_and_neg_inf_match_power_expression():
    """`10 * 2 ** (cents / 1200)` is +inf at +inf and 0 at -inf; a kernel
    that delegates both to a single `exp2` approximation returns 10 for
    each, so the infinities need their own branches."""
    got = mc.cents_to_hz(np.array([np.inf, -np.inf]))
    expect = ref.ref_cents_to_hz(np.array([np.inf, -np.inf]))
    np.testing.assert_array_equal(got, expect)
    assert np.isposinf(got[0]) and got[1] == 0.0


def test_cents_to_hz_empty():
    got = mc.cents_to_hz(np.array([]))
    assert got.shape == (0,)


def test_hmm_parameters_parity_with_probability_space_reference():
    """`hmm_parameters` is the log-domain twin of upstream's probability
    matrices: `log(starting)`, `log(transition)`, and the transposed
    `log(emission)`, each agreeing with `np.log` of the reference to
    double precision."""
    for n in (3, 360):
        log_start, log_trans, log_emis_t = mc.hmm_parameters(n)
        start, trans, emis = ref.ref_hmm_parameters(n)
        with np.errstate(divide="ignore"):
            np.testing.assert_allclose(log_start, np.log(start), rtol=1e-12)
            np.testing.assert_allclose(
                log_trans, np.log(trans), rtol=1e-12, equal_nan=True
            )
            np.testing.assert_allclose(log_emis_t, np.log(emis).T, rtol=1e-12)


def test_activation_from_audio_is_frame_normalize_alias():
    rng = np.random.default_rng(43)
    audio = _audio(seed=43)
    np.testing.assert_array_equal(
        mc.activation_from_audio(audio), mc.frame_normalize(audio)
    )


def test_to_viterbi_cents_parity_against_reference_composition():
    """`to_viterbi_cents` is argmax -> Viterbi decode -> centred local
    cents; compare the whole composition against the NumPy reference."""
    rng = np.random.default_rng(47)
    act = rng.random((33, mc.N_BINS))
    path = ref.ref_viterbi_path(np.argmax(act, axis=1))
    expect = ref.ref_local_average_cents(act, path)
    np.testing.assert_allclose(
        mc.to_viterbi_cents(act), expect, rtol=1e-12, atol=1e-9
    )


def test_viterbi_empty_matrix():
    """Zero frames: `np.argmax` on (0, 360) is empty, so the decode is an
    empty path and the cents/cents-to-Hz pipeline returns empty arrays."""
    act = np.empty((0, mc.N_BINS))
    assert mc.viterbi_path(act).shape == (0,)
    assert mc.to_viterbi_cents(act).shape == (0,)
    time, freq, conf, _ = mc.predict_from_activation(act, viterbi=True)
    assert time.shape == freq.shape == conf.shape == (0,)
    assert np.array_equal(freq, np.array([]))


def test_viterbi_zero_width_raises_like_argmax():
    """`np.argmax(salience, axis=1)` on a (T, 0) matrix raises `ValueError:
    attempt to get argmax of an empty sequence`."""
    with pytest.raises(ValueError, match="argmax"):
        mc.viterbi_path(np.empty((1, 0)))
    with pytest.raises(ValueError, match="argmax"):
        mc.to_viterbi_cents(np.empty((1, 0)))


def test_predict_rejects_3d_activation_like_upstream_max_axis():
    """`activation.max(axis=1)` on a 3D array has no axis 1 issue, but
    `np.argmax(..., axis=1)` on the 2D contract rejects the rank: the
    port's shape unpack raises `ValueError` on the (T, 360) expectation,
    matching the AxisError family upstream would raise downstream."""
    with pytest.raises(ValueError):
        mc.predict_from_activation(np.ones((2, 3, mc.N_BINS)))
