"""NumPy transcription of the CREPE expressions this port mirrors.

`crepe` itself cannot be imported in the test venv: `crepe.core` imports keras,
hmmlearn, resampy, imageio and matplotlib at module scope, and the trained
`model.h5` is not redistributable. The functions below are line-for-line
transcriptions of crepe 0.0.0 `core.py` (sdist `crepe-0.0.0`, functions named in
each docstring) written only with NumPy, which is the same dependency crepe's
own code uses for these expressions. They are the parity reference; the
analytic tests in `test_dsp.py` are the second opinion.
"""

from __future__ import annotations

import numpy as np

N_BINS = 360


def cents_mapping() -> np.ndarray:
    """crepe.core.to_local_average_cents: the cached bin number-to-cents map."""
    return np.linspace(0, 7180, N_BINS) + 1997.3794084376191


def ref_frame_normalize(frames: np.ndarray) -> np.ndarray:
    """crepe.core.get_activation: `frames -= mean(axis=1)[:, None]` then
    `frames /= std(axis=1)[:, None]` on the strided framing."""
    frames = np.array(frames, dtype=np.float64, copy=True)
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]
    return frames


def ref_local_average_cents(salience: np.ndarray, center=None) -> np.ndarray:
    """crepe.core.to_local_average_cents, row by row."""
    salience = np.asarray(salience, dtype=np.float64)
    mapping = cents_mapping()
    out = np.empty(salience.shape[0], dtype=np.float64)
    for i in range(salience.shape[0]):
        if center is None:
            c = int(np.argmax(salience[i, :]))
        else:
            c = int(np.atleast_1d(center)[i])
        start = max(0, c - 4)
        end = min(len(salience[i, :]), c + 5)
        window = salience[i, start:end]
        product_sum = np.sum(window * mapping[start:end])
        weight_sum = np.sum(window)
        out[i] = product_sum / weight_sum
    return out


def ref_hmm_parameters(n_bins: int = N_BINS):
    """crepe.core.to_viterbi_cents: the transition, starting and emission
    matrices, in probability space."""
    starting = np.ones(n_bins) / n_bins
    xx, yy = np.meshgrid(range(n_bins), range(n_bins))
    transition = np.maximum(12 - np.abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]
    self_emission = 0.1
    emission = (
        np.eye(n_bins) * self_emission
        + np.ones(shape=(n_bins, n_bins)) * ((1 - self_emission) / n_bins)
    )
    return starting, transition, emission


def ref_viterbi_path(observations: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Log-space max-product Viterbi over the crepe parameters.

    Upstream delegates the decode to `hmmlearn.MultinomialHMM.predict`; hmmlearn
    is not installed here, so the standard log-domain formulation of that
    algorithm is written out in NumPy. Ties keep the lower predecessor.
    """
    starting, transition, emission = ref_hmm_parameters(n_bins)
    with np.errstate(divide="ignore"):
        log_start = np.log(starting)
        log_trans = np.log(transition)
        log_emis = np.log(emission)
    n_obs = observations.shape[0]
    delta = log_start.copy()
    psi = np.zeros((n_obs, n_bins), dtype=np.int64)
    for t in range(n_obs):
        o = int(observations[t])
        scores = delta[:, None] + log_trans
        psi[t, :] = np.argmax(scores, axis=0)
        delta = scores[psi[t, :], np.arange(n_bins)] + log_emis[:, o]
    path = np.zeros(n_obs, dtype=np.int64)
    state = int(np.argmax(delta))
    for t in range(n_obs - 1, -1, -1):
        path[t] = state
        state = psi[t, state]
    return path


def ref_cents_to_hz(cents: np.ndarray) -> np.ndarray:
    """crepe.core.predict: `10 * 2 ** (cents / 1200)`, NaN to 0."""
    cents = np.asarray(cents, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0
    return frequency


def ref_row_max(salience: np.ndarray) -> np.ndarray:
    """crepe.core.predict: `activation.max(axis=1)`."""
    return np.asarray(salience, dtype=np.float64).max(axis=1)
