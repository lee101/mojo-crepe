"""The numeric half of `crepe.core`, on Mojo.

Upstream CREPE is a Keras CNN: `get_activation` pushes 1024-sample frames
through `model.predict`, and everything before it is NumPy framing code while
everything after it (`to_local_average_cents`, `to_viterbi_cents`, the cents to
Hz map) is the numeric tail of `predict`. This module ports that tail plus the
framing, and exposes `predict_from_activation` so a caller that has an
activation matrix from any source gets crepe's pitch curve out of Mojo.

The function names and formulas follow crepe 0.0.0 `core.py` exactly; the
differences are dtype and the constant-frame edge case, both documented in the
README.
"""

from __future__ import annotations

import numpy as np

from . import _lib

N_BINS = 360
MODEL_SRATE = 16000
FRAME_LEN = 1024
HALF_WINDOW = 4
SELF_EMISSION = 0.1

__all__ = [
    "N_BINS",
    "MODEL_SRATE",
    "FRAME_LEN",
    "HALF_WINDOW",
    "frame_normalize",
    "activation_from_audio",
    "to_local_average_cents",
    "hmm_parameters",
    "to_viterbi_cents",
    "cents_to_hz",
    "row_max",
    "predict_from_activation",
]


def frame_normalize(audio, frame_len: int = FRAME_LEN,
                    hop: int | None = None) -> np.ndarray:
    """Normalise a strided framing of `audio`, the framing `get_activation` uses.

    Returns a fresh (n_frames, frame_len) float64 array. Upstream builds the
    same windows with `as_strided` and normalises them in place in float32;
    overlapping windows cannot be normalised in place here, so the copy is
    materialised and the arithmetic is done in float64.
    """
    if hop is None:
        hop = int(MODEL_SRATE / 100)
    audio = np.asarray(audio, dtype=np.float64).ravel()
    n_frames = 1 + int((audio.size - frame_len) / hop)
    if n_frames < 1:
        raise ValueError(
            f"{audio.size} samples cannot fill a {frame_len}-sample frame"
        )
    return _lib.frame_normalize(audio, frame_len, hop, n_frames)


def activation_from_audio(audio, frame_len: int = FRAME_LEN,
                          hop: int | None = None) -> np.ndarray:
    """The normalised frames, in the (T, frame_len) layout the CNN expects."""
    return frame_normalize(audio, frame_len, hop)


def to_local_average_cents(salience, center=None) -> np.ndarray:
    """Bin-to-cents map, `crepe.core.to_local_average_cents` for a whole matrix.

    `center` is either None (use each row's first argmax) or a per-row sequence
    of centre bins, which is what `to_viterbi_cents` passes.
    """
    salience = np.ascontiguousarray(salience, dtype=np.float64)
    if salience.ndim == 1:
        salience = salience.reshape(1, -1)
    if center is None:
        return _lib.local_cents(salience, HALF_WINDOW)
    return _lib.local_cents_centred(salience, center, HALF_WINDOW)


def hmm_parameters(n_bins: int = N_BINS) -> tuple[np.ndarray, ...]:
    """The multinomial-HMM parameters of `to_viterbi_cents`, in log space.

    A uniform starting distribution, a pitch-continuity transition prior
    `max(12 - |i - j|, 0)` normalised per row, and an emission that gives the
    observed bin 0.1 and spreads the remaining 0.9 over all 360 bins. Returns
    `(log_start, log_trans, log_emis_t)`; the emission is transposed so the
    kernel reads it row-major.
    """
    bins = np.arange(n_bins)
    transition = np.maximum(12 - np.abs(bins[:, None] - bins[None, :]), 0)
    transition = transition / transition.sum(axis=1)[:, None]
    starting = np.ones(n_bins) / n_bins
    emission = (
        np.eye(n_bins) * SELF_EMISSION
        + np.ones((n_bins, n_bins)) * ((1 - SELF_EMISSION) / n_bins)
    )
    return (
        np.log(starting),
        np.ascontiguousarray(np.log(transition)),
        np.ascontiguousarray(np.log(emission).T),
    )


def to_viterbi_cents(salience) -> np.ndarray:
    """`crepe.core.to_viterbi_cents`: argmax observations, Viterbi, then the
    same local cents average taken around the decoded bin."""
    salience = np.ascontiguousarray(salience, dtype=np.float64)
    observations = np.argmax(salience, axis=1).astype(np.int32)
    log_start, log_trans, log_emis_t = hmm_parameters(salience.shape[1])
    path = _lib.viterbi(observations, log_trans, log_emis_t, log_start)
    return to_local_average_cents(salience, path)


def viterbi_path(salience) -> np.ndarray:
    """The decoded bin path itself, which upstream never returns."""
    salience = np.ascontiguousarray(salience, dtype=np.float64)
    observations = np.argmax(salience, axis=1).astype(np.int32)
    log_start, log_trans, log_emis_t = hmm_parameters(salience.shape[1])
    return _lib.viterbi(observations, log_trans, log_emis_t, log_start)


def cents_to_hz(cents) -> np.ndarray:
    """`10 * 2 ** (cents / 1200)`, with NaN mapped to 0."""
    return _lib.cents_to_hz(cents)


def row_max(salience) -> np.ndarray:
    """`activation.max(axis=1)`, the confidence curve."""
    salience = np.ascontiguousarray(salience, dtype=np.float64)
    return _lib.row_max(salience)


def predict_from_activation(activation, viterbi: bool = False) -> tuple:
    """Everything `crepe.core.predict` does after `get_activation`.

    Takes the (T, 360) activation matrix the CNN produces and returns the same
    4-tuple `predict` returns: `(time, frequency, confidence, activation)`.
    """
    activation = np.ascontiguousarray(activation, dtype=np.float64)
    confidence = row_max(activation)
    if viterbi:
        cents = to_viterbi_cents(activation)
    else:
        cents = to_local_average_cents(activation)
    frequency = cents_to_hz(cents)
    time = np.arange(confidence.shape[0], dtype=np.float64) * 0.01
    return time, frequency, confidence, activation
