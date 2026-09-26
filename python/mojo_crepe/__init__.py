"""mojo-crepe: the numeric tail of CREPE pitch estimation on Mojo.

Upstream `crepe` is a Keras CNN; this package ports the framing, the
bin-to-cents mapping, the 360-state Viterbi decode and the cents-to-Hz map.
See README.md for the coverage table.
"""

from .core import (
    FRAME_LEN,
    HALF_WINDOW,
    MODEL_SRATE,
    N_BINS,
    SELF_EMISSION,
    activation_from_audio,
    cents_to_hz,
    frame_normalize,
    hmm_parameters,
    predict_from_activation,
    row_max,
    to_local_average_cents,
    to_viterbi_cents,
    viterbi_path,
)

__version__ = "0.1.0"

__all__ = [
    "FRAME_LEN",
    "HALF_WINDOW",
    "MODEL_SRATE",
    "N_BINS",
    "SELF_EMISSION",
    "activation_from_audio",
    "cents_to_hz",
    "frame_normalize",
    "hmm_parameters",
    "predict_from_activation",
    "row_max",
    "to_local_average_cents",
    "to_viterbi_cents",
    "viterbi_path",
]
