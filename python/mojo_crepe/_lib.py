"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-crepe.so"


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    i = ctypes.c_int64
    f = ctypes.c_double

    lib.crepe_frame_normalize.restype = None
    lib.crepe_frame_normalize.argtypes = [i, i, i, i, i, i]

    lib.crepe_local_cents.restype = None
    lib.crepe_local_cents.argtypes = [i, i, i, i, i]

    lib.crepe_local_cents_centred.restype = None
    lib.crepe_local_cents_centred.argtypes = [i, i, i, i, i, i]

    lib.crepe_viterbi.restype = None
    lib.crepe_viterbi.argtypes = [i, i, i, i, i, i, i, i, i, i, i]

    lib.crepe_cents_to_hz.restype = None
    lib.crepe_cents_to_hz.argtypes = [i, i, i]

    lib.crepe_row_max.restype = None
    lib.crepe_row_max.argtypes = [i, i, i, i]
    return lib


lib = _load()


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def i32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32)


def frame_normalize(audio, frame_len: int, hop: int, n_frames: int) -> np.ndarray:
    """Per-frame `(x - mean) / std` of a strided framing; returns (T, frame_len)."""
    x = f64(audio).ravel()
    out = np.empty(n_frames * frame_len, dtype=np.float64)
    lib.crepe_frame_normalize(
        addr(x), x.size, frame_len, hop, n_frames, addr(out)
    )
    return out.reshape(n_frames, frame_len)


def local_cents(activation, half: int = 4) -> np.ndarray:
    act = f64(activation)
    rows, bins = act.shape
    out = np.empty(rows, dtype=np.float64)
    lib.crepe_local_cents(addr(act), rows, bins, half, addr(out))
    return out


def local_cents_centred(activation, centers, half: int = 4) -> np.ndarray:
    act = f64(activation)
    centers = i32(centers)
    rows, bins = act.shape
    out = np.empty(rows, dtype=np.float64)
    lib.crepe_local_cents_centred(
        addr(act), addr(centers), rows, bins, half, addr(out)
    )
    return out


def viterbi(obs, log_trans, log_emis_t, log_start):
    """Decode `obs` and backtrack, returning the state path as int32.

    `log_trans` is (S, S) row-major `log P(j|i)`; `log_emis_t` is (S, S) with
    row `o` holding `log P(o|j)` for all `j`; `log_start` is (S,).
    """
    obs = i32(obs)
    n_states = log_trans.shape[0]
    n_frames = obs.size
    delta = np.empty(n_states, dtype=np.float64)
    nxt = np.empty(n_states, dtype=np.float64)
    acc = np.empty(n_states, dtype=np.float64)
    arg = np.empty(n_states, dtype=np.float64)
    psi = np.empty(n_frames * n_states, dtype=np.int32)
    lib.crepe_viterbi(
        n_frames, n_states, addr(obs), addr(log_trans), addr(log_emis_t),
        addr(log_start), addr(delta), addr(nxt), addr(acc), addr(arg), addr(psi),
    )
    path = np.empty(n_frames, dtype=np.int32)
    if n_frames:
        state = int(np.argmax(delta))
        for t in range(n_frames - 1, -1, -1):
            path[t] = state
            state = int(psi[t * n_states + state])
    return path


def cents_to_hz(cents) -> np.ndarray:
    c = f64(cents).ravel()
    out = np.empty(c.size, dtype=np.float64)
    lib.crepe_cents_to_hz(addr(c), c.size, addr(out))
    return out


def row_max(activation) -> np.ndarray:
    act = f64(activation)
    rows, bins = act.shape
    out = np.empty(rows, dtype=np.float64)
    lib.crepe_row_max(addr(act), rows, bins, addr(out))
    return out
