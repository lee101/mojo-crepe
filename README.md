# mojo-crepe

The numeric half of [CREPE](https://github.com/marl/autocrepe) pitch estimation,
ported to Mojo.

CREPE is a Keras CNN: the salience matrix that drives everything downstream is
`model.predict(frames)`, and the CNN needs `keras` plus a trained `model.h5`.
What surrounds it — the framing, the bin-to-cents mapping, the 360-state Viterbi
smoothing, the cents-to-Hz map — is NumPy, and all of it has a compiled inner
loop. That is what this package ports, behind the same names and the same
numbers, so an activation matrix from any source (the real CNN, a cached
`.npy`, synthetic data) gets crepe's pitch curve out of Mojo.

## Coverage

| Upstream (`crepe.core`) | Ported | Mojo symbol |
| --- | --- | --- |
| `get_activation` framing + `frames -= mean; frames /= std` | yes | `crepe_frame_normalize` |
| `to_local_average_cents` (argmax centre, 9-bin weighted mean) | yes | `crepe_local_cents` |
| `to_local_average_cents(salience[i], path[i])` (Viterbi centre) | yes | `crepe_local_cents_centred` |
| `to_viterbi_cents` HMM parameters + decode | yes | `crepe_viterbi` |
| `predict`: `10 * 2 ** (cents / 1200)`, NaN to 0 | yes | `crepe_cents_to_hz` |
| `predict`: `activation.max(axis=1)` (confidence) | yes | `crepe_row_max` |
| `predict` plumbing (time axis, tuple assembly) | yes, in Python | `predict_from_activation` |
| `build_and_load_model`, the CNN itself | no | — |
| `resample` to 16 kHz (resampy) | no | — |
| `process_file`, `cli`, `.f0.csv` / `.npy` / `.png` output | no | — |
| `hmmlearn.MultinomialHMM.predict` | replaced | own log-domain Viterbi |

## Not implemented, and why

- **The CNN.** `build_and_load_model` reconstructs a six-layer Conv2D stack in
  Keras and loads `model.h5`, which is a 79 MB binary artefact that upstream
  does not redistribute through the wheel. There is nothing to port and nothing
  to test. Everything downstream of it is.
- **Resampling.** `get_activation` calls `resampy.resample` when the input is
  not 16 kHz. `resampy` is not installed here and its polyphase Kaiser
  resampler is a large body of code unrelated to the parts of crepe this port
  covers. Feed the kernel 16 kHz audio, or resample upstream.
- **File IO, the CLI, plotting.** `process_file`, `crepe.cli` and the
  matplotlib salience plot are IO and formatting, not compute.
- **Float32.** Upstream casts the audio to float32 before framing and the
  salience is a float32 matrix. This port is float64 throughout, which is the
  more accurate of the two and one dtype conversion away from upstream. The
  tests pin both behaviours.
- **Constant frames.** A 1024-sample window of silence has zero variance, so
  upstream produces `0/0 = NaN` and a warning. This port emits an all-zero
  frame. This is the one place the output differs, and it only differs where
  upstream's output is undefined.
- **Tie-breaking in the Viterbi decode.** Upstream hands the decode to
  `hmmlearn`; this port implements the standard log-domain max-product
  formulation and keeps the lower predecessor on a tie, matching `np.argmax`
  everywhere else in crepe. On real data (float64 log scores) the two agree.

## Install

```bash
source /nvme0n1-disk/mojo-toolchain/activate.sh   # mojo 1.2.0.dev2026092605
bash build/build.sh                              # -> dist/libmojo-crepe.so
PYTHONPATH=python pytest tests -q
python bench/bench.py
```

## Use

```python
import numpy as np
import mojo_crepe as mc

# what the CNN would hand to crepe.predict: (T, 360)
activation = np.load("salience.npy")

time, frequency, confidence, activation = mc.predict_from_activation(activation)
time, frequency, confidence, activation = mc.predict_from_activation(
    activation, viterbi=True          # the to_viterbi_cents branch
)
```

`mc.frame_normalize(audio)` returns the (T, 1024) normalised framing that
`get_activation` feeds the CNN, and `mc.viterbi_path(activation)` returns the
decoded bin path, which upstream never exposes.

## Tests

28 tests. `crepe` cannot be imported in the test venv — `crepe.core` imports
keras, hmmlearn and resampy at module scope, and `crepe` is not installed at
all — so `tests/upstream_ref.py` is a line-for-line NumPy transcription of the
upstream expressions it mirrors (`to_local_average_cents`, `to_viterbi_cents`,
`get_activation`, `predict`), and the analytic tests are the second opinion:

- an impulse must light up exactly the frames whose window covers it, at the
  right column (pins the hop and the frame origin);
- a one-hot salience row must return the exact bin-to-cents mapping value (pins
  the table, its step and its offset);
- peaks 8 bins apart must pull the average toward the middle, and 9 bins apart
  must not (pins the 9-bin window);
- two tied maxima must resolve to the first bin (pins the argmax rule);
- the decoded path may never jump more than 12 bins between frames, because
  `max(12 - |i - j|, 0)` is zero beyond that (a kernel that dropped the
  transition term sails across the whole range);
- the total log-probability of the returned path must equal the DP optimum
  (pins the backtrack, not just the forward pass);
- one octave is exactly a doubling, and NaN cents map to 0 Hz.

## Benchmark

30 s of 16 kHz audio, a 3000x360 salience matrix, 1 M cents. `python
bench/bench.py` re-checks correctness before every timing, so a drifting kernel
fails the gate instead of posting a good number. NumPy is the reference because
crepe cannot run here at all; the NumPy formulations are the vectorised ones
(masked gather plus weighted reduction for the local average, a per-frame
broadcast max for the decode), not per-row Python loops.

| case | numpy | mojo-crepe | ratio |
| --- | ---: | ---: | ---: |
| frame-normalize, 30 s audio | 68.9 ms | 18.7 ms | 3.68x |
| local-average-cents, 3000x360 | 3.47 ms | 2.79 ms | 1.25x |
| viterbi decode, 1000 frames | 1455 ms | 636 ms | 2.29x |
| cents-to-hz, n=1048576 | 134 ms | 67.0 ms | 2.00x |
| predict tail, 2000 frames | 3168 ms | 603 ms | 5.26x |

The frame normalisation is the clearest win: it is three passes over
contiguous data that NumPy has to make through a strided view, and the read,
the two reductions and the write all fit in one compiled loop. The Viterbi
decode is sequential in frames — frame `t` depends on frame `t - 1` — so it is
one serial kernel with no thread pool; the win there is from reading both
matrices row-major in the inner loop instead of NumPy's broadcast temporary.

Measured on a shared 72-core Xeon E5-2697 v4; run-to-run spread on this box is
a few percent.
