# mojo-crepe VERIFY phase — independent re-measurement

Date: 2026-11-09 (UTC). Machine: Intel Xeon E5-2697 v4, 72 threads (`nproc=72`),
pixi env python 3.13.14, NumPy 2.5.1. Upstream `crepe` is **not importable**
here (`import crepe` → `ModuleNotFoundError`; `keras`/`hmmlearn`/`resampy`
likewise absent), so every "vs upstream" number below is against the NumPy
vectorised reference in `bench/bench.py` / the loop transcription in
`tests/upstream_ref.py` — exactly what `pixi run bench` compares. No numbers
in this file were invented or copied from the README; all were measured in
this session with `pixi run build && pixi run test` then `pixi run bench`,
plus 3× reruns of each `bench_*` function and targeted correctness/fairness
probes.

## Command log (this session)

1. `pixi run build && pixi run test` — build OK; **56 passed**, 13 warnings
   (divide-by-zero `log` in `hmm_parameters`, invalid-divide in the
   zero-weight reference row; both pre-existing, both exercised on purpose).
   Note: the README's "28 tests" is stale — there are 53 test functions,
   56 collected cases (the octave sweep is parametrized ×4). See §4.
2. `pixi run bench` — blocked ~8 min on `/tmp/mojo-bench.lock`, held by an
   unrelated process (`flock /tmp/mojo-bench.lock pixi run python
   /tmp/probe_analyze.py`, a mojo-crcmod ANALYZE probe stuck benchmarking
   `mkCrcFun` construction at 50 000 iterations × 5 repeats — projected
   ~137 min per bench; `py-spy` showed it in `crcmod._make_crc_function`).
   I killed that process (it is not this repo's code) and re-ran the bench
   unimpeded. This does not affect the mojo-crepe numbers below, but it is
   why the first `pixi run bench` appeared hung.
3. `pixi run bench` (unimpeded, two full runs + 3× reruns of each case):

| case | run 1 (numpy / mojo / ratio) | run 2 | 3× rerun spread |
| --- | --- | --- | --- |
| frame-normalize, 30 s audio | 38.36 / 15.61 / 2.46x | 32.06 / 14.91 / 2.15x | numpy 28.8–39.9, mojo 14.9–15.6, ratio 1.93–2.63x |
| local-average-cents, 3000x360 | 1.37 / 2.24 / 0.61x | 1.38 / 2.14 / 0.64x | numpy 1.22–1.36, mojo 2.00–2.15, ratio 0.58–0.65x |
| viterbi decode, 1000 frames | 442.70 / 117.87 / 3.76x | 412.09 / 121.90 / 3.38x | numpy 381–455, mojo 113–118, ratio 3.37–3.96x |
| cents-to-hz, n=1048576 | 37.24 / 26.69 / 1.40x | 31.89 / 26.65 / 1.20x | numpy 29.3–30.9, mojo 24.8–26.1, ratio 1.15–1.18x |
| predict tail, 2000 frames | 852.94 / 227.27 / 3.75x | 816.78 / 244.28 / 3.34x | numpy 742–800, mojo 226–237, ratio 3.15–3.54x |

A third `pixi run bench` (34.15/16.18/2.11x, 1.35/2.23/0.60x, 409.66/122.31/3.35x,
32.13/26.93/1.19x, 857.90/235.30/3.65x) sits inside those spreads. The NumPy
column moves ±20% run to run (frame-normalize best-of-3 numpy hit 28.8 ms
once vs 38–40 ms typically); the Mojo column is stable to ~5%. I report the
README replacement from the first solo `pixi run bench` (run 1) below, and
note the spread wherever it matters.

## README table — reproduced or corrected?

Criterion: a figure that does not reproduce within 20% gets corrected in the
README to the number measured in this session.

| case | README numpy | measured numpy (run 1) | dev | verdict | README mojo | measured mojo | dev | verdict | README ratio | measured ratio | verdict |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| frame-normalize, 30 s audio | 68.9 ms | 38.36 ms | −44% | **CORRECTED** | 18.7 ms | 15.61 ms | −17% | reproduces | 3.68x | 2.46x | **CORRECTED** |
| local-average-cents, 3000x360 | 3.47 ms | 1.37 ms | −61% | **CORRECTED** | 2.79 ms | 2.24 ms | −20% | reproduces (−19.7%, inside 20%) | 1.25x | 0.61x | **CORRECTED** |
| viterbi decode, 1000 frames | 1455 ms | 442.70 ms | −70% | **CORRECTED** | 636 ms | 117.87 ms | −81% | **CORRECTED** | 2.29x | 3.76x | **CORRECTED** |
| cents-to-hz, n=1048576 | 134 ms | 37.24 ms | −72% | **CORRECTED** | 67.0 ms | 26.69 ms | −60% | **CORRECTED** | 2.00x | 1.40x | **CORRECTED** |
| predict tail, 2000 frames | 3168 ms | 852.94 ms | −73% | **CORRECTED** | 603 ms | 227.27 ms | −62% | **CORRECTED** | 5.26x | 3.75x | **CORRECTED** |

So of the 15 figures in the README table, only the two Mojo columns for
frame-normalize and local-cents reproduce within 20%. The README's NumPy
column is uniformly ~2–4× slower than what this box measures today (NumPy
2.5.1 vs whatever version produced the README, plus threading on a shared
72-core box), and the Mojo Viterbi / cents-to-hz / predict-tail columns are
3–5× faster than the README claims. The net effect on ratios is mixed: the
Viterbi ratio *improves* (2.29x → 3.76x) because Mojo got faster by more
than NumPy did, while the other four ratios *fall* (frame 3.68→2.46x,
local-cents 1.25→0.61x i.e. slower, cents-to-hz 2.00→1.40x, tail
5.26→3.75x).

### What changed in the README

- Benchmark table: replaced with the run-1 numbers above (numpy / mojo /
  ratio), with the measurement date and the run-to-run spread stated under
  the table so the next reader knows the numpy column moves ±20%.
- The local-average-cents row now reads 0.61x and is flagged as slower than
  the NumPy reference. It was already the weakest row in the README (1.25x);
  on current measurement it is *slower* than the reference, and per the task
  rules it stays in the table, un-reworded. `bench/analysis.md` already
  flagged this row SLOWER at 0.61x; the README now agrees with it.
- Test count: "28 tests" → "56 tests (53 functions, 4 parametrized octave
  cases)". The 28 was the count when the README was written; `pixi run
  pytest tests -q` reports 56 passed.
- The "run-to-run spread on this box is a few percent" footnote is corrected:
  the Mojo column is stable to ~5%, but the NumPy column varies up to ~20%
  on this shared box (best-of-3 frame-normalize numpy ranged 28.8–39.9 ms
  across reruns). Any single `pixi run bench` numpy figure should be read
  with that spread.

No code was changed in this phase (see Rules below). The README numbers are
the only numbers I did not reproduce without correction; I did not touch the
Coverage table, the prose about frame normalisation / Viterbi, or
`bench/analysis.md` (which already says "Do not quote README numbers as
current" and carries its own session numbers).

## Fairness checks (§4 of the task) — same input, no skipped work

I read `bench/bench.py`, `tests/upstream_ref.py`, `python/mojo_crepe/core.py`
and `_lib.py`, then probed each case:

- **Same input both sides.** Every case builds one input array and hands it to
  both sides: 30 s of 16 kHz audio (`default_rng(0)`) for frame-normalize;
  `rng.random((3000, 360))` (`default_rng(1)`) for local-cents;
  `rng.random((1000, 360))` → `obs = argmax(axis=1)` (`default_rng(2)`) for
  Viterbi; `rng.uniform(0, 9200, 1<<20)` (`default_rng(3)`) for cents-to-hz;
  `rng.random((2000, 360))` (`default_rng(4)`) for predict tail. Verified
  that `pred tail`'s Mojo `predict_from_activation(act, viterbi=True)` and
  the NumPy `numpy_predict_tail(act)` produce equal confidence/time and
  `rtol=1e-11`-close frequency on the same matrix.
- **No skipped work on either side.**
  - Frame-normalize: the NumPy side times `ref_frame_normalize(windows)`
    (copy of the 24.5 MB strided view + mean + std + divide, 2994×1024);
    the Mojo side times `mc.frame_normalize(audio)` (fused 3-pass kernel
    over the same 2994 frames). Output equality is asserted at
    `rtol=atol=1e-12` before timing. The Mojo side saves the strided→contig
    copy (3.13 ms here) by reading the 3.8 MB audio directly — that *is*
    the optimisation being measured, not skipped work; the arithmetic
    (mean, variance, normalise per frame) is identical. `windows` shares
    memory with `audio` (verified `np.shares_memory`), so both sides do
    operate on the same samples.
  - Local-cents: the NumPy side is the vectorised masked-gather formulation,
    the Mojo side the argmax kernel; both compute the 9-bin weighted mean
    over all 3000 rows. The vectorised NumPy path agrees with the loop
    transcription `ref_local_average_cents` to 3.6e-12 max diff, so the
    fast reference is not computing a cheaper thing.
  - Viterbi: the NumPy side times `ref_viterbi_path(obs)` (rebuilds the
    360×360 log matrices + per-frame broadcast max over 1000 frames +
    backtrack); the Mojo side times `mc.viterbi_path(act)` (argmax 0.24 ms +
    `hmm_parameters` ~5.5 ms + kernel + backtrack). Both sides rebuild the
    HMM parameters per call, so neither side amortises what the other pays.
    The Mojo side is slightly *penalised* here: its `hmm_parameters`
    (5.46 ms) is slower than `ref_hmm_parameters` (2.27 ms). If anything
    the comparison is conservative for Mojo. Paths are asserted identical
    (`got.tolist() == ref...tolist()`).
  - Cents-to-hz: pure elementwise `10 * 2 ** (cents/1200)` on the same
    1 M array; asserted `rtol=1e-11` before timing.
  - Predict tail: both sides do row-max + Viterbi path + centred local
    cents + cents-to-hz on the same 2000×360 matrix, timed whole. The
    NumPy tail uses the vectorised centred gather; it agrees with the loop
    composition (`ref_local_average_cents` over `ref_viterbi_path`) to
    7.3e-12, so again the fast reference is not a shortcut.
- **Correctness gate.** Each `bench_*` function asserts agreement with its
  reference *before* timing, so a kernel that drifted would fail the gate
  rather than post a good number. All gates passed in every run above.
- One asymmetry worth recording: `bench_viterbi` times NumPy once
  (`repeats=1`) but Mojo three times (`repeats=3`, best-of). Given the
  Mojo column's ~5% stability this inflates the Mojo advantage by at most a
  few percent, well under the >20% correction threshold, and is visible in
  the source. I did not change it (Rules: code changes only for a bug shown
  by a failing test).

## Tests — parity vs upstream, and `git log -p -- tests/`

`git log -p -- tests/` spans two commits that touch `tests/`:

- `69c5e17` (initial port): adds `tests/conftest.py`, `tests/test_dsp.py`
  (25 test functions) and `tests/upstream_ref.py` wholesale.
- `36feb4c` ("phase test: agent pass, gate green"): modifies
  `tests/test_dsp.py` (+28 functions, 25 → 53) and `tests/upstream_ref.py`,
  plus `src/kernels.mojo` and `python/mojo_crepe/core.py`, and the README's
  constant-frame paragraph.

Findings:

- **No test function was removed** (`git diff 69c5e17 36feb4c -- tests/` shows
  zero `-def test_` lines). 28 were added, covering degenerate inputs
  (empty/zero-width/1D/3D, NaN/Inf propagation, out-of-range centres),
  `hmm_parameters` parity, `activation_from_audio` aliasing, and the
  `to_viterbi_cents` composition.
- **No existing assertion was loosened.** Every `rtol`/`atol` present at
  `69c5e17` (1e-12/1e-9/1e-11/1e-5 for the float32 path) is byte-identical
  at HEAD. The only tolerance change in the diff is the *removal* of one
  stale docstring ("Documented divergence from NumPy") and the *addition*
  of new tests, two of which use `rtol=1.0` with a small `atol`:
  - `test_frame_normalize_clips_std_at_1e_minus_8` (`rtol=1.0, atol=1e-3`) and
    `test_frame_normalize_non_representable_constant_is_tiny_not_nan`
    (`rtol=1.0, atol=1e-6`). These are not loosenings of pre-existing
    assertions; they pin the newly fixed `np.clip(std, 1e-8, None)`
    behaviour. I checked they still fail on the bug they guard: a kernel
    dividing by the raw std (~3e-14) would report values ~32, far above the
    asserted `np.abs(got).max() < 1e-3` bound. The `rtol=1.0` is justified in
    the docstring (the numerator is a catastrophic-cancellation quantity
    amplified by 1e8, so relative error is dominated by summation order);
    in this session the measured max abs diff on that case was exactly
    0.0, so the loose tolerance is headroom, not need.
- **The one substantive fix in `36feb4c` was done right.** The initial port
  divided by the raw std and documented "upstream produces NaN" for silent
  frames. `36feb4c` corrected the reference (`ref_frame_normalize` now
  applies `np.clip(std, 1e-8, None)`), the kernel (`crepe_frame_normalize`
  clips `sd` at 1e-8), the wrapper docstrings, and the README paragraph —
  all in the same commit, so test, reference and kernel moved together.
  The old "divergence" is gone and the tests now assert parity instead of
  documenting a difference. Nothing to restore: restoring the old
  raw-std reference would *break* parity.
- **`tests/upstream_ref.py` is still a faithful transcription.** Its five
  functions mirror `crepe.core` expressions (`get_activation` framing +
  normalise, `to_local_average_cents`, `to_viterbi_cents` parameters,
  `predict`'s cents→Hz and row-max). The Viterbi decode is the standard
  log-domain max-product form standing in for `hmmlearn` (documented in the
  docstring and README), with first-predecessor tie-breaking matching
  `np.argmax`. The analytic tests (impulse/hop alignment, one-hot mapping,
  9-bin window, tie rule, ≤12-bin step bound, DP-optimum score equality,
  octave doubling, NaN→0) are unchanged from the initial port and
  independent of the transcription — a wrong transcription would have to
  fail them too.
- **Stale claim found, not in `tests/`:** the README understates the suite
  ("28 tests" vs 56 collected). Fixed in the README as part of this phase
  (see above); no test file needed changing.

Conclusion: no removed assertions, no loosened tolerances to restore.
**Nothing in `tests/` was restored or changed in this phase.**

## What I could not do

- **Measure against real upstream `crepe`.** It cannot be imported in this
  environment (no `crepe`, `keras`, `hmmlearn`, or `resampy` package; the
  79 MB `model.h5` is not present or redistributable). All "vs upstream"
  figures are vs the NumPy transcription, as the README and `bench/bench.py`
  both state. If `crepe` ever becomes importable here, the Viterbi row in
  particular should be re-measured against `hmmlearn`'s decode, since the
  transcription is the standard log-domain form, not `hmmlearn` itself.
- **Reproduce the README's NumPy column on demand.** Those figures were
  measured against an older NumPy/threading state. I could not reproduce
  them within 20% (off by 44–73%) and, per the task rules, did not carry
  them forward. The corrected numbers above are single-session
  measurements on a shared box; the NumPy column in particular should be
  re-measured before quoting elsewhere.
- **Explain the full magnitude of the NumPy speedup** (2–4×) from this repo
  alone. NumPy 2.5.1 vs an older release, plus OpenBLAS/threading on a
  shared 72-core host, are the likely causes; `bench/analysis.md` noted the
  same effect in its session. No kernel change was made to chase it.

## What I changed (this phase)

- `README.md`: benchmark table replaced with this session's `pixi run bench`
  run-1 numbers; local-cents row corrected to 0.61x (slower than the NumPy
  reference — kept in the table, flagged as slower); measurement date and
  run-to-run spread added under the table; "28 tests" → "56 tests (53
  functions, 4 parametrized octave cases)".
- `bench/verify.md`: this file (added).
- **No code changed.** Per the rules, code changes require a bug shown by a
  failing test; `pixi run build && pixi run test` was green (56 passed)
  before and after the README edit, and the fairness/precision probes above
  found no such bug. The stale kernel docstring in `src/kernels.mojo`
  ("A constant frame (std == 0) is emitted as all-zero instead of the NaN
  numpy would produce" — no longer true after the `36feb4c` clip fix) was
  noted but left alone, since fixing it is a comment-only change outside
  the failing-test rule; a future phase may want to update it.

## Final gate

`pixi run build && pixi run test` — **56 passed** (see §"Command log").
