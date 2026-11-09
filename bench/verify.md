# mojo-crepe VERIFY phase — independent re-measurement (2026-11-09, second pass)

Date: 2026-11-09 (UTC). Machine: Intel Xeon E5-2697 v4, 72 threads (`nproc=72`),
x86_64, pixi env python 3.13.14, NumPy 2.5.1. This is a fresh verify pass over
the already-corrected README (committed as `943c946 phase verify: agent pass`).
Upstream `crepe` is **not importable** here (`import crepe` → `ModuleNotFoundError`;
`keras`/`hmmlearn`/`resampy` likewise absent), so every "vs upstream" number below
is against the NumPy vectorised reference in `bench/bench.py` / the loop
transcription in `tests/upstream_ref.py` — exactly what `pixi run bench` compares.
No numbers in this file were invented or copied from the README; all were measured
in this session.

## Command log (this session)

1. `pixi run build && pixi run test` — build OK; **56 passed**, 13 warnings
   (divide-by-zero `log` in `hmm_parameters`, invalid-divide in the
   zero-weight reference row; both pre-existing, both exercised on purpose).
2. `pixi run bench` (solo, two full runs) plus 3× reruns of each `bench_*`
   function via `pixi run python` importing `bench/bench.py`, plus 6 extra solo
   reruns of the frame-normalize case (noisiest row). No lock contention this
   session: the first `pixi run bench` returned immediately.

| case | `pixi run bench` run A (numpy / mojo / ratio) | run B | 3× rerun spread |
| --- | --- | --- | --- |
| frame-normalize, 30 s audio | 28.90 / 14.06 / 2.06x | 30.41 / 13.54 / 2.25x | numpy 29.4–40.7, mojo 14.7–16.4, ratio 1.98–2.76x |
| local-average-cents, 3000x360 | 1.32 / 2.08 / 0.64x | 1.34 / 2.08 / 0.64x | numpy 1.22–1.33, mojo 1.96–2.18, ratio 0.58–0.68x |
| viterbi decode, 1000 frames | 434.03 / 122.03 / 3.56x | 412.71 / 117.06 / 3.53x | numpy 395–458, mojo 123–144, ratio 2.97–3.71x |
| cents-to-hz, n=1048576 | 31.24 / 25.82 / 1.21x | 30.50 / 22.99 / 1.33x | numpy 31.2–35.6, mojo 25.0–27.8, ratio 1.12–1.43x |
| predict tail, 2000 frames | 873.74 / 233.93 / 3.74x | 908.15 / 232.24 / 3.91x | numpy 770–823, mojo 229–268, ratio 2.91–3.65x |

6 extra solo frame-normalize reruns: numpy 31.1–39.0 ms, mojo 14.2–16.2 ms,
ratio 2.19–2.56x. The NumPy column moves up to ~±20% run to run on this shared
box (best-of-3 numpy hit 28.9 ms once vs ~40 ms typical); the Mojo column is
stable to ~5%.

## README table — reproduced or corrected?

Criterion: a figure that does not reproduce within 20% gets corrected in the
README to the number measured in this session. The README at HEAD already
carries the first verify pass's run-1 numbers; I compare against those.

| case | README numpy | measured numpy (run A) | dev | verdict | README mojo | measured mojo | dev | verdict | README ratio | measured ratio | verdict |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| frame-normalize, 30 s audio | 38.36 ms | 28.90 ms | −25% | **CORRECTED** | 15.61 ms | 14.06 ms | −10% | reproduces | 2.46x | 2.06x | reproduces (−16%) |
| local-average-cents, 3000x360 | 1.37 ms | 1.32 ms | −4% | reproduces | 2.24 ms | 2.08 ms | −7% | reproduces | 0.61x | 0.64x | reproduces |
| viterbi decode, 1000 frames | 442.70 ms | 434.03 ms | −2% | reproduces | 117.87 ms | 122.03 ms | +4% | reproduces | 3.76x | 3.56x | reproduces |
| cents-to-hz, n=1048576 | 37.24 ms | 31.24 ms | −16% | reproduces | 26.69 ms | 25.82 ms | −3% | reproduces | 1.40x | 1.21x | reproduces (−14%) |
| predict tail, 2000 frames | 852.94 ms | 873.74 ms | +2% | reproduces | 227.27 ms | 233.93 ms | +3% | reproduces | 3.75x | 3.74x | reproduces |

So 14 of 15 figures reproduce within 20%. The one that does not — the
frame-normalize NumPy column, 38.36 ms vs measured 28.90 ms (−25%) — is the
noisiest cell in the table: the same function measured 29.4–40.7 ms across the
3× reruns this session (and 28.8–39.9 ms in the first pass), so 38.36 sits at
the top of the observed spread rather than outside it. The ratio for that row
still reproduces (2.46x vs 2.06x, −16%, inside 20%) because the Mojo column is
stable.

### What changed in the README

- Frame-normalize NumPy column: 38.36 → 28.90 ms, and the row ratio 2.46x →
  2.06x, to the numbers measured in this session (`pixi run bench` run A).
  Nothing else in the table moved: every other figure reproduced within 20%
  and is left as-is.
- The footnote under the table now reports this session's spread (NumPy
  column up to ~±25% on this shared box for frame-normalize, Mojo ~5%) and
  notes that the frame-normalize NumPy cell was re-corrected this pass.
- The local-average-cents row stays slower than the NumPy reference (0.64x,
  README 0.61x — reproduces) and stays in the table un-reworded, per the task
  rules. No row was removed or softened.

## Fairness checks (§4 of the task) — same input, no skipped work

I read `bench/bench.py`, `tests/upstream_ref.py`, `python/mojo_crepe/core.py`
and `_lib.py`, then probed each case in-session:

- **Same input both sides.** Every case builds one input array and hands it to
  both sides: 30 s of 16 kHz audio (`default_rng(0)`) for frame-normalize;
  `rng.random((3000, 360))` (`default_rng(1)`) for local-cents;
  `rng.random((1000, 360))` → `obs = argmax(axis=1)` (`default_rng(2)`) for
  Viterbi; `rng.uniform(0, 9200, 1<<20)` (`default_rng(3)`) for cents-to-hz;
  `rng.random((2000, 360))` (`default_rng(4)`) for predict tail. Each
  `bench_*` function asserts agreement with its reference *before* timing, so
  a kernel that drifted would fail the gate rather than post a good number.
  All gates passed in every run above.
- **No skipped work on either side.**
  - Frame-normalize: the NumPy side times `ref_frame_normalize(windows)`
    (copy of the strided view + mean + std + clipped divide, 2994×1024); the
    Mojo side times `mc.frame_normalize(audio)` (fused 3-pass kernel over the
    same 2994 frames). Verified `np.shares_memory(windows, audio)` is True —
    both sides operate on the same samples — and max abs output diff is
    2.9e-15. The Mojo side saves the strided→contig copy by reading the 3.8 MB
    audio directly; that *is* the optimisation being measured, not skipped
    work; the arithmetic is identical.
  - Local-cents: the NumPy side is the vectorised masked-gather formulation,
    the Mojo side the argmax kernel; both compute the 9-bin weighted mean over
    all 3000 rows. The vectorised NumPy path agrees with the loop
    transcription `ref_local_average_cents` to 3.6e-12 max diff (re-measured
    this session), so the fast reference is not computing a cheaper thing.
  - Viterbi: the NumPy side times `ref_viterbi_path(obs)` (rebuilds the
    360×360 log matrices + per-frame broadcast max over 1000 frames +
    backtrack); the Mojo side times `mc.viterbi_path(act)` (argmax + Python
    `hmm_parameters` + kernel + Python backtrack). Both rebuild the HMM
    parameters per call, so neither side amortises what the other pays. The
    Mojo side is slightly *penalised*: measured `hmm_parameters` 3.77 ms vs
    `ref_hmm_parameters` 1.41 ms. Paths are asserted identical
    (`got.tolist() == ref...tolist()`).
  - Cents-to-hz: pure elementwise `10 * 2 ** (cents/1200)` on the same 1 M
    array; asserted `rtol=1e-11` before timing.
  - Predict tail: both sides do row-max + Viterbi path + centred local cents
    + cents-to-hz on the same 2000×360 matrix, timed whole. The vectorised
    centred gather agrees with the loop composition to 0.0 max diff on a
    50×360 sample this session.
- One asymmetry, unchanged from the first pass: `bench_viterbi` times NumPy
  once (`repeats=1`) but Mojo three times (best-of). Given the Mojo column's
  ~5% stability this inflates the Mojo advantage by at most a few percent,
  well under the correction threshold, and is visible in the source. I did
  not change it (Rules: code changes only for a bug shown by a failing test).

## Tests — parity vs upstream, and `git log -p -- tests/`

`git log -p -- tests/` re-checked this session: 891 lines, spanning `69c5e17`
(initial port, adds `tests/` wholesale) and `36feb4c` (test phase: +28 test
functions, 25 → 53).

- **No test function was removed.** `grep -E "^-.*def test_"` over the full
  diff returns zero lines. 28 functions were added (degenerate inputs,
  `hmm_parameters` parity, the `to_viterbi_cents` composition, the clip
  behaviour). Nothing to restore.
- **No pre-existing tolerance was loosened.** All `rtol`/`atol` present at
  `69c5e17` are byte-identical at HEAD; `grep` over the diff shows tolerance
  lines only as additions. The only changed assertions are the rewritten
  constant-frame docstring and the `ref_frame_normalize` clip fix (raw
  `std` → `np.clip(std, 1e-8, None)`), which *restores* upstream parity —
  `get_activation` really does clip, so the old raw-std reference was the
  bug, not the fix.
- Two newly added tests use `rtol=1.0` with a small `atol`
  (`test_frame_normalize_clips_std_at_1e_minus_8`, `atol=1e-3`, and
  `test_frame_normalize_non_representable_constant_is_tiny_not_nan`,
  `atol=1e-6`). These are not loosenings of pre-existing assertions; they pin
  the clip fix. I re-verified this session that they still fail on the bug
  they guard: a raw-std kernel would report ~32.0 (measured by simulation:
  31.98) on the near-constant case, far above the asserted `max < 1e-3`
  bound; the measured Mojo max abs is 9.99e-5. The loose `rtol` is justified
  (the numerator is a catastrophic-cancellation quantity amplified by 1e8).
- `tests/upstream_ref.py` is still a faithful transcription of the five
  upstream expressions it mirrors; the analytic tests (impulse/hop, one-hot
  mapping, 9-bin window, tie rule, ≤12-bin step, DP-optimum score, octave
  doubling, NaN→0) are unchanged since the initial port.

Conclusion: no removed assertions, no loosened tolerances to restore.
**Nothing in `tests/` was restored or changed in this phase.**

## What I could not do

- **Measure against real upstream `crepe`.** It cannot be imported in this
  environment. All "vs upstream" figures are vs the NumPy transcription, as
  the README and `bench/bench.py` both state. The Viterbi row in particular
  should be re-measured against `hmmlearn`'s decode if it ever becomes
  importable: the transcription is the standard log-domain form, not
  `hmmlearn` itself.
- **Pin the frame-normalize NumPy cell to better than ±25%.** Best-of-3 on a
  shared 72-core box moves too much run to run. The README number is one
  `pixi run bench` run's figure with the spread stated alongside it; do not
  quote it outside this box without re-measuring.

## What I changed (this phase)

- `README.md`: frame-normalize NumPy column 38.36 → 28.90 ms and ratio
  2.46x → 2.06x (the only figure outside 20% this pass); footnote spread
  updated to this session's re-measurement. The local-cents row (0.61x,
  slower than the NumPy reference) reproduces and is kept un-reworded.
- `bench/verify.md`: this file (rewritten for this session's measurements).
- **No code changed.** Per the rules, code changes require a bug shown by a
  failing test; `pixi run build && pixi run test` was green (56 passed)
  before and after the README edit, and the probes above found no such bug.

## Final gate

`pixi run build && pixi run test` — **56 passed**, 13 warnings.
