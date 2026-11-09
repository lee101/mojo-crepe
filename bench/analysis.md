# mojo-crepe ANALYZE phase — measured map of where time goes

Date: 2026-11-09. Machine: Intel Xeon E5-2697 v4, 72 threads (`nproc=72`),
numpy 2.5.1 (OpenBLAS, X86_V3 detected), pixi env python 3.13.
Upstream `crepe` is **not importable** in this env (`import crepe` fails in
both system and pixi python; `crepe.core` needs keras/hmmlearn/resampy at
module scope, per `bench/bench.py` footer and `tests/upstream_ref.py`).
All "vs upstream" comparisons below are therefore vs the NumPy reference
transcription in `tests/upstream_ref.py` / the vectorised NumPy timings in
`bench/bench.py` — exactly what `pixi run bench` compares against. Rows where
the port is slower than that reference are kept and flagged **SLOWER**.

Method: `pixi run bench` (flock `/tmp/mojo-bench.lock`, run solo) plus
additional in-session timings via `pixi run python` timing the public entry
points in `python/mojo_crepe/core.py` and the `_lib` (ctypes) layer directly,
min-of-5 (or 3/20 for small cases), after 2 warm-up calls. Bytes/flops are
counted from the kernel source (`src/kernels.mojo`) at the measured shape —
not estimated times. float64 throughout, so 1 element = 8 B (psi is int32,
4 B). "Bytes touched" counts bytes the kernel reads + writes (re-reads count
once per pass); unique-byte counts are given where they differ materially
(frame-normalize audio reuse).

`pixi run bench` this session (best-of-5):

| case | numpy | mojo-crepe | ratio |
|---|---:|---:|---:|
| frame-normalize 30 s audio | 27.20 ms | 15.53 ms | 1.75x |
| local-average-cents 3000x360 | 1.21 ms | 1.98 ms | 0.61x — **SLOWER** |
| viterbi decode 1000 frames | 391.56 ms | 116.20 ms | 3.37x |
| cents-to-hz n=1048576 | 29.61 ms | 26.01 ms | 1.14x |
| predict tail 2000 frames | 774.56 ms | 255.98 ms | 3.03x |

README quotes 68.9/18.7, 3.47/2.79, 1455/636, 134/67.0, 3168/603 ms on the
same class of box — i.e. this box is ~2.5x faster on the NumPy reference
than when the README was written (NumPy 2.5.1 vs older, plus different
threading). Ratios differ, especially `local_cents` and `cents_to_hz`.
Do not quote README numbers as current.

## Ranked hotspot table

Ranked by mojo-crepe wall time at the bench size (best-of, with median in
the profiling run for small cases). "GPU" verdicts in the last column cite
the flops-per-byte figure from this table.

| # | function | input size | time (mojo) | bytes touched | flops | flops/byte | memory- or compute-bound | Python/ctypes glue or kernel |
|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | `crepe_viterbi` (via `viterbi_path` / `to_viterbi_cents` / `predict_from_activation(viterbi=True)`), 1000 frames x 360 states | 1000 frames | ~102 ms in `_lib.viterbi` (incl. ~0.27 ms Python backtrack); 116–129 ms end-to-end via `viterbi_path` | ~1,041 MB (logT 1.037 MB re-read per frame x 1000 + logE 2.9 MB + psi write 1.44 MB) | ~260 M (2 flop per (t,i,j): add+cmp; + init/emission) | 0.25 | memory-bound (logT streaming; ~10.2 GB/s achieved) | kernel; ~0.2% backtrack loop is Python (see §4) |
| 2 | `crepe_frame_normalize` (via `frame_normalize`), 30 s audio | 480,000 samples -> 2,994 x 1,024 frames | 14.7 ms (`_lib`), 14.7–18.3 ms end-to-end | 98 MB touched (3 read passes over 24.5 MB window bytes + 24.5 MB write; unique: 3.8 MB audio + 24.5 MB out) | ~18.4 M (6 flop/out elem: mean add; var sub+mul+add; write sub+div) | 0.19 (0.65 unique) | memory-bound (~6.5 GB/s touched) | kernel; ~0 valid glue (public minus `_lib` ≈ 0) |
| 3 | `crepe_cents_to_hz`, n=1,048,576 | 1.05 M els | 13.9 ms (`_lib`), 14.5–26 ms end-to-end (bench: 26.0 ms) | 16.8 MB (8.4 MB read + 8.4 MB write) | ~17.8 M (div+mul+exp2, exp2 counted ~15 flop equiv. => ~17/elt) | 1.06 | compute-bound at the libm `exp2` (only 1.2 GB/s — far below memory BW, so it is not bandwidth-limited) | kernel; ~0.6 ms glue (`f64` copy + out alloc) at 1 M |
| 4 | `hmm_parameters` (pure Python/NumPy — called fresh on **every** `to_viterbi_cents` / `viterbi_path` / `predict_from_activation(viterbi=True)`) | 360 states | ~3.6–5.9 ms (vs 1.4 ms for the `ref_hmm_parameters` equivalent — **SLOWER**) | ~3.1 MB written (log_start 2.9 KB + logT 1.037 MB + logE^T 1.037 MB), plus ~0.5 MB transient (`transition`, `emission` before `log`) | ~5.8 M (abs/sub/max/div per 129.6 k-band + 3 x 129.6 k `log`, log ~15 flop equiv.) | ~1.9 | memory-bound (builds 3 x 360x360 temporaries, writes 3 matrices) | Python/ctypes **glue** (no kernel at all); pure NumPy allocations + `log` calls |
| 5 | `crepe_local_cents` (via `to_local_average_cents`, argmax centre), 3000 x 360 | 1.08 M els | 2.08 ms (`_lib`), 2.15–1.98 ms end-to-end — **SLOWER** than vectorised NumPy (1.21–1.37 ms) | 8.64 MB read (whole matrix for argmax) + 24 KB write | ~1.21 M (359 cmp/row + 9 window elts x ~5 flop: `mapping_at` mul+add, mul+add, add) | 0.14 | memory-bound (~4.2 GB/s — should be ~10 GB/s like Viterbi, so it is scalar-lossy, not saturating BW) | kernel; ~0.1 ms glue |
| 6 | `crepe_row_max`, 3000 x 360 | 1.08 M els | 1.39 ms (`_lib` and end-to-end) — **SLOWER** than `np.max(axis=1)` (0.44 ms, ~3.2x) | 8.64 MB read + 24 KB write | ~1.08 M (359 cmp/row, NaN branch per cmp) | 0.125 | memory-bound in principle (~6.2 GB/s) but scalar `max` lane is ~3x behind vectorised NumPy, so it is **latency/scalar-bound** as built | kernel |
| 7 | `crepe_local_cents_centred` (Viterbi centre), 3000 x 360 | 1.08 M els | 62 µs — faster than NumPy centred gather (0.66 ms) | 216 KB window bytes read (9 x 3000 x 8 B) + 2.9 KB centres + 24 KB write; (full matrix never read) | ~135 k (9 x ~5 flop/row) | 0.55 | neither (latency-bound at 62 µs; over-provisioned for its work) | kernel |
| 8 | ctypes floor (any entry): `cents_to_hz` n=1 / n=10 / n=100 | 1–100 els | 5.3 µs (n=1), 4.7 µs (n=10), 6.2 µs (n=100) | 8–800 B | ~17–170 | ~1 | neither — pure call overhead | glue: ~5 µs per call (arg `.ctypes.data`, `ascontiguousarray` check, `np.empty`, CDLL dispatch); kernels add ~<1 µs below n~100 |

NOT a hotspot (verified, for completeness — do not touch in a hot-phase):

- Python backtrack in `_lib.viterbi`: 0.27 ms / 1000 frames (0.26% of the
  kernel) — a per-frame Python loop, but at 1000 iterations it is 0.27 ms
  against 102 ms of kernel. It scales linearly; at 40,000 frames it would be
  ~11 ms against ~4 s of kernel — still 0.26%. Do not optimise it before the
  kernel itself.
- `predict_from_activation` plumbing (time axis `arange`, tuple): 3.7 µs.
- `activation_from_audio`: alias of `frame_normalize`, same numbers.
- Glue at bench sizes: `frame_normalize` public ≈ `_lib` (0 ms gap);
  `local_cents` ≈ +0.1 ms (5%); `cents_to_hz` ≈ +0.6 ms (4%); `row_max` ≈ 0;
  `viterbi_path` end-to-end 117–129 ms vs `_lib.viterbi` 102–107 ms — the
  ~10–20 ms gap is `hmm_parameters` (see §4), not ctypes.

## Per-entry timing vs NumPy baseline (this session)

All at bench sizes unless noted. Best-of; medians in profiling runs agree
within a few percent.

| entry | mojo | baseline | verdict |
|---|---:|---:|---|
| `frame_normalize` 30 s (480 k samples) | 14.7–18.3 ms | 37.5–51.9 ms NumPy (`ref_frame_normalize` on strided `sliding_window_view`) | mojo 2–3.4x faster |
| `to_local_average_cents` 3000x360 (argmax) | 1.98–2.22 ms | 1.21–1.41 ms vectorised NumPy (masked gather) | **SLOWER (~1.6x)** |
| `to_local_average_cents` 3000 rows, centred | 62 µs | 0.66 ms NumPy centred gather | mojo ~10x faster |
| `to_local_average_cents` 1 row | 7.3–8.0 µs | 26–28 µs NumPy row ref | mojo ~3.5x faster |
| `to_local_average_cents` 10 / 100 rows | 14.7 µs / 79 µs | 61 µs / 110 µs | mojo faster |
| `viterbi_path` 1000 frames | 117–129 ms | 372 ms NumPy (frame-loop broadcast `argmax`, `tests/upstream_ref.ref_viterbi_path`) | mojo ~3.2x faster |
| `to_viterbi_cents` 1000 frames | 116 ms | — (includes `viterbi_path`) | same kernel |
| `cents_to_hz` n=1,048,576 | 14.5–26 ms | 30–36 ms NumPy (`ref_cents_to_hz`) | mojo ~1.2–2.5x faster (spread is noise on this box) |
| `cents_to_hz` n=100 k | 1.27 ms | 2.72 ms | mojo ~2.1x |
| `cents_to_hz` n=10 k / 1 k / 100 / 10 / 1 | 123 / 18.5 / 6.2 / 4.7 / 5.3 µs | 275 / 35.7 / 10.2 / 6.9 / 6.7 µs | mojo faster at every size; glue floor ~5 µs dominates below n~200 |
| `row_max` 3000x360 | 1.39 ms | 0.44 ms NumPy `max(axis=1)` | **SLOWER (~3.2x)** |
| `predict_from_activation` 2000 frames, viterbi | 245–256 ms | 774–816 ms NumPy tail | mojo ~3.2x faster |
| `predict_from_activation` 2000 frames, no viterbi | 2.49 ms | 1.35–1.40 ms NumPy tail (row_max + centred local + cents) | **SLOWER (~1.8x)** — driven by `row_max` + `local_cents` |
| `predict_from_activation` 10 frames, viterbi | 4.98 ms | — | dominated by the ~3.6–5.9 ms `hmm_parameters` rebuild, not the kernel (Viterbi on 10 frames ≈ 1.7 ms at the 100-frame rate) |
| `hmm_parameters` | 3.6–5.9 ms | 1.4 ms (`ref_hmm_parameters`, same expressions in probability space) | **SLOWER (~2.6–4x)** |

Viterbi frame scaling (mojo `viterbi_path`, best-of-2): 100 frames 16.8 ms,
500 frames 65.5 ms, 1000 frames 129 ms, 2000 frames 239 ms, 4000 frames
468 ms — linear in frames at ~0.117 ms/frame, i.e. ~0.9 µs per (frame x
360x360) sweep. No fixed-cost plateau; the kernel dominates at every size.

## Where time sits: glue vs kernel, and per-call copies

No Python-level loop re-enters ctypes per element anywhere except the
Viterbi **backtrack** (and `to_local_average_cents`'s 1D path reshapes once,
not per element). Every kernel is one ctypes call per public call, all
buffers passed as `c_int64` addresses (`_lib.addr`). That is good.

The real per-call costs are the **copies and allocations in the wrapper**:

- `np.ascontiguousarray(..., dtype=float64)` on **every** call in
  `core.py` and `_lib.f64/i32`. For an already-contiguous float64 input this
  is a no-op view (no copy), but it is still a function call + dispatch on
  every entry point. For a float32 / strided / non-contiguous caller input it
  makes one full copy per call (e.g. a float32 3000x360 caller pays 8.6 MB
  copy before the kernel even runs). Verified with `tracemalloc` peaks for
  one call (including the returned array):
  - `frame_normalize` 30 s: peak 24.5 MB — exactly the (2994,1024) output;
    input is ravel'd (view if float64, copy if not).
  - `local_cents` 3000: peak 24.7 KB — the 3000-float output only.
  - `viterbi_path` 1000: peak 5.2 MB — `psi` (1000x360 int32 = 1.4 MB) +
    4 x 360-float scratch + the `hmm_parameters` rebuild (~3.1 MB) re-made
    per call.
  - `cents_to_hz` 1 M: peak 8.4 MB — the output array.
  - `row_max` 3000: peak 24.6 KB — the 3000-float output.
  - `predict_from_activation` 2000 frames + viterbi: peak 5.2 MB — same
    `psi` + `hmm_parameters` rebuild; the activation matrix itself is
    ascontiguousarray'd (copied again if non-contiguous).
  - `hmm_parameters`: peak 5.2 MB for a 3.1 MB result — ~2 MB of
    intermediate temporaries (`transition`, `emission`, meshgrid-free outer
    diff), allocated and discarded on every Viterbi call.
- `_lib.viterbi` allocates `delta/nxt/acc/arg` (4 x 360 x 8 B) and
  `psi` (frames x 360 x 4 B) per call, and the Python backtrack reads
  `psi` (1.4 MB at 1000 frames) through Python indexing after the kernel
  returns. Bound and fine — 0.27 ms.
- `cents_to_hz` / `frame_normalize` / `local_cents` / `row_max` allocate
  exactly one output array per call — no scratch, no intermediate copies at
  bench size. The only copy is a dtype/contiguity copy of the caller's input
  when needed.

**Conclusion: glue is not the problem at bench sizes (0–5%), except for two
items — the per-call `hmm_parameters` rebuild (3.6–5.9 ms, pure Python/NumPy,
recomputed identically every call for the default 360-state case) and the
~5 µs fixed ctypes floor, which is only visible below n≈200 or <10 frames,
where it decides the `predict - 10 frames = 5.0 ms` row above.**

## Flops / bytes accounting per kernel (float64)

Formulas counted from `src/kernels.mojo`:

- `frame_normalize`: 6 flop/out elem (mean: 1 add; var: 1 sub + 1 mul + 1 add;
  write: 1 sub + 1 div). 3,065,856 out elems => 18.4 M flop. Bytes: 3 read
  passes over the 24.5 MB window bytes (audio itself is 3.8 MB unique) +
  24.5 MB write = 98 MB touched, 28.4 MB unique. 0.19 flop/B touched, 0.65
  flop/B unique. **Memory-bound.**
- `local_cents` (argmax): 359 comparisons/row + 9 window elts x ~5 flop
  (`mapping_at`: 1 mul + 1 add; weighted: 1 mul + 1 add; weight: 1 add) =>
  1.21 M flop over 8.64 MB read + 24 KB write. 0.14 flop/B. **Memory-bound**
  (but at 4.2 GB/s, not saturating — scalar NaN-branch + `mapping_at` per
  element are the suspects, see §Next phase).
- `local_cents_centred`: ~5 flop per window elt, 9 elts/row => 135 k flop.
  0.55 flop/B on the 216 KB of window bytes actually touched. Latency-bound
  at 62 µs.
- `row_max`: 359 comparisons/row => 1.08 M flop over 8.64 MB. 0.125 flop/B.
  **Memory-bound** in principle; as built, scalar-bound (~3x behind NumPy's
  vectorised `max`).
- `cents_to_hz`: per elt 1 div + 1 mul + `exp2` (~15 flop equiv.) => ~17
  flop/elt, 17.8 M flop over 16.8 MB. 1.06 flop/B, but it runs at 1.2 GB/s,
  i.e. 8x below the machine's streaming rate — it is **compute-bound on the
  scalar `exp2`** (Mojo `std.math.exp2` on float64, one element at a time,
  no SIMD) rather than bandwidth-bound.
- `viterbi`: 2 flop per (t,i,j) (add + compare-exchange, plus ~1 flop per
  (t,j) for init/emission) => ~260 M flop at 1000 frames. Bytes: `logT`
  1.037 MB re-read every frame => 1,037 MB + `logE` 2.9 MB + `psi` write
  1.44 MB = ~1,041 MB. 0.25 flop/B. **Memory-bound** (10.2 GB/s achieved —
  the best streaming rate in this table, so it is genuinely bandwidth-limited
  on the logT stream).
- `hmm_parameters`: 360x360 band construction (~4 flop/band elt) + `eye`/`ones`
  + 3 x 129.6 k `log` (~15 flop equiv. each) => ~5.8 M flop over ~3.1 MB
  written. ~1.9 flop/B. **Memory-bound** (transient temporaries + 3 full
  360x360 `log` passes).

## Next-phase verdict per hotspot

| # | hotspot | next phase | one-line reason (evidence) |
|---|---|---|---|
| 1 | `crepe_viterbi` 102 ms, 0.25 flop/B, 10.2 GB/s | **simd + unroll** (then parallel) | Already at streaming BW on the logT stream, but the inner loop is a scalar `max`/branch with no SIMD — SIMDising the (i,j) max-reduction cuts flops-per-byte headroom before adding threads. Parallelism cannot cross frames (sequential recurrence) but **can** split the 360-wide `j` dimension across workers; content-change note only: it needs `max.algorithm.parallelize` with an explicit worker count behind a size threshold (`MOJO_NOTES.md` §4/§5b — `std.algorithm.parallelize` does not compile, `parallelize(func, n, workers)` does). GPU: **no** — 0.25 flop/B is 8x below the ~2 flop/B GPU-worthiness rule in `MOJO_NOTES.md` §5, and the decode is a sequential recurrence with a 1 MB logT table re-streamed per frame; conversion cost would dominate. |
| 2 | `frame_normalize` 14.7 ms, 0.19 flop/B | **parallel** (simd second) | Pure streaming, 72 threads idle, rows are independent. `parallelize` over `t` (2,994 frames) with SIMD inside the per-frame mean/var passes. GPU: **no** — 0.19 flop/B, far below the ~2 flop/B rule; PCIe round-trip on 24.5 MB of output would cost more than the CPU pass does. |
| 3 | `cents_to_hz` 13.9 ms, 1.06 flop/B, 1.2 GB/s | **simd** | The `exp2` scalar loop is the bottleneck (8x below streaming BW). SIMD `exp2` over 4-wide lanes (or `exp2` via a polynomial) is the fix; parallel is second. GPU: **no** — 1.06 flop/B is below the ~2 flop/B rule, and at ≤1 M elements the answer fits in L3; GPU copy-out alone would be comparable to the current 13.9 ms. |
| 4 | `hmm_parameters` 3.6–5.9 ms per Viterbi call | **none** (cache the constant) | It is a 360-state constant rebuilt identically every call — memoise it in Python (`functools.lru_cache` or a module-level constant) and never recompute. Not unroll/simd/parallel/gpu: 5.2 MB of one-off temporaries, already NumPy. If it must be computed, it is a one-time ~4 ms — still not a kernel to SIMD. |
| 5 | `local_cents` (argmax) 2 ms, **SLOWER** than NumPy | **simd** (unroll the 9-wide window) | 0.14 flop/B and 4.2 GB/s — it is streaming but scalar. Vectorise the 360-wide argmax scan and precompute the `mapping_at` table once instead of recomputing `j * (7180/359) + c` per window element (a per-element mul+add inside the hot loop). Parallel over rows is second; GPU: **no** — 0.14 flop/B ≪ 2 flop/B. |
| 6 | `row_max` 1.39 ms, **SLOWER** ~3.2x than NumPy | **simd** | 0.125 flop/B; the scalar NaN-branch per element is the whole story. A 4-wide SIMD `max` scan (with the same NaN-propagation semantics) is a drop-in. Parallel over rows second. GPU: **no** — 0.125 flop/B ≪ 2 flop/B. |
| 7 | `local_cents_centred` 62 µs | **none** | Already 10x faster than the NumPy gather; 135 k flop is latency-bound noise. Leave it. |
| 8 | ctypes floor ~5 µs/call | **none** (unroll/simd/parallel rewrite would not move it) | 5 µs is the `_lib.py` floor (address extraction + `ascontiguousarray` + `np.empty` + call). It only matters for tiny inputs and <10-frame predicts, where `hmm_parameters` caching (hotspot 4) already removes the dominant term. If a tiny-input phase ever runs, the fix is early-return in `_lib` for small `n`, not a kernel change. |

### The two rows where the port is slower than the baseline (kept, per instructions)

- `to_local_average_cents` (argmax centre, 3000x360): mojo 1.98 ms vs
  NumPy 1.21 ms in `pixi run bench` (0.61x). Both paths are memory-bound at
  0.14 flop/B; NumPy's gather argmax is vectorised and the Mojo loop is
  scalar with a per-element NaN branch and a recomputed `mapping_at` table.
  Next phase: **simd** (+ unroll the 9-wide window).
- `row_max` (3000x360): mojo 1.39 ms vs NumPy 0.44 ms (~3.2x). Same shape:
  0.125 flop/B, scalar NaN-branch scan vs NumPy's vectorised `max`.
  Next phase: **simd**. This is also why `predict_from_activation` **without**
  Viterbi is slower than NumPy (2.49 ms vs 1.35 ms) — its cost is
  `row_max` + `local_cents`, the two rows above.
- Incidental: `hmm_parameters` (3.6–5.9 ms) is slower than its NumPy
  equivalent (1.4 ms) because it takes `log` of `transition` and `log` of
  `emission` then transposes, through ~5.2 MB of Python temporaries, on
  every Viterbi call. Next phase: **none** (cache).

### GPU summary (required verdicts)

No kernel in this port clears the ~2 flop/B bar from `MOJO_NOTES.md` §5, so
**no GPU phase is recommended for any hotspot**:
viterbi 0.25, frame 0.19, cents_to_hz 1.06, hmm 1.9, local 0.14, row_max
0.125, local_centred 0.55 flop/B. The closest, `cents_to_hz` at 1.06 flop/B,
is `exp2`-compute-bound and should get SIMD, not a device hop. `hmm_parameters`
at 1.9 flop/B is a one-time constant, not a per-call kernel. If a future
phase makes `viterbi` compute-dense (e.g. exploiting the 25-wide transition
band to skip zero `logT` entries, which would lower bytes touched, not raise
flops/byte enough to matter here), re-measure flop/B before revisiting GPU.

## Constraints respected

No files under `src/` or `tests/` were changed; the only file added is this
`bench/analysis.md`. No algorithm was changed — timings, bytes and flops are
from this session's runs and from reading `src/kernels.mojo` /
`python/mojo_crepe/{core,_lib}.py`, plus `tracemalloc` peaks and the
`bench/bench.py` + reference timings quoted above.
