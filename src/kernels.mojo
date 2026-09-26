"""CREPE pitch-estimation DSP kernels.

Ported surface (everything in `crepe.core` that does not need the Keras CNN):

  * per-frame zero-mean / unit-std normalisation of the 1024-sample, 160-hop
    framing that `get_activation` builds with `as_strided`;
  * the 360-bin -> cents mapping and the 9-bin weighted local average that
    `to_local_average_cents` performs;
  * the 360-state multinomial Viterbi decode and its transition/emission
    construction from `to_viterbi_cents`;
  * the cents -> Hz map `10 * 2 ** (cents / 1200)` from `predict`, and the
    per-row `activation.max(axis=1)` confidence.

Every exported symbol takes buffer addresses as plain `Int` and rebuilds the
pointer inside the body: `@export` rejects parametric functions, and an
inferred pointer origin would make the symbol parametric.
"""

from std.math import exp2, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]

def mapping_at(j: Int) -> Float64:
    """`np.linspace(0, 7180, 360)[j] + 1997.3794084376191`, the bin-to-cents
    table of `to_local_average_cents`."""
    var step = Float64(7180.0) / Float64(359.0)
    return Float64(j) * step + Float64(1997.3794084376191)



def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


@export("crepe_frame_normalize")
def crepe_frame_normalize(
    audio_addr: Int, n_samples: Int, frame_len: Int, hop: Int, n_frames: Int,
    out_addr: Int
) abi("C"):
    """Per-frame normalisation of a strided framing of `audio`.

    `audio` holds `n_samples` samples; frame `t` is
    `audio[t*hop : t*hop + frame_len]`, i.e. exactly the window
    `get_activation` produces with `as_strided`. Each frame is written as
    `(x - mean) / std` with the population standard deviation (`ddof=0`), the
    same quantities `np.mean(frames, axis=1)` and `np.std(frames, axis=1)`
    produce. A constant frame (std == 0) is emitted as all-zero instead of the
    NaN numpy would produce; the divergence is documented in the README.

    Writes `n_frames * frame_len` doubles to `out` in frame order.
    """
    var x = fp(audio_addr)
    var dst = fp(out_addr)
    var fl = Float64(frame_len)
    for t in range(n_frames):
        var base = t * hop
        var wbase = t * frame_len
        var acc = Float64(0.0)
        for i in range(frame_len):
            acc += x[unsafe_offset=base + i]
        var mean = acc / fl
        var sq = Float64(0.0)
        for i in range(frame_len):
            var d = x[unsafe_offset=base + i] - mean
            sq = d * d + sq
        var sd = sqrt(sq / fl)
        if sd > Float64(0.0):
            for i in range(frame_len):
                dst[unsafe_offset=wbase + i] = (
                    x[unsafe_offset=base + i] - mean
                ) / sd
        else:
            for i in range(frame_len):
                dst[unsafe_offset=wbase + i] = Float64(0.0)


@export("crepe_local_cents")
def crepe_local_cents(
    act_addr: Int, n_rows: Int, n_bins: Int, half: Int, out_addr: Int
) abi("C"):
    """`to_local_average_cents` over a whole (rows, bins) salience matrix.

    For each row the centre bin is the first argmax (the tie-breaking rule of
    `np.argmax`); the window is `[centre - half, centre + half + 1)` clipped to
    the row, and the result is the salience-weighted mean of the bin-to-cents
    mapping over that window.
    """
    var act = fp(act_addr)
    var dst = fp(out_addr)
    for t in range(n_rows):
        var base = t * n_bins
        var center = 0
        var best = act[unsafe_offset=base]
        for j in range(1, n_bins):
            var v = act[unsafe_offset=base + j]
            if v > best:
                best = v
                center = j
        var start = center - half
        if start < 0:
            start = 0
        var end = center + half + 1
        if end > n_bins:
            end = n_bins
        var psum = Float64(0.0)
        var wsum = Float64(0.0)
        for j in range(start, end):
            psum += act[unsafe_offset=base + j] * mapping_at(j)
            wsum += act[unsafe_offset=base + j]
        dst[unsafe_offset=t] = psum / wsum


@export("crepe_local_cents_centred")
def crepe_local_cents_centred(
    act_addr: Int, centers_addr: Int, n_rows: Int, n_bins: Int, half: Int,
    out_addr: Int
) abi("C"):
    """`to_local_average_cents(salience[i], path[i])` for a supplied path.

    Identical to `crepe_local_cents` except the centre bin of each row is read
    from `centers` instead of being found with an argmax. This is the Viterbi
    branch of `to_viterbi_cents`.
    """
    var act = fp(act_addr)
    var centers = ip(centers_addr)
    var dst = fp(out_addr)
    for t in range(n_rows):
        var base = t * n_bins
        var center = Int(centers[unsafe_offset=t])
        var start = center - half
        if start < 0:
            start = 0
        var end = center + half + 1
        if end > n_bins:
            end = n_bins
        var psum = Float64(0.0)
        var wsum = Float64(0.0)
        for j in range(start, end):
            psum += act[unsafe_offset=base + j] * mapping_at(j)
            wsum += act[unsafe_offset=base + j]
        dst[unsafe_offset=t] = psum / wsum


@export("crepe_viterbi")
def crepe_viterbi(
    n_frames: Int, n_states: Int, obs_addr: Int, logtrans_addr: Int,
    logemis_addr: Int, startlog_addr: Int, delta_addr: Int, next_addr: Int,
    acc_addr: Int, arg_addr: Int, psi_addr: Int
) abi("C"):
    """Log-space max-product Viterbi decode of `obs`.

    `logtrans[i * n_states + j]` is `log P(j | i)` and
    `logemis[o * n_states + j]` is `log P(o | j)` transposed, so both matrices
    are read row-major in the inner loop. `obs[t]` is the observed bin of
    frame `t`. `acc` and `arg` are `n_states` scratch buffers; `psi` receives
    `n_states` Int32 backpointers per frame, row `t` at offset `t * n_states`.
    `delta` is seeded from `startlog`.

    The recurrence is sequential in `t` (frame `t` depends on frame `t - 1`),
    so this is a single serial kernel rather than a range-split one: no thread
    pool can overlap the frames.
    """
    var obs = ip(obs_addr)
    var logt = fp(logtrans_addr)
    var loge = fp(logemis_addr)
    var delta = fp(delta_addr)
    var next = fp(next_addr)
    var acc = fp(acc_addr)
    var arg = fp(arg_addr)
    var psi = ip(psi_addr)
    var sl = fp(startlog_addr)
    for j in range(n_states):
        delta[unsafe_offset=j] = sl[unsafe_offset=j]
    for t in range(n_frames):
        var o = Int(obs[unsafe_offset=t])
        for j in range(n_states):
            acc[unsafe_offset=j] = delta[unsafe_offset=0] + logt[unsafe_offset=j]
            arg[unsafe_offset=j] = 0.0
        for i in range(1, n_states):
            var d = delta[unsafe_offset=i]
            var row = i * n_states
            for j in range(n_states):
                var v = d + logt[unsafe_offset=row + j]
                if v > acc[unsafe_offset=j]:
                    acc[unsafe_offset=j] = v
                    arg[unsafe_offset=j] = Float64(i)
        var psibase = t * n_states
        var erow = o * n_states
        for j in range(n_states):
            var a = Int32(arg[unsafe_offset=j])
            psi[unsafe_offset=psibase + j] = a
            next[unsafe_offset=j] = acc[unsafe_offset=j] + loge[unsafe_offset=erow + j]
        for j in range(n_states):
            delta[unsafe_offset=j] = next[unsafe_offset=j]


@export("crepe_cents_to_hz")
def crepe_cents_to_hz(cents_addr: Int, n: Int, out_addr: Int) abi("C"):
    """`10 * 2 ** (cents / 1200)`, with NaN mapped to 0 as `predict` does."""
    var cents = fp(cents_addr)
    var dst = fp(out_addr)
    for i in range(n):
        var c = cents[unsafe_offset=i]
        if c != c:
            dst[unsafe_offset=i] = Float64(0.0)
        else:
            dst[unsafe_offset=i] = Float64(10.0) * exp2(c / Float64(1200.0))


@export("crepe_row_max")
def crepe_row_max(act_addr: Int, n_rows: Int, n_bins: Int, out_addr: Int) abi("C"):
    """`activation.max(axis=1)`, the confidence curve of `predict`."""
    var act = fp(act_addr)
    var dst = fp(out_addr)
    for t in range(n_rows):
        var base = t * n_bins
        var best = act[unsafe_offset=base]
        for j in range(1, n_bins):
            var v = act[unsafe_offset=base + j]
            if v > best:
                best = v
        dst[unsafe_offset=t] = best
