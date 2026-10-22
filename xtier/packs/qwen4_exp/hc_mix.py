"""Hyper-connection low-rank mix of decode-size batches, overlapped (replaces SGLang's ``fused_hc_mix``).

SGLang's persistent kernel (``sglang.kernels.ops.gemm.hc_mix``) zeroes its FP32 accumulator, waits at a grid barrier,
accumulates ``x @ W_down^T`` with device atomics, waits again, then computes the gated up projection. This kernel
keeps DRAM streaming where that one waits (RTX PRO 5000, 110 SMs, Qwen3.8 shapes: 16.3 -> 13.4 us per call at 4
rows; -1.7% per speculative step in production since 2026-10-22):

* no zeroing pass and no first barrier: the accumulator is a persistent pair, ``state[3]`` names the half that is zero
  on entry, and each call zeroes the other half after its phase B for the next call;
* each CTA starts a TMA bulk prefetch (sm_90+) of its own phase-B block of ``W_up`` into L2 as soon as its phase-A
  tiles are issued, so ``W_up`` streams while the slower CTAs finish phase A and while everyone waits at the barrier;
* phase-A tiles walk along K, so the CTAs of one round read contiguous rows of ``W_down``;
* the barrier counter alternates with the parity and CTA 0 clears the other one once every CTA has arrived (no
  end-of-kernel ticket); CTAs without a phase-B block arrive and leave.
Same arithmetic as SGLang's kernel; the summation order of the atomics varies across replays in both. Ported from the
lsglang_q3.8fn patch (``hc_mix_triton.py``, 2026-10-22). XTIER_HC_MIX_OVERLAP=0 keeps SGLang's kernel.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _bulk_prefetch_l2(ptrs, nbytes, pred):
    # One TMA bulk prefetch into L2 per element whose pred is nonzero.
    tl.inline_asm_elementwise(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $3, 0;\n\t"
        "@p cp.async.bulk.prefetch.L2.global [$1], $2;\n\tmov.u32 $0, 0;\n\t}",
        "=r,l,r,r",
        [ptrs, nbytes, pred],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _hc_mix_overlap_kernel(x_ptr, w_down_ptr, w_up_ptr, acc_pair_ptr, out_ptr, state_ptr, K, LOWRANK, HS, num_rows,
                           num_ctas, inv_hc, ROWS: tl.constexpr, HC: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_R: tl.constexpr,
                           PF_BYTES: tl.constexpr, PF_LANES: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    span = ROWS * LOWRANK
    parity = tl.load(state_ptr + 3)
    acc_ptr = acc_pair_ptr + parity * span
    spare_ptr = acc_pair_ptr + (1 - parity) * span
    j_blocks = tl.cdiv(HS, BLOCK_J)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    for tile in range(pid, tl.cdiv(LOWRANK, BLOCK_N) * k_chunks, num_ctas):
        n = (tile // k_chunks) * BLOCK_N + offs_n
        k = (tile % k_chunks) * BLOCK_K + offs_k
        mask_n = n < LOWRANK
        xt = tl.load(x_ptr + offs_m[:, None] * K + k[None, :], mask=mask_m[:, None], other=0.0)
        w = tl.load(w_down_ptr + n[:, None] * K + k[None, :], mask=mask_n[:, None], other=0.0)
        tl.atomic_add(acc_ptr + offs_m[:, None] * LOWRANK + n[None, :], tl.dot(xt, tl.trans(w)),
                      mask=mask_m[:, None] & mask_n[None, :], sem="relaxed", scope="gpu")

    # The HC row groups of this CTA's first phase-B block are contiguous runs of BLOCK_J * LOWRANK elements; one
    # PF_BYTES request per lane.
    lane = tl.arange(0, PF_LANES)
    per_group = BLOCK_J * LOWRANK * 2 // PF_BYTES
    _bulk_prefetch_l2(
        w_up_ptr + ((lane // per_group) * HS + pid * BLOCK_J) * LOWRANK + (lane % per_group) * (PF_BYTES // 2),
        tl.full((PF_LANES,), PF_BYTES, tl.int32),
        ((lane < HC * per_group) & (pid < j_blocks)).to(tl.int32),
    )

    # Join the warps so every atomic is issued before the arrival.
    tl.debug_barrier()
    bar_ptr = state_ptr + parity
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    if pid < j_blocks:
        while tl.atomic_add(bar_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
            pass
    if pid == 0:
        # Every CTA has arrived, so every CTA has read the parity.
        tl.store(state_ptr + (1 - parity), 0)
        tl.store(state_ptr + 3, 1 - parity)

    offs_j = tl.arange(0, BLOCK_J)
    offs_r = tl.arange(0, BLOCK_R)
    offs_g = tl.arange(0, HC)
    for jb in range(pid, j_blocks, num_ctas):
        j = jb * BLOCK_J + offs_j
        mask_j = j < HS
        gj = offs_g[:, None] * HS + j[None, :]
        gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
        mask_gj = tl.reshape(tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,))
        acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
        for r0 in range(0, LOWRANK, BLOCK_R):
            r = r0 + offs_r
            mask_r = r < LOWRANK
            a = tl.load(acc_ptr + offs_m[:, None] * LOWRANK + r[None, :], mask=mask_r[None, :], other=0.0)
            a = a * inv_hc
            t = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
            w = tl.load(w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :], mask=mask_gj[:, None] & mask_r[None, :],
                        other=0.0)
            acc = tl.dot(t, tl.trans(w), acc)
        gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
        xg = tl.load(x_ptr + offs_m[:, None, None] * (HC * HS) + offs_g[None, :, None] * HS + j[None, None, :],
                     mask=mask_m[:, None, None] & mask_j[None, None, :], other=0.0).to(tl.float32)
        out = tl.sum(gate * xg, axis=1) * inv_hc
        tl.store(out_ptr + offs_m[:, None] * HS + j[None, :], out.to(out_ptr.dtype.element_ty),
                 mask=mask_m[:, None] & mask_j[None, :])

    # Nothing reads the spare accumulator in this call.
    offs_z = tl.arange(0, 256)
    for z0 in range(pid * 256, span, num_ctas * 256):
        idx = z0 + offs_z
        tl.store(spare_ptr + idx, 0.0, mask=idx < span)


ROWS_PAD = 16
_state = {}
_sms = {}


def _overlap_state(device, rows, lowrank):
    """Barrier counter per parity, unused, parity; and the accumulator pair. Zeroed here; every call leaves the half
    named by the parity zeroed and that parity's barrier counter cleared."""
    key = (device, rows, lowrank)
    entry = _state.get(key)
    if entry is None:
        entry = _state[key] = (torch.zeros(4, dtype=torch.int32, device=device),
                               torch.zeros((2, rows, lowrank), dtype=torch.float32, device=device))
    return entry


def _prefetch_bytes(block_j, lowrank):
    """Largest power of two up to 4 KB dividing one row group of a block."""
    group = block_j * lowrank * 2
    return min(4096, group & -group)


def enabled():
    # the TMA bulk prefetch (cp.async.bulk) needs sm_90 or newer
    return os.environ.get('XTIER_HC_MIX_OVERLAP', '1') == '1' and torch.cuda.get_device_capability()[0] >= 9


def fused_hc_mix(original, hyper_input_normed, w_down, w_up, hc, hs):
    """AROUND hook of sglang.kernels.ops.gemm.hc_mix.fused_hc_mix (same arguments and result)."""
    if not enabled():
        return original(hyper_input_normed, w_down, w_up, hc, hs)
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    device = hyper_input_normed.device
    out = torch.empty((rows, hs), dtype=hyper_input_normed.dtype, device=device)
    if rows == 0:
        return out
    state, acc_pair = _overlap_state(device, ROWS_PAD, lowrank)
    num_ctas = _sms.get(device)
    if num_ctas is None:   # one CTA per SM keeps every CTA resident (the grid barrier cannot deadlock)
        num_ctas = _sms[device] = torch.cuda.get_device_properties(device).multi_processor_count
    _hc_mix_overlap_kernel[(num_ctas,)](hyper_input_normed, w_down, w_up, acc_pair, out, state, k, lowrank, hs, rows,
                                        num_ctas, 1.0 / hc, ROWS=ROWS_PAD, HC=hc, BLOCK_N=32, BLOCK_K=256, BLOCK_J=32,
                                        BLOCK_R=64, PF_BYTES=_prefetch_bytes(32, lowrank), PF_LANES=256, num_warps=8)
    return out
