"""Zero-sync GPU per-request 4-gram suffix table & Triton kernels for universal Recall + propose_after.

Design goals:
- Zero GPU->CPU sync in decode / verify hot paths (fully compatible with SGLang overlap scheduling).
- Tiny fixed VRAM footprint (~0.95 MiB per active slot; default 8 slots = ~7.6 MiB total, lazily allocated).
- Lock-free parallel GPU hash table build (~25 us for 28K prompt) and O(1) incremental decode append (~3 us).
- Single ~6 us Triton kernel ``_suffix_propose_after_kernel`` chains suffix lookup AFTER neural draft tokens
  (Strata ``--lookup-chain`` / ``propose_after``), overwriting ONLY the matched tail ``d_p..d_{gamma-1}`` while
  keeping neural prefix ``d_0..d_{p-1}`` and their exact probabilities untouched (100% lossless).
- Dedicated CUDA-graph-capturable ``_dfly_recall_merge_kernel`` fuses CPU anchor Recall merge + GPU
  ``propose_after`` inside ``HYV4FlashDFly.draft_into_verify()`` with static buffer pointers.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

MAX_SLOTS = 8
HIST_CAP = 32768
HT_CAP = 65536
HT_MASK = HT_CAP - 1
SLIDE_MARGIN = 4096
PROBE_LIMIT = 16
GRAM = 6


@triton.jit
def _hash6_u64(t0, t1, t2, t3, t4, t5):
    a = t0.to(tl.uint64) & 0xFFFFFFFF
    b = t1.to(tl.uint64) & 0xFFFFFFFF
    c = t2.to(tl.uint64) & 0xFFFFFFFF
    d = t3.to(tl.uint64) & 0xFFFFFFFF
    e = t4.to(tl.uint64) & 0xFFFFFFFF
    f = t5.to(tl.uint64) & 0xFFFFFFFF
    h = (
        (a * 0x9E3779B97F4A7C15)
        ^ (b * 0xBF58476D1CE4E5B9)
        ^ (c * 0x94D049BB133111EB)
        ^ (d * 0x517CC1B727220A95)
        ^ (e * 0x2545F4914F6CDD1D)
        ^ (f * 0xD6E8FEB86659FD93)
    )
    h = h ^ (h >> 33)
    h = h * 0xFF51AFD7ED558CCD
    h = h ^ (h >> 33)
    h = h * 0xC4CEB9FE1A85EC53
    h = h ^ (h >> 33)
    return (h | 1).to(tl.int64)


@triton.jit
def _suffix_insert_kernel(
    HIST,
    HT_KEYS,
    HT_POS,
    start_pos,
    count,
    HIST_BASE: tl.constexpr,
    HT_BASE: tl.constexpr,
    HT_MASK_C: tl.constexpr,
    PROBE_C: tl.constexpr,
):
    """Insert 6-grams ending at [start_pos, start_pos + count) into open-addressed (HT_KEYS, HT_POS).

    6-gram keys avoid collapsing distinct code lines that share generic 4-token prefixes (e.g.
    `\\n        if `), and atomic_cas + atomic_max keep parallel builds race-free.
    """
    tid = tl.program_id(0)
    end = start_pos + tid
    if tid >= count or end < 5:
        return
    h_ptr = HIST + HIST_BASE
    t0 = tl.load(h_ptr + end - 5)
    t1 = tl.load(h_ptr + end - 4)
    t2 = tl.load(h_ptr + end - 3)
    t3 = tl.load(h_ptr + end - 2)
    t4 = tl.load(h_ptr + end - 1)
    t5 = tl.load(h_ptr + end)
    key = _hash6_u64(t0, t1, t2, t3, t4, t5)
    slot0 = (key.to(tl.uint64) & HT_MASK_C).to(tl.int32)
    k_ptr = HT_KEYS + HT_BASE
    p_ptr = HT_POS + HT_BASE
    done = 0
    zero_i64 = tl.full((), 0, tl.int64)
    end_i32 = end.to(tl.int32)
    for j in range(PROBE_C):
        if done == 0:
            idx = (slot0 + j) & HT_MASK_C
            old = tl.atomic_cas(k_ptr + idx, zero_i64, key)
            if old == 0 or old == key:
                tl.atomic_max(p_ptr + idx, end_i32)
                done = 1


@triton.jit
def _read_rel_tok(
    h_ptr,
    hlen,
    lag,
    anchor,
    d_ptr,
    rel,
):
    """Read token at offset ``rel`` relative to ``anchor`` (0=anchor, >0=draft[rel-1], <0=hist)."""
    if rel > 0:
        return tl.load(d_ptr + (rel - 1)).to(tl.int32)
    if rel == 0:
        return anchor.to(tl.int32)
    if lag < 0 or lag > 1:
        return tl.full((), -1, tl.int32)
    h_idx = (hlen - 1 + rel) if lag == 0 else (hlen + rel)
    if h_idx >= 0 and h_idx < hlen:
        return tl.load(h_ptr + h_idx).to(tl.int32)
    return tl.full((), -1, tl.int32)


@triton.jit
def _probe_and_chain_one(
    b,
    anchor,
    seq_len,
    d_ptr,
    ow_ptr,
    conf_ptr,
    SLOT_IDS,
    HIST_LENS,
    BASE_LENS,
    HIST,
    HT_KEYS,
    HT_POS,
    gamma,
    min_match,
    min_conf,
    HIST_CAP_C: tl.constexpr,
    HT_CAP_C: tl.constexpr,
    HT_MASK_C: tl.constexpr,
    PROBE_C: tl.constexpr,
    HAS_OW: tl.constexpr,
    HAS_CONF: tl.constexpr,
):
    """Run zero-sync GPU suffix propose_after for request b; returns 1 if any draft token was overwritten."""
    zero_i32 = tl.full((), 0, tl.int32)
    one_i32 = tl.full((), 1, tl.int32)
    slot = tl.load(SLOT_IDS + b).to(tl.int32)
    if slot < 0:
        return zero_i32
    hlen = tl.load(HIST_LENS + b).to(tl.int32)
    if hlen < 7:
        return zero_i32
    blen = tl.load(BASE_LENS + b).to(tl.int64)
    lag = (seq_len - blen + 1).to(tl.int32)
    h_ptr = HIST + slot.to(tl.int64) * HIST_CAP_C
    if lag == 0:
        if tl.load(h_ptr + hlen - 1).to(tl.int32) != anchor:
            lag = tl.full((), 2, tl.int32)
    p_start = tl.where((lag == 0) | (lag == 1), zero_i32, tl.full((), 5, tl.int32))
    if p_start >= gamma:
        return zero_i32
    k_ptr = HT_KEYS + slot.to(tl.int64) * HT_CAP_C
    p_ptr = HT_POS + slot.to(tl.int64) * HT_CAP_C

    surv = tl.full((), 1.0, tl.float32)
    matched = zero_i32
    for p in range(0, gamma):
        if matched == 0:
            if HAS_CONF:
                if p > 0:
                    surv = surv * tl.load(conf_ptr + (p - 1)).to(tl.float32)
            if p >= p_start and (p == 0 or (not HAS_CONF) or surv >= min_conf):
                t5 = _read_rel_tok(h_ptr, hlen, lag, anchor, d_ptr, p)
                t4 = _read_rel_tok(h_ptr, hlen, lag, anchor, d_ptr, p - 1)
                t3 = _read_rel_tok(h_ptr, hlen, lag, anchor, d_ptr, p - 2)
                t2 = _read_rel_tok(h_ptr, hlen, lag, anchor, d_ptr, p - 3)
                t1 = _read_rel_tok(h_ptr, hlen, lag, anchor, d_ptr, p - 4)
                t0 = _read_rel_tok(h_ptr, hlen, lag, anchor, d_ptr, p - 5)
                if t0 >= 0 and t1 >= 0 and t2 >= 0 and t3 >= 0 and t4 >= 0 and t5 >= 0:
                    key = _hash6_u64(t0, t1, t2, t3, t4, t5)
                    slot0 = (key.to(tl.uint64) & HT_MASK_C).to(tl.int32)
                    found_pos = tl.full((), -1, tl.int32)
                    stop_probe = zero_i32
                    for j in range(PROBE_C):
                        if stop_probe == 0:
                            idx = (slot0 + j) & HT_MASK_C
                            kv = tl.load(k_ptr + idx)
                            if kv == 0:
                                stop_probe = one_i32
                            elif kv == key:
                                pos = tl.load(p_ptr + idx).to(tl.int32)
                                max_pos = tl.where((lag == 0) & (p == 0), hlen - 2, hlen - 1)
                                if pos >= 5 and pos < max_pos:
                                    if (
                                        tl.load(h_ptr + pos) == t5
                                        and tl.load(h_ptr + pos - 1) == t4
                                        and tl.load(h_ptr + pos - 2) == t3
                                        and tl.load(h_ptr + pos - 3) == t2
                                        and tl.load(h_ptr + pos - 4) == t1
                                        and tl.load(h_ptr + pos - 5) == t0
                                    ):
                                        found_pos = pos
                                stop_probe = one_i32
                    if found_pos >= 0:
                        ml = tl.full((), 6, tl.int32)
                        ext_ok = one_i32
                        for step in range(6, 16):
                            if ext_ok != 0 and found_pos >= step:
                                rtok = _read_rel_tok(h_ptr, hlen, lag, anchor, d_ptr, p - step)
                                if rtok >= 0 and tl.load(h_ptr + found_pos - step) == rtok:
                                    ml = ml + 1
                                else:
                                    ext_ok = zero_i32
                        # Require >= 8 tokens before replacing neural d_0 (p == 0);
                        # tail chaining (p >= 1) activates at min_match (>= 6).
                        eight_i32 = tl.full((), 8, tl.int32)
                        req_ml = tl.where(p == 0, tl.maximum(min_match, eight_i32), min_match)
                        # On confidence-equipped drafters (DSpark/DFly2), do not clobber already
                        # >=0.90-confident neural predictions unless we have a strong >=10-token match.
                        conf_ok = one_i32
                        if HAS_CONF:
                            if ml < 10:
                                if tl.load(conf_ptr + p).to(tl.float32) >= 0.90:
                                    conf_ok = zero_i32
                        if ml >= req_ml and conf_ok != 0:
                            rem = (gamma - p).to(tl.int32)
                            avail = (hlen - 1 - found_pos).to(tl.int32)
                            cap_k = tl.minimum(rem, avail)
                            if ml < 16:
                                cap_k = tl.minimum(cap_k, 2 * ml)
                            if cap_k >= 1:
                                for k in range(0, cap_k):
                                    ntok = tl.load(h_ptr + found_pos + 1 + k)
                                    tl.store(d_ptr + p + k, ntok.to(d_ptr.dtype.element_ty))
                                    if HAS_OW:
                                        tl.store(ow_ptr + p + k, tl.full((), 1, ow_ptr.dtype.element_ty))
                                    if HAS_CONF:
                                        tl.store(conf_ptr + p + k, tl.full((), 1.0, conf_ptr.dtype.element_ty))
                                matched = one_i32
    return matched


@triton.jit
def _suffix_propose_after_kernel(
    SKIP_MASK,
    SEQ_LENS,
    ANCHORS,
    DRAFT_TOKENS,
    OVERWRITE,
    CONF,
    SLOT_IDS,
    HIST_LENS,
    BASE_LENS,
    HIST,
    HT_KEYS,
    HT_POS,
    bs,
    gamma,
    seq_stride,
    anchor_stride,
    draft_stride,
    ow_stride,
    conf_stride,
    min_match,
    min_conf,
    HIST_CAP_C: tl.constexpr,
    HT_CAP_C: tl.constexpr,
    HT_MASK_C: tl.constexpr,
    PROBE_C: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    HAS_OW: tl.constexpr,
    HAS_CONF: tl.constexpr,
):
    b = tl.program_id(0)
    if b >= bs:
        return
    if HAS_SKIP:
        if tl.load(SKIP_MASK + b) != 0:
            return
    anchor = tl.load(ANCHORS + b.to(tl.int64) * anchor_stride).to(tl.int32)
    seq_len = tl.load(SEQ_LENS + b.to(tl.int64) * seq_stride).to(tl.int64)
    d_ptr = DRAFT_TOKENS + b.to(tl.int64) * draft_stride
    ow_ptr = OVERWRITE + b.to(tl.int64) * ow_stride
    conf_ptr = CONF + b.to(tl.int64) * conf_stride
    _probe_and_chain_one(
        b,
        anchor,
        seq_len,
        d_ptr,
        ow_ptr,
        conf_ptr,
        SLOT_IDS,
        HIST_LENS,
        BASE_LENS,
        HIST,
        HT_KEYS,
        HT_POS,
        gamma,
        min_match,
        min_conf,
        HIST_CAP_C=HIST_CAP_C,
        HT_CAP_C=HT_CAP_C,
        HT_MASK_C=HT_MASK_C,
        PROBE_C=PROBE_C,
        HAS_OW=HAS_OW,
        HAS_CONF=HAS_CONF,
    )


@triton.jit
def _dfly_recall_merge_kernel(
    BLOCK_IDS,
    POSITIONS,
    PLAN,
    CONF,
    CFG,
    SLOT_IDS,
    HIST_LENS,
    BASE_LENS,
    HIST,
    HT_KEYS,
    HT_POS,
    bs,
    width,
    plan_stride,
    conf_stride,
    min_conf,
    HIST_CAP_C: tl.constexpr,
    HT_CAP_C: tl.constexpr,
    HT_MASK_C: tl.constexpr,
    PROBE_C: tl.constexpr,
    HAS_CONF: tl.constexpr,
    ENABLE_CHAIN: tl.constexpr,
):
    """CUDA-graph-capturable Recall anchor merge + GPU propose_after inside HYV4FlashDFly.draft_into_verify."""
    b = tl.program_id(0)
    if b >= bs:
        return
    gamma = width - 1
    row_ptr = BLOCK_IDS + b.to(tl.int64) * width
    anchor = tl.load(row_ptr).to(tl.int32)
    seq_len = tl.load(POSITIONS + b.to(tl.int64) * width).to(tl.int64)
    p_row = PLAN + b.to(tl.int64) * plan_stride
    real_cnt = tl.load(p_row)
    anchor_full = tl.full((), 0, tl.int32)
    if real_cnt > 0:
        L = tl.load(p_row + 1)
        o = seq_len - L + 1
        om = gamma + 1
        if o >= 0 and o <= om:
            if tl.load(p_row + 2 + o).to(tl.int32) == anchor:
                avail_real = (real_cnt - o).to(tl.int32)
                n_fill = tl.minimum(gamma, avail_real)
                if n_fill > 0:
                    for k in range(0, n_fill):
                        ntok = tl.load(p_row + 2 + o + 1 + k)
                        tl.store(row_ptr + 1 + k, ntok.to(BLOCK_IDS.dtype.element_ty))
                        if HAS_CONF:
                            tl.store(
                                CONF + b.to(tl.int64) * conf_stride + k,
                                tl.full((), 1.0, CONF.dtype.element_ty),
                            )
                    if n_fill >= gamma:
                        anchor_full = tl.full((), 1, tl.int32)
    if ENABLE_CHAIN:
        if anchor_full == 0:
            min_match = tl.load(CFG).to(tl.int32)
            _probe_and_chain_one(
                b,
                anchor,
                seq_len,
                row_ptr + 1,
                row_ptr,  # unused when HAS_OW=False
                CONF + b.to(tl.int64) * conf_stride,
                SLOT_IDS,
                HIST_LENS,
                BASE_LENS,
                HIST,
                HT_KEYS,
                HT_POS,
                gamma,
                min_match,
                min_conf,
                HIST_CAP_C=HIST_CAP_C,
                HT_CAP_C=HT_CAP_C,
                HT_MASK_C=HT_MASK_C,
                PROBE_C=PROBE_C,
                HAS_OW=False,
                HAS_CONF=HAS_CONF,
            )


@triton.jit
def _onehot_masked_kernel(
    TENSOR,
    OVERWRITE,
    TOKENS,
    stride_b,
    stride_g,
    ow_stride_b,
    tok_stride_b,
    V,
    FILL_VAL: tl.constexpr,
    HOT_VAL: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Set TENSOR[b, j, :] = FILL_VAL and TENSOR[b, j, TOKENS[b, j]] = HOT_VAL ONLY when OVERWRITE[b, j] != 0."""
    b = tl.program_id(0)
    j = tl.program_id(1)
    if tl.load(OVERWRITE + b * ow_stride_b + j) == 0:
        return
    tok = tl.load(TOKENS + b * tok_stride_b + j).to(tl.int64)
    tok = tl.maximum(0, tl.minimum(tok, V - 1))
    row = TENSOR + b.to(tl.int64) * stride_b + j.to(tl.int64) * stride_g
    fill_vec = tl.full([BLOCK_V], FILL_VAL, TENSOR.dtype.element_ty)
    hot_vec = tl.full([BLOCK_V], HOT_VAL, TENSOR.dtype.element_ty)
    for v in range(0, V, BLOCK_V):
        offs = v + tl.arange(0, BLOCK_V)
        m = offs < V
        vals = tl.where(offs == tok, hot_vec, fill_vec)
        tl.store(row + offs, vals, mask=m)


def onehot_masked(
    tensor: Optional[torch.Tensor],
    overwrite: torch.Tensor,
    tokens: torch.Tensor,
    *,
    is_probs: bool,
) -> None:
    """One-hot ONLY overwritten (b, j) rows of draft_probs (0/1) or corrected_logits (-inf/0)."""
    if tensor is None or tensor.dim() != 3:
        return
    bs, g = int(tokens.shape[0]), int(tokens.shape[1])
    if bs <= 0 or g <= 0 or tensor.shape[0] < bs or tensor.shape[1] < g:
        return
    V = int(tensor.shape[2])
    fill_val = 0.0 if is_probs else float("-inf")
    hot_val = 1.0 if is_probs else 0.0
    _onehot_masked_kernel[(bs, g)](
        tensor,
        overwrite,
        tokens,
        tensor.stride(0),
        tensor.stride(1),
        overwrite.stride(0),
        tokens.stride(0),
        V,
        FILL_VAL=fill_val,
        HOT_VAL=hot_val,
        BLOCK_V=4096,
    )


class GpuSuffixPool:
    """Pooled fixed-address GPU 4-gram suffix tables for active requests (~0.95 MiB/slot)."""

    def __init__(self, device: torch.device, max_slots: int = MAX_SLOTS) -> None:
        self.device = torch.device(device)
        self.max_slots = int(max_slots)
        self.hist = torch.zeros((self.max_slots, HIST_CAP), dtype=torch.int32, device=self.device)
        self.ht_keys = torch.zeros((self.max_slots, HT_CAP), dtype=torch.int64, device=self.device)
        self.ht_pos = torch.full((self.max_slots, HT_CAP), -1, dtype=torch.int32, device=self.device)
        self._rid_to_slot: Dict[object, int] = {}
        self._slot_to_rid: list[Optional[object]] = [None] * self.max_slots
        self._slot_hlen: list[int] = [0] * self.max_slots
        self._slot_blen: list[int] = [0] * self.max_slots
        self._slot_out_len: list[int] = [0] * self.max_slots
        self._lru_tick = 0
        self._slot_tick: list[int] = [0] * self.max_slots
        # Race-free pinned host staging:
        # - Per-slot full buffer for initial prompt rebuild (max_slots x 128 KiB = 1 MiB)
        # - 8-deep ring for per-step incremental decode appends & batch metadata (no in-flight DMA clobber)
        self._ring = 8
        self._ring_idx = 0
        self._host_hist = torch.empty((self.max_slots, HIST_CAP), dtype=torch.int32, pin_memory=True)
        self._host_inc_cap = 256
        self._host_inc = torch.empty((self._ring, self.max_slots, self._host_inc_cap), dtype=torch.int32, pin_memory=True)
        # Static per-batch metadata buffers (max_bs = 64) for CUDA-graph safety
        self.slot_ids = torch.full((64,), -1, dtype=torch.int32, device=self.device)
        self.hist_lens = torch.zeros((64,), dtype=torch.int32, device=self.device)
        self.base_lens = torch.zeros((64,), dtype=torch.int64, device=self.device)
        self._host_meta_slots = torch.full((self._ring, 64), -1, dtype=torch.int32, pin_memory=True)
        self._host_meta_hlens = torch.zeros((self._ring, 64), dtype=torch.int32, pin_memory=True)
        self._host_meta_blens = torch.zeros((self._ring, 64), dtype=torch.int64, pin_memory=True)

    def _alloc_slot(self, rid: object) -> int:
        slot = self._rid_to_slot.get(rid)
        if slot is not None:
            self._lru_tick += 1
            self._slot_tick[slot] = self._lru_tick
            return slot
        for s in range(self.max_slots):
            if self._slot_to_rid[s] is None:
                slot = s
                break
        if slot is None:
            slot = min(range(self.max_slots), key=lambda s: self._slot_tick[s])
            old_rid = self._slot_to_rid[slot]
            if old_rid is not None:
                self._rid_to_slot.pop(old_rid, None)
        self._rid_to_slot[rid] = slot
        self._slot_to_rid[slot] = rid
        self._lru_tick += 1
        self._slot_tick[slot] = self._lru_tick
        return slot

    def release(self, rid: object) -> None:
        slot = self._rid_to_slot.pop(rid, None)
        if slot is not None:
            self._slot_to_rid[slot] = None
            self._slot_hlen[slot] = 0
            self._slot_blen[slot] = 0
            self._slot_out_len[slot] = 0

    def _rebuild_slot(self, slot: int, tokens: np.ndarray, total_len: int, out_len: int) -> None:
        keep_max = HIST_CAP - SLIDE_MARGIN
        arr = np.asarray(tokens, dtype=np.int32).ravel()
        if arr.size > keep_max:
            arr = arr[-keep_max:]
        n = int(arr.size)
        self.ht_keys[slot].zero_()
        self.ht_pos[slot].fill_(-1)
        if n > 0:
            self._host_hist[slot, :n].copy_(torch.from_numpy(arr))
            self.hist[slot, :n].copy_(self._host_hist[slot, :n], non_blocking=True)
            # Only insert 6-grams ending at <= n - 2 (every inserted key has >= 1 continuation token
            # and never self-clobbers an earlier occurrence of the current tail in single-slot HT_POS).
            if n >= GRAM + 1:
                count = n - GRAM
                _suffix_insert_kernel[(count,)](
                    self.hist,
                    self.ht_keys,
                    self.ht_pos,
                    GRAM - 1,
                    count,
                    HIST_BASE=slot * HIST_CAP,
                    HT_BASE=slot * HT_CAP,
                    HT_MASK_C=HT_MASK,
                    PROBE_C=PROBE_LIMIT,
                    num_warps=1,
                )
        self._slot_hlen[slot] = n
        self._slot_blen[slot] = int(total_len)
        self._slot_out_len[slot] = int(out_len)

    def sync_request(self, rid: object, origin_ids, output_ids) -> int:
        """Incrementally sync request ``rid`` into its GPU suffix slot (O(1) per decode step)."""
        n_orig = len(origin_ids)
        n_out = len(output_ids)
        total = n_orig + n_out
        if total < GRAM + 1:
            return -1
        slot = self._rid_to_slot.get(rid)
        if slot is None or self._slot_out_len[slot] > n_out or self._slot_blen[slot] > total:
            slot = self._alloc_slot(rid)
            keep_max = HIST_CAP - SLIDE_MARGIN
            if n_out >= keep_max:
                tail = np.asarray(output_ids[-keep_max:], dtype=np.int32)
            elif total <= keep_max:
                if n_out:
                    tail = np.concatenate((
                        np.asarray(origin_ids, dtype=np.int32),
                        np.asarray(output_ids, dtype=np.int32),
                    ))
                else:
                    tail = np.asarray(origin_ids, dtype=np.int32)
            else:
                need_orig = keep_max - n_out
                if n_out:
                    tail = np.concatenate((
                        np.asarray(origin_ids[-need_orig:], dtype=np.int32),
                        np.asarray(output_ids, dtype=np.int32),
                    ))
                else:
                    tail = np.asarray(origin_ids[-keep_max:], dtype=np.int32)
            self._rebuild_slot(slot, tail, total, n_out)
            return slot

        self._lru_tick += 1
        self._slot_tick[slot] = self._lru_tick
        prev_out = self._slot_out_len[slot]
        if n_out <= prev_out:
            return slot
        delta = n_out - prev_out
        hlen = self._slot_hlen[slot]
        if hlen + delta > HIST_CAP:
            keep_max = HIST_CAP - SLIDE_MARGIN
            need_orig = max(0, keep_max - n_out)
            if need_orig > 0:
                tail = np.concatenate((
                    np.asarray(origin_ids[-need_orig:], dtype=np.int32),
                    np.asarray(output_ids, dtype=np.int32),
                ))
            else:
                tail = np.asarray(output_ids[-keep_max:], dtype=np.int32)
            self._rebuild_slot(slot, tail, total, n_out)
            return slot

        new_arr = np.asarray(output_ids[prev_out:n_out], dtype=np.int32)
        if delta <= self._host_inc_cap:
            r = self._ring_idx
            self._host_inc[r, slot, :delta].copy_(torch.from_numpy(new_arr))
            self.hist[slot, hlen:hlen + delta].copy_(self._host_inc[r, slot, :delta], non_blocking=True)
        else:
            self._host_hist[slot, :delta].copy_(torch.from_numpy(new_arr))
            self.hist[slot, hlen:hlen + delta].copy_(self._host_hist[slot, :delta], non_blocking=True)
        new_hlen = hlen + delta
        ins_start = max(GRAM - 1, hlen - 1)
        ins_count = (new_hlen - 1) - ins_start
        if ins_count > 0:
            _suffix_insert_kernel[(ins_count,)](
                self.hist,
                self.ht_keys,
                self.ht_pos,
                ins_start,
                ins_count,
                HIST_BASE=slot * HIST_CAP,
                HT_BASE=slot * HT_CAP,
                HT_MASK_C=HT_MASK,
                PROBE_C=PROBE_LIMIT,
                num_warps=1,
            )
        self._slot_hlen[slot] = new_hlen
        self._slot_blen[slot] = total
        self._slot_out_len[slot] = n_out
        return slot

    def sync_batch(self, reqs: Sequence) -> bool:
        """Sync active batch requests and update static GPU metadata buffers. Returns True if any slot is active."""
        bs = min(len(reqs), 64)
        r = self._ring_idx
        self._ring_idx = (r + 1) % self._ring
        m_slots = self._host_meta_slots[r]
        m_hlens = self._host_meta_hlens[r]
        m_blens = self._host_meta_blens[r]
        any_slot = False
        for i in range(bs):
            req = reqs[i]
            if req.finished():
                m_slots[i] = -1
                m_hlens[i] = 0
                m_blens[i] = 0
                continue
            slot = self.sync_request(req.rid, req.origin_input_ids, req.output_ids)
            if slot >= 0:
                m_slots[i] = slot
                m_hlens[i] = self._slot_hlen[slot]
                m_blens[i] = self._slot_blen[slot]
                any_slot = True
            else:
                m_slots[i] = -1
                m_hlens[i] = 0
                m_blens[i] = 0
        if bs > 0:
            self.slot_ids[:bs].copy_(m_slots[:bs], non_blocking=True)
            self.hist_lens[:bs].copy_(m_hlens[:bs], non_blocking=True)
            self.base_lens[:bs].copy_(m_blens[:bs], non_blocking=True)
        return any_slot

    def disable_batch(self, bs: int) -> None:
        bs = min(int(bs), 64)
        if bs > 0:
            self.slot_ids[:bs].fill_(-1)

    def propose_after(
        self,
        *,
        bs: int,
        gamma: int,
        seq_lens: torch.Tensor,
        anchors: torch.Tensor,
        draft_tokens: torch.Tensor,
        skip_mask: Optional[torch.Tensor] = None,
        overwrite: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        min_match: int = 4,
        min_conf: float = 0.35,
    ) -> None:
        """Launch zero-sync GPU propose_after over draft_tokens[bs, gamma]."""
        if bs <= 0 or gamma <= 0:
            return
        has_skip = skip_mask is not None
        has_ow = overwrite is not None
        has_conf = confidence is not None
        _suffix_propose_after_kernel[(bs,)](
            skip_mask if has_skip else self.slot_ids,
            seq_lens,
            anchors,
            draft_tokens,
            overwrite if has_ow else self.slot_ids,
            confidence if has_conf else self.slot_ids,
            self.slot_ids,
            self.hist_lens,
            self.base_lens,
            self.hist,
            self.ht_keys,
            self.ht_pos,
            int(bs),
            int(gamma),
            int(seq_lens.stride(0)),
            int(anchors.stride(0)),
            int(draft_tokens.stride(0)),
            int(overwrite.stride(0)) if has_ow else 1,
            int(confidence.stride(0)) if has_conf else 1,
            max(GRAM, int(min_match)),
            float(min_conf),
            HIST_CAP_C=HIST_CAP,
            HT_CAP_C=HT_CAP,
            HT_MASK_C=HT_MASK,
            PROBE_C=PROBE_LIMIT,
            HAS_SKIP=has_skip,
            HAS_OW=has_ow,
            HAS_CONF=has_conf,
            num_warps=1,
        )
