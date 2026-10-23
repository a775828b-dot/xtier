"""QSA indexer prefill logits with the keys split over CTAs (replaces ``qsa.mqa.tilelang_qsa_mqa_prefill``).

SGLang's TileLang kernel gives each CTA a block of query rows and loops over all compressed keys of that block. The
indexer calls it per row chunk of a 128 MiB FP32 logits budget, so deep in a long prefill (95K compressed keys at
380K context) one launch covers ~350 rows: 11 CTAs on a 110-SM GPU, ~47 launches per 16K prefill chunk, 1.7 s per
chunk in this kernel alone. Here the grid also splits the keys (XTIER_QSA_KEYS_PER_CTA, default 4096) so a launch
has hundreds of CTAs. Every logit is still one tile GEMM over head_dim of the same operands, the same ReLU and head
sum, so the values are SGLang's; a CTA stores only its own key range. The logits buffer is not zero-filled first: every
column outside a row's [start, end) is set to -inf by the mask kernel afterwards, and every column inside is stored.
Scaling and masking stay SGLang's.
"""

import math
import os

import torch

KEYS_PER_CTA = int(os.environ.get('XTIER_QSA_KEYS_PER_CTA', '4096'))
_kernels = {}


def _kernel(heads, head_dim, block_q, dtype, keys_per_cta):
    key = (heads, head_dim, block_q, dtype, keys_per_cta)
    if key not in _kernels:
        import tilelang
        from tilelang import language as T

        block_n, num_stages, threads = 64, 3, 512

        @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                                    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
                                    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
        def build():
            rows = T.dynamic('rows')
            keys = T.dynamic('keys')

            @T.prim_func
            def kernel(Q: T.Tensor([rows * heads, head_dim], dtype),  # type: ignore
                       K: T.Tensor([keys, head_dim], dtype),  # type: ignore
                       Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
                       Starts: T.Tensor([rows], T.int32),  # type: ignore
                       Ends: T.Tensor([rows], T.int32)):  # type: ignore
                with T.Kernel(T.ceildiv(rows, block_q), T.ceildiv(keys, keys_per_cta), threads=threads) as (bx, by):
                    q_shared = T.alloc_shared([block_q * heads, head_dim], dtype)
                    k_shared = T.alloc_shared([block_n, head_dim], dtype)
                    scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                    scores_3d = T.reshape(scores, (block_n, block_q, heads))
                    reduced = T.alloc_fragment([block_n, block_q], T.float32)
                    row_base = bx * block_q
                    start_min = T.alloc_var(T.int32)
                    end_max = T.alloc_var(T.int32)
                    start_min = 2147483647
                    end_max = -2147483648
                    for qi in T.serial(block_q):
                        start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                        end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))
                    lo = T.max(start_min, by * keys_per_cta)
                    hi = T.min(end_max, by * keys_per_cta + keys_per_cta)
                    T.copy(Q[row_base * heads, 0], q_shared)
                    for ni in T.Pipelined(T.ceildiv(T.max(hi - lo, 0), block_n), num_stages=num_stages):
                        T.copy(K[lo + ni * block_n, 0], k_shared)
                        T.gemm(k_shared, q_shared, scores, transpose_B=True, clear_accum=True,
                               policy=T.GemmWarpPolicy.FullCol)
                        for n, qi, head in T.Parallel(block_n, block_q, heads):
                            scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                        T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                        for qi, n in T.Parallel(block_q, block_n):
                            if lo + ni * block_n + n < hi:
                                Logits[row_base + qi, lo + ni * block_n + n] = reduced[n, qi]

            return kernel

        _kernels[key] = build()
    return _kernels[key]


def qsa_mqa_prefill(original, q, k, row_starts, row_ends, score_scale=None):
    from sglang.srt.layers.attention.qsa import mqa

    rows, keys = q.shape[0], k.shape[0]
    if KEYS_PER_CTA <= 0 or not rows or not keys or not q.is_cuda:
        return original(q, k, row_starts, row_ends, score_scale)
    mqa._validate_q(q)
    mqa._validate_k(k)
    heads, head_dim = q.shape[1:]
    block_q = max(1, 128 // heads)
    padding = (-rows) % block_q
    logits = torch.empty((rows + padding, keys), dtype=torch.float32, device=q.device)
    scoring_dtype = mqa._scoring_dtype(q, k)
    q_padded = q.to(scoring_dtype).contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    if padding:
        q_padded = torch.cat([q_padded, q_padded.new_zeros(padding, heads, head_dim)])
        starts = torch.cat([starts, starts[-1:].expand(padding)])
        ends = torch.cat([ends, ends[-1:].expand(padding)])
    _kernel(heads, head_dim, block_q, mqa._TILELANG_DTYPES[scoring_dtype], KEYS_PER_CTA)(
        q_padded.reshape(-1, head_dim), k[:, 0].to(scoring_dtype).contiguous(), logits, starts, ends)
    logits = logits[:rows]
    logits.div_(score_scale or math.sqrt(head_dim))
    mqa._tilelang_qsa_mqa_mask_kernel()(logits, starts[:rows], ends[:rows])
    return logits
