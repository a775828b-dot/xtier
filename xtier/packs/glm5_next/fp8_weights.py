"""FP8 storage of selected BF16 linear weights (XTIER_FP8_WEIGHTS=1; ported from the GLM-5.3 pipeline's
lkqmoe/gpu/fp8_weights.py, the parts its production configuration uses).

GLM-5.3 Flash NVFP4 keeps its attention and dense layers in BF16 (~15 GB, read in full every decode step). After
loading, the weights of the selected UnquantizedLinearMethod layers become FP8 E4M3 with one FP32 scale per output
row and 128-column group (amax / 448), and the BF16 copy is freed. Activations stay BF16 except where noted:
- M <= 16 rows (decode, verify): one-launch split-K Triton kernel, FP8 tiles dequantized to BF16 in registers,
  FP32 accumulation, the slices added in a fixed order by the last CTA of each column tile (deterministic);
- larger M (prefill): layers matching XTIER_FP8_W8A8_PREFILL run a W8A8 GEMM (activations quantized to E4M3 per
  token and 128 columns, FP32 accumulation of the per-group scaled FP8 dots); the others dequantize the weight to a
  temporary BF16 tensor for F.linear (the decode path's values).
Selection by full module name after loading: matches XTIER_FP8_INCLUDE and not XTIER_FP8_EXCLUDE, a 2-D BF16
weight with K % 128 == 0 and at least XTIER_FP8_MIN_ELEMS elements. Excluded by default: weights other kernels read
directly (fused_qkv_a_proj, kv_b_proj: MLA absorbed matrices), shared experts, routers, vision.
"""

import logging
import os
import re

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
GROUP = 128
MAX_M = 16
INCLUDE = r'[.](self_attn|mlp)[.]|eh_proj'
EXCLUDE = r'(fused_qkv_a_proj|kv_b_proj|shared_experts|[.]gate$|visual|vision|weights_proj)'


def quantize(w):
    """w [N, K] BF16 -> (FP8 [N, K], FP32 scale [N, K / 128])."""
    n, k = w.shape
    g = w.float().view(n, k // GROUP, GROUP)
    s = (g.abs().amax(-1) / 448.0).clamp_min(1e-12)
    q = (g / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(n, k)
    return q.contiguous(), s.contiguous()


@triton.jit
def _w8_fused(X, W, S, P, CNT, BIAS, Y, M, N, K, SPAN, NG, SLICES: tl.constexpr, BM: tl.constexpr,
              BN: tl.constexpr, BK: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Decode W8A16 in one launch: split-K slices of FP8 tiles dequantized to BF16, FP32 accumulation; the last CTA
    of a column tile adds the slice partials in slice order and resets the tile counter."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    rows = tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = ps * SPAN
    for kk in range(0, SPAN, BK):
        ks = k0 + kk + tl.arange(0, BK)
        kmask = ks < K
        x = tl.load(X + rows[:, None] * K + ks[None, :], mask=(rows[:, None] < M) & kmask[None, :], other=0.0)
        w = tl.load(W + cols[:, None] * K + ks[None, :], mask=cmask[:, None] & kmask[None, :], other=0.0)
        # the last slice may extend past K: its group scale is masked too
        s = tl.load(S + cols * NG + (k0 + kk) // 128, mask=cmask & (k0 + kk < K), other=0.0)
        wb = (w.to(tl.float32) * s[:, None]).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(wb), acc)
    omask = (rows[:, None] < M) & cmask[None, :]
    if SLICES == 1:
        r = acc
        if HAS_BIAS:
            r += tl.load(BIAS + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
        tl.store(Y + rows[:, None] * N + cols[None, :], r.to(tl.bfloat16), mask=omask)
    else:
        tl.store(P + (ps * BM + rows[:, None]) * N + cols[None, :], acc, mask=omask)
        tl.debug_barrier()
        done = tl.atomic_add(CNT + pn, 1, sem='acq_rel')
        if done == SLICES - 1:
            r = tl.zeros((BM, BN), dtype=tl.float32)
            for sl in range(SLICES):   # fixed order: deterministic
                r += tl.load(P + (sl * BM + rows[:, None]) * N + cols[None, :], mask=omask, other=0.0,
                             cache_modifier='.cg')
            if HAS_BIAS:
                r += tl.load(BIAS + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
            tl.store(Y + rows[:, None] * N + cols[None, :], r.to(tl.bfloat16), mask=omask)
            tl.atomic_xchg(CNT + pn, 0)


@triton.jit
def _w8_dequant(W, S, D, N, K, NG, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = (rows[:, None] < N) & (ks[None, :] < K)
    w = tl.load(W + rows[:, None] * K + ks[None, :], mask=mask, other=0.0)
    s = tl.load(S + rows * NG + (tl.program_id(1) * BK) // 128, mask=rows < N, other=0.0)
    tl.store(D + rows[:, None] * K + ks[None, :], (w.to(tl.float32) * s[:, None]).to(tl.bfloat16), mask=mask)


@triton.jit
def _act_quant_fp8(X, Q, S, M, K, NG, BK: tl.constexpr):
    r = tl.program_id(0)
    g = tl.program_id(1)
    ks = g * BK + tl.arange(0, BK)
    x = tl.load(X + r * K + ks, mask=ks < K, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), 0) / 448.0, 1e-12)
    tl.store(Q + r * K + ks, (x / s).to(tl.float8e4nv), mask=ks < K)
    tl.store(S + r * NG + g, s)


@triton.jit
def _w8a8_gemm(A, SA, B, SB, C, M, N, K, NG, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = pid // (GROUP_M * num_n)
    first_m = group * GROUP_M
    gsize = tl.minimum(num_m - first_m, GROUP_M)
    pm = first_m + (pid % (GROUP_M * num_n)) % gsize
    pn = (pid % (GROUP_M * num_n)) // gsize
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(0, NG):
        ks = g * BK + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * K + ks[None, :], mask=rm[:, None] < M, other=0.0)
        b = tl.load(B + rn[:, None] * K + ks[None, :], mask=rn[:, None] < N, other=0.0)
        sa = tl.load(SA + rm * NG + g, mask=rm < M, other=0.0)
        sb = tl.load(SB + rn * NG + g, mask=rn < N, other=0.0)
        acc += tl.dot(a, tl.trans(b)) * (sa[:, None] * sb[None, :])
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def _split(n, k):
    block_n = 32 if n <= 512 else 64
    slices = max(1, min(k // 128, triton.cdiv(440, triton.cdiv(n, block_n))))
    return block_n, 128, slices


# (N, K) -> (BN, warps, stages) of _w8_fused on the RTX PRO 5000 (M = 5, cold L2; the GLM-5.3 pipeline's
# bench/w8_tune.py); other shapes: BN of _split, 4 warps, 3 stages.
_W8_CFG = {(4096, 8192): (64, 4, 2), (16384, 1536): (64, 4, 2), (4096, 16384): (64, 4, 2), (4096, 1536): (32, 4, 2),
           (4096, 12288): (64, 4, 2), (4096, 4096): (128, 8, 2), (4096, 2048): (32, 4, 2)}


def gemm_w8(x, q, s, bias=None, cnt=None):
    """x [M, K] BF16 (M <= 16), q FP8 [N, K], s FP32 [N, K / 128] -> [M, N] BF16. cnt: int32 zeros, the split-K
    tile counters (one tensor per layer, so CUDA graphs need no memset: every use resets its entries)."""
    m, k = x.shape
    n = q.shape[0]
    block_n, block_k, slices = _split(n, k)
    span = triton.cdiv(triton.cdiv(k, slices), block_k) * block_k
    slices = triton.cdiv(k, span)
    bn, warps, stages = _W8_CFG.get((n, k), (block_n, 4, 3))
    if cnt is None or cnt.numel() < triton.cdiv(n, bn):
        cnt = torch.zeros(triton.cdiv(n, bn), dtype=torch.int32, device=x.device)
    part = torch.empty((slices, MAX_M, n), device=x.device, dtype=torch.float32) if slices > 1 else x
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _w8_fused[(triton.cdiv(n, bn), slices)](x, q, s, part, cnt, bias if bias is not None else x, y, m, n, k, span,
                                            k // GROUP, slices, MAX_M, bn, block_k, bias is not None,
                                            num_warps=warps, num_stages=stages)
    return y


def dequant(q, s):
    n, k = q.shape
    d = torch.empty((n, k), device=q.device, dtype=torch.bfloat16)
    _w8_dequant[(triton.cdiv(n, 32), triton.cdiv(k, 128))](q, s, d, n, k, k // GROUP, 32, 128)
    return d


def gemm_w8a8(x, q, s, bias=None, bm=128, bn=128, warps=8, stages=3):
    """x [M, K] BF16 (prefill), q FP8 [N, K], s FP32 [N, K / 128] -> [M, N] BF16."""
    m, k = x.shape
    n, ng = q.shape[0], k // GROUP
    xq = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    xs = torch.empty((m, ng), device=x.device, dtype=torch.float32)
    _act_quant_fp8[(m, ng)](x, xq, xs, m, k, ng, GROUP)
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _w8a8_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](xq, xs, q, s, y, m, n, k, ng, bm, bn, GROUP, 8,
                                                          num_warps=warps, num_stages=stages)
    return y if bias is None else y + bias


def forward(layer, x, bias=None):
    q, s = layer.weight, layer._xtier_fp8_scale
    k = x.shape[-1]
    m = x.numel() // k
    x2 = x.reshape(m, k)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if m <= MAX_M:
        y = gemm_w8(x2, q, s, bias, layer._xtier_w8_cnt)
    elif layer._xtier_w8a8:
        y = gemm_w8a8(x2, q, s, bias)
    else:
        y = torch.nn.functional.linear(x2, dequant(q, s), bias)
    return y.view(*x.shape[:-1], q.shape[0])


def quantize_model(model):
    """After every weight is loaded and post-processed: the selected layers, by full module name (the LinearBase
    prefix is empty for many layers)."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    include = re.compile(os.environ.get('XTIER_FP8_INCLUDE', INCLUDE))
    exclude = re.compile(os.environ.get('XTIER_FP8_EXCLUDE', EXCLUDE))
    w8a8 = os.environ.get('XTIER_FP8_W8A8_PREFILL', '')
    w8a8 = re.compile(w8a8) if w8a8 else None
    min_elems = int(os.environ.get('XTIER_FP8_MIN_ELEMS', str(4 << 20)))
    names, w8a8_layers, bf16_bytes, fp8_bytes = [], 0, 0, 0
    for name, layer in model.named_modules():
        if not isinstance(getattr(layer, 'quant_method', None), UnquantizedLinearMethod):
            continue
        w = getattr(layer, 'weight', None)
        if (not isinstance(w, torch.Tensor) or not w.is_cuda or w.dtype != torch.bfloat16 or w.dim() != 2
                or w.shape[1] % GROUP or w.numel() < min_elems or getattr(layer, '_xtier_fp8_scale', None) is not None
                or not include.search(name) or exclude.search(name)):
            continue
        q, sc = quantize(w.data)
        layer.weight = torch.nn.Parameter(q, requires_grad=False)
        layer._xtier_fp8_scale = sc
        layer._xtier_w8_cnt = torch.zeros(triton.cdiv(q.shape[0], 32), dtype=torch.int32, device=q.device)
        layer._xtier_w8a8 = bool(w8a8 is not None and w8a8.search(name))
        w8a8_layers += layer._xtier_w8a8
        bf16_bytes += w.numel() * 2
        fp8_bytes += q.numel() + sc.numel() * 4
        names.append(name)
        del w
    torch.cuda.empty_cache()
    if names:
        logger.info('xtier: FP8 weights for %d BF16 linear layers of %s (%d with W8A8 prefill): %.2f -> %.2f GiB, '
                    'e.g. %s', len(names), type(model).__name__, w8a8_layers, bf16_bytes / 2**30, fp8_bytes / 2**30,
                    names[:3])


def _stored(layer):
    return getattr(layer, '_xtier_fp8_scale', None) is not None


def apply(original, self, layer, x, bias=None):
    return forward(layer, x, bias) if _stored(layer) else original(self, layer, x, bias)


def apply_into(original, self, layer, x, output, bias=None):
    if not _stored(layer):
        return original(self, layer, x, output, bias)
    output.copy_(forward(layer, x, bias).view_as(output))
    return output


def apply_with_addend(original, self, layer, x, addend, bias=None):
    if not _stored(layer):   # SGLang's cuBLAS route would accumulate into addend through the BF16 weight
        return original(self, layer, x, addend, bias)
    return forward(layer, x, bias).add_(addend)


def load_weights_and_postprocess(original, model, weights, target_device):
    result = original(model, weights, target_device)
    quantize_model(model)
    return result
