"""Lossless SM120 prefill hooks for DeepSeek V4.1:
1. ``XTIER_DS_MHC_POST_TRITON``: bitwise-exact Triton ``deepseek_v4_mhc.post`` for ``T > 384`` (60.1 ms + 22.5 GiB
   temporary broadcast spike -> 4.36 ms, ``torch.equal == True``).
2. ``XTIER_DS_MHC_BF16X3``: prepares SGLang's 3-term BF16 exact 24-bit mantissa mHC weights (``split_bf16_hc_weight``)
   on SM120 without turning on ``SGLANG_OPT_DEEPGEMM_HC_PRENORM`` (which would call SM90/SM100-only
   ``tf32_hc_prenorm_gemm`` on unfused boundary layers), speeding fused mHC pre-norm from 5.55 ms to 1.94 ms per
   sublayer (-152 ms per 48K chunk).
3. ``XTIER_DS_MXFP8_AUTOTUNE``: keeps ``b12x`` pinned for decode / small turns (``< 512`` rows) while routing large
   prefill MXFP8 dense linears (``>= 512`` rows) in ``_b12x_or_cutlass_mxfp8_linear`` to autotuned ``cutlass`` and
   extending ``DeepseekV4ForCausalLM.autotune_prefill_kernels`` to ``is_flashinfer_cutlass()`` (1,156 ms on ``b12x`` ->
   815 ms on tuned ``cutlass`` per 40 layers at M=49,152, ``torch.equal == True``).
"""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

MHC_POST = 'sglang.srt.models.deepseek_v4_mhc.post'
MHC_POST_COMBINE_NORM_PREFILL = (
    'sglang.kernels.ops.layernorm.mhc_post_combine_norm_prefill.mhc_post_combine_norm_prefill'
)
MHC_MIX_STATS = 'sglang.srt.models.deepseek_v4_mhc._mix_stats_impl'
REFRESH_MHC = 'sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayerBase.refresh_mhc_norm_weight_cache'
AUTOTUNE_PREFILL = 'sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.autotune_prefill_kernels'
B12X_OR_CUTLASS = 'sglang.srt.layers.quantization.fp8_utils._b12x_or_cutlass_mxfp8_linear'


@triton.jit
def _post_torch_exact(p, x, c0, r0, c1, r1, c2, r2, c3, r3):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 px, m0, m1, m2, m3, s0, s1, s2;
            mul.rn.f32 px, $1, $2;
            mul.rn.f32 m0, $3, $4;
            mul.rn.f32 m1, $5, $6;
            mul.rn.f32 m2, $7, $8;
            mul.rn.f32 m3, $9, $10;
            add.rn.f32 s0, m0, m1;
            add.rn.f32 s1, s0, m2;
            add.rn.f32 s2, s1, m3;
            add.rn.f32 $0, px, s2;
        }
        """,
        constraints='=f,f,f,f,f,f,f,f,f,f,f',
        args=[p, x, c0, r0, c1, r1, c2, r2, c3, r3],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mhc_post_exact_kernel(
    X,
    RES,
    POST,
    COMB,
    OUT,
    M,
    H: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_h * BH + tl.arange(0, BH)
    rmask = rows < M
    cmask = cols < H
    mask = rmask[:, None] & cmask[None, :]

    rows64 = rows.to(tl.int64)
    x = tl.load(X + rows64[:, None] * H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    r0 = tl.load(RES + (rows64[:, None] * 4 + 0) * H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(RES + (rows64[:, None] * 4 + 1) * H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    r2 = tl.load(RES + (rows64[:, None] * 4 + 2) * H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    r3 = tl.load(RES + (rows64[:, None] * 4 + 3) * H + cols[None, :], mask=mask, other=0.0).to(tl.float32)

    for i in tl.static_range(4):
        p = tl.load(POST + rows64 * 4 + i, mask=rmask, other=0.0)[:, None]
        c0 = tl.load(COMB + rows64 * 16 + 0 * 4 + i, mask=rmask, other=0.0)[:, None]
        c1 = tl.load(COMB + rows64 * 16 + 1 * 4 + i, mask=rmask, other=0.0)[:, None]
        c2 = tl.load(COMB + rows64 * 16 + 2 * 4 + i, mask=rmask, other=0.0)[:, None]
        c3 = tl.load(COMB + rows64 * 16 + 3 * 4 + i, mask=rmask, other=0.0)[:, None]
        u = _post_torch_exact(p, x, c0, r0, c1, r1, c2, r2, c3, r3).to(tl.bfloat16)
        tl.store(OUT + (rows64[:, None] * 4 + i) * H + cols[None, :], u, mask=mask)


def _inplace_enabled() -> bool:
    return os.environ.get('XTIER_DS_MHC_INPLACE', '1') != '0'


def mhc_post_exact(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb: torch.Tensor,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    m, h = x.shape
    out = residual if inplace else torch.empty_like(residual)
    bm, bh = 32, 128
    _mhc_post_exact_kernel[(triton.cdiv(m, bm), triton.cdiv(h, bh))](
        x, residual, post_mix, comb, out, m, H=h, BM=bm, BH=bh, num_warps=4, num_stages=2
    )
    return out


@triton.jit
def _post_fma_exact(p, x, c0, r0, c1, r1, c2, r2, c3, r3):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 acc;
            mul.rn.f32 acc, $3, $4;
            fma.rn.f32 acc, $1, $2, acc;
            fma.rn.f32 acc, $5, $6, acc;
            fma.rn.f32 acc, $7, $8, acc;
            fma.rn.f32 $0, $9, $10, acc;
        }
        """,
        constraints='=f,f,f,f,f,f,f,f,f,f,f',
        args=[p, x, c0, r0, c1, r1, c2, r2, c3, r3],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _comb4_fma_exact(p0, u0, p1, u1, p2, u2, p3, u3):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 acc;
            mul.rn.f32 acc, $1, $2;
            fma.rn.f32 acc, $3, $4, acc;
            fma.rn.f32 acc, $5, $6, acc;
            fma.rn.f32 $0, $7, $8, acc;
        }
        """,
        constraints='=f,f,f,f,f,f,f,f,f',
        args=[p0, u0, p1, u1, p2, u2, p3, u3],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _norm_mul_exact(cf, inv_rms, wf):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 t;
            mul.rn.f32 t, $1, $2;
            mul.rn.f32 $0, t, $3;
        }
        """,
        constraints='=f,f,f,f',
        args=[cf, inv_rms, wf],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mhc_post_combine_norm_inplace_kernel(
    X,
    RES,
    POST,
    COMB,
    PRE,
    WEIGHT,
    OUT,
    M,
    eps,
    H: tl.constexpr,
    BH: tl.constexpr,
):
    """Tile-safe in-place prefill mHC post + combine + RMSNorm (saves [M, 4, H] BF16 copy = 1.88 GiB at 48K)."""
    row = tl.program_id(0).to(tl.int64)
    if row >= M:
        return
    p0 = tl.load(POST + row * 4 + 0)
    p1 = tl.load(POST + row * 4 + 1)
    p2 = tl.load(POST + row * 4 + 2)
    p3 = tl.load(POST + row * 4 + 3)
    pre0 = tl.load(PRE + row * 4 + 0)
    pre1 = tl.load(PRE + row * 4 + 1)
    pre2 = tl.load(PRE + row * 4 + 2)
    pre3 = tl.load(PRE + row * 4 + 3)
    c00 = tl.load(COMB + row * 16 + 0)
    c10 = tl.load(COMB + row * 16 + 4)
    c20 = tl.load(COMB + row * 16 + 8)
    c30 = tl.load(COMB + row * 16 + 12)
    c01 = tl.load(COMB + row * 16 + 1)
    c11 = tl.load(COMB + row * 16 + 5)
    c21 = tl.load(COMB + row * 16 + 9)
    c31 = tl.load(COMB + row * 16 + 13)
    c02 = tl.load(COMB + row * 16 + 2)
    c12 = tl.load(COMB + row * 16 + 6)
    c22 = tl.load(COMB + row * 16 + 10)
    c32 = tl.load(COMB + row * 16 + 14)
    c03 = tl.load(COMB + row * 16 + 3)
    c13 = tl.load(COMB + row * 16 + 7)
    c23 = tl.load(COMB + row * 16 + 11)
    c33 = tl.load(COMB + row * 16 + 15)

    x_base = X + row * H
    r_base = RES + row * 4 * H
    o_base = OUT + row * H

    sqr_sum = tl.zeros([BH], dtype=tl.float32)
    for start in tl.static_range(0, H, BH):
        cols = start + tl.arange(0, BH)
        xv = tl.load(x_base + cols).to(tl.float32)
        r0 = tl.load(r_base + 0 * H + cols).to(tl.float32)
        r1 = tl.load(r_base + 1 * H + cols).to(tl.float32)
        r2 = tl.load(r_base + 2 * H + cols).to(tl.float32)
        r3 = tl.load(r_base + 3 * H + cols).to(tl.float32)

        u0_bf = _post_fma_exact(p0, xv, c00, r0, c10, r1, c20, r2, c30, r3).to(tl.bfloat16)
        u1_bf = _post_fma_exact(p1, xv, c01, r0, c11, r1, c21, r2, c31, r3).to(tl.bfloat16)
        u2_bf = _post_fma_exact(p2, xv, c02, r0, c12, r1, c22, r2, c32, r3).to(tl.bfloat16)
        u3_bf = _post_fma_exact(p3, xv, c03, r0, c13, r1, c23, r2, c33, r3).to(tl.bfloat16)

        tl.store(r_base + 0 * H + cols, u0_bf)
        tl.store(r_base + 1 * H + cols, u1_bf)
        tl.store(r_base + 2 * H + cols, u2_bf)
        tl.store(r_base + 3 * H + cols, u3_bf)

        comb_f = _comb4_fma_exact(
            pre0, u0_bf.to(tl.float32),
            pre1, u1_bf.to(tl.float32),
            pre2, u2_bf.to(tl.float32),
            pre3, u3_bf.to(tl.float32),
        ).to(tl.bfloat16)
        tl.store(o_base + cols, comb_f)
        cf = comb_f.to(tl.float32)
        sqr_sum += cf * cf

    mean = tl.sum(sqr_sum, axis=0) / float(H)
    inv_rms = tl.rsqrt(mean + eps)

    for start in tl.static_range(0, H, BH):
        cols = start + tl.arange(0, BH)
        cf = tl.load(o_base + cols).to(tl.float32)
        wf = tl.load(WEIGHT + cols).to(tl.float32)
        tl.store(o_base + cols, _norm_mul_exact(cf, inv_rms, wf).to(tl.bfloat16))


def mhc_post_combine_norm_prefill_inplace(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    m, h = x.shape
    out = torch.empty_like(x)
    _mhc_post_combine_norm_inplace_kernel[(m,)](
        x, residual, post, comb, pre, weight, out, m, float(eps), H=h, BH=1024, num_warps=8
    )
    return residual, out


def _post_combine_norm_prefill_hook(original, x, residual, post, comb, pre, weight, eps):
    if (
        _inplace_enabled()
        and torch.version.hip is None
        and x.is_cuda
        and not torch.cuda.is_current_stream_capturing()
        and x.dim() == 2
        and 4096 <= x.shape[0] <= 65536
        and x.shape[1] == 5120
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and post.dtype == torch.float32
        and comb.dtype == torch.float32
        and pre.dtype == torch.float32
        and residual.shape == (x.shape[0], 4, 5120)
        and post.shape == (x.shape[0], 4)
        and pre.shape == (x.shape[0], 4)
        and comb.shape == (x.shape[0], 4, 4)
        and weight.shape == (5120,)
        and x.is_contiguous()
        and residual.is_contiguous()
        and post.is_contiguous()
        and comb.is_contiguous()
        and pre.is_contiguous()
        and weight.is_contiguous()
        and residual.data_ptr() != x.data_ptr()
    ):
        return mhc_post_combine_norm_prefill_inplace(x, residual, post, comb, pre, weight, eps)
    return original(x, residual, post, comb, pre, weight, eps)


def _post_hook(original, cfg, x, residual, post_mix, comb):
    if (
        torch.version.hip is None
        and x.is_cuda
        and x.dim() == 2
        and x.shape[0] > 384
        and cfg.mult == 4
        and x.shape[1] % 128 == 0
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and post_mix.dtype == torch.float32
        and comb.dtype == torch.float32
        and residual.shape == (x.shape[0], 4, x.shape[1])
        and post_mix.shape == (x.shape[0], 4)
        and comb.shape == (x.shape[0], 4, 4)
        and x.is_contiguous()
        and residual.is_contiguous()
        and post_mix.is_contiguous()
        and comb.is_contiguous()
    ):
        inplace = _inplace_enabled() and not torch.cuda.is_current_stream_capturing()
        return mhc_post_exact(x, residual, post_mix, comb, inplace=inplace)
    return original(cfg, x, residual, post_mix, comb)


def _refresh_mhc_hook(original, self, *args, **kwargs):
    ret = original(self, *args, **kwargs)
    if (
        self.hc_pre_from_prev_sublayer
        and self._hc_attn_bf16_parts is None
        and self.hc_attn_fn.shape == (24, 20480)
        and getattr(self.config, 'model_type', None) == 'deepseek_v41'
    ):
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

        if not is_batch_invariant_mode_enabled():
            from sglang.kernels.ops.layernorm.mhc import split_bf16_hc_weight

            self._hc_attn_bf16_parts = split_bf16_hc_weight(self.hc_attn_fn.data)
            self._hc_ffn_bf16_parts = split_bf16_hc_weight(self.hc_ffn_fn.data)
            self._init_hyper_connections()
            self.__dict__.pop('_can_fuse_attn_mhc', None)
            self.__dict__.pop('_can_fuse_ffn_mhc', None)
    return ret


def _mix_stats_hook(original, hc, x):
    bf16_parts = hc.bf16_parts
    if bf16_parts is not None and x.is_cuda and x.dtype == torch.bfloat16:
        x_flat = x.flatten(1)
        if 4096 <= x_flat.shape[0] <= 65536 and x_flat.is_contiguous():
            from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

            if not is_batch_invariant_mode_enabled():
                from sglang.kernels.ops.layernorm.mhc import hc_mix_stats_sinkhorn_bf16x3

                cfg = hc.cfg
                return hc_mix_stats_sinkhorn_bf16x3(
                    x_flat,
                    bf16_parts,
                    hc.scale,
                    hc.base,
                    cfg.sinkhorn_iters,
                    cfg.rms_eps,
                    cfg.eps,
                    decode_grid=False,
                )
    return original(hc, x)


def _b12x_or_cutlass_hook(original, input, weight, *args, **kwargs):
    rows = input.numel() // input.shape[-1]
    if rows >= int(os.environ.get('XTIER_DS_MXFP8_CUTLASS_MIN_M', '768')):
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_blockscaled_linear

        pin = kwargs.pop('pin_tactic', is_batch_invariant_mode_enabled())
        return flashinfer_mxfp8_blockscaled_linear(
            input, weight, *args, backend='cutlass', pin_tactic=pin, **kwargs
        )
    return original(input, weight, *args, **kwargs)


def _autotune_prefill_cutlass_hook(original, self, num_tokens: int, *, dtype: torch.dtype) -> int:
    tuned = original(self, num_tokens, dtype=dtype)
    if tuned or getattr(self.config, 'model_type', None) != 'deepseek_v41':
        return tuned
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    seen = set()
    for layer in self.model.modules():
        method = getattr(layer, 'quant_method', None)
        if not isinstance(method, Fp8LinearMethod):
            continue
        if not (method.use_mxfp8 or method.block_fp8_as_mxfp8):
            continue
        if method.block_fp8_as_mxfp8 and not getattr(layer, 'block_fp8_mxfp8_ready', False):
            continue
        backend = method.mxfp8_dense_backend
        if backend is None or not backend.is_flashinfer_cutlass():
            continue
        weight = layer.weight
        if weight.numel() == 0:
            continue
        scale = layer.weight_scale_inv_swizzled
        key = (weight.shape, weight.stride(), weight.dtype, scale.shape, scale.stride(), scale.dtype)
        if key in seen:
            continue
        seen.add(key)
        x = torch.zeros((num_tokens, weight.shape[1]), dtype=dtype, device=weight.device)
        method.apply(layer, x)
        del x
    if seen:
        logger.info('xtier: FlashInfer CUTLASS prefill autotune: %d MXFP8 weight layouts at M=%d', len(seen), num_tokens)
    return len(seen)


def install(registry, hook_type):
    cap = torch.cuda.get_device_capability()
    if os.environ.get('XTIER_DS_MHC_POST_TRITON', '1') != '0' and cap[0] >= 8:
        registry.register(MHC_POST, _post_hook, hook_type.AROUND)
    if _inplace_enabled() and cap[0] >= 10:
        registry.register(MHC_POST_COMBINE_NORM_PREFILL, _post_combine_norm_prefill_hook, hook_type.AROUND)
    if os.environ.get('XTIER_DS_MHC_BF16X3', '1') != '0' and cap == (12, 0):
        registry.register(REFRESH_MHC, _refresh_mhc_hook, hook_type.AROUND)
        registry.register(MHC_MIX_STATS, _mix_stats_hook, hook_type.AROUND)
    if os.environ.get('XTIER_DS_MXFP8_AUTOTUNE', '1') != '0' and cap[0] >= 10:
        registry.register(AUTOTUNE_PREFILL, _autotune_prefill_cutlass_hook, hook_type.AROUND)
        if cap == (12, 0):
            registry.register(B12X_OR_CUTLASS, _b12x_or_cutlass_hook, hook_type.AROUND)
