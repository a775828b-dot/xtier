"""DeepSeek V4.1 attention ``wo_a`` (the grouped low-rank output projection) at decode sizes on a Triton grouped
einsum, the target model's weights kept in the checkpoint's FP8 (``XTIER_DS_WO_A``, default on; the 1017 pipeline's
LKQMOE_DS41_WO_A_TRITON and LKQMOE_DS41_WO_A_FP8).

SGLang's FP8 wo_a GEMM takes 128 x 128 block scales only. DeepSeek V4.1's wo_a has 32 x 32 E8M0 blocks, so SGLang
dequantizes it to BF16 at load and, off its TP4 fast-path shapes, runs ``torch.einsum``, which cuBLAS serves on SM120
with an SM80 WMMA batched GEMM: on the RTX PRO 5000 71-75 us per layer at the DSpark verify's 6 tokens, 3.2 ms of a
33.7 ms decode step (the 1017 pipeline: 1.4 ms). Here (tests/ds_wo_a_check.py, weights out of L2): FP8 30 us, the
BF16 form 57 us; within 1e-4 of cuBLAS, 99.99% of outputs bitwise equal.

* FP8 weights (``XTIER_DS_WO_A_FP8``, default on): the E8M0 scales are powers of two, so a target layer's BF16 weight
  goes back to the checkpoint's FP8 exactly. A layer is converted only if its FP8 dequantizes to the loaded BF16 bit
  for bit and the kernel matches SGLang's einsum on a random input (1e-3 relative); otherwise it keeps BF16. 40
  layers: 1.25 GiB less VRAM. The scales are taken from the checkpoint stream on its way to SGLang's dequantization.
* Calls of up to ``XTIER_DS_WO_A_MAX_T`` tokens (default 16: decode and the verify of two DSpark requests) run one
  Triton kernel that forms each weight tile in BF16 (FP8 x scale, or the BF16 weight) and accumulates in FP32: the
  same values into the same kind of dot, so the output differs from cuBLAS's by the summation order only. Larger
  calls (prefill) dequantize an FP8 layer into a reused BF16 buffer and keep SGLang's path.
* The DSpark draft (its own class, BF16 weights) calls the same function and gets the BF16 form of the kernel.
* FP8 weights are read as bytes and decoded by xtier.arch.e4m3_f32: the hardware conversion from SM89 on, the same
  values in software on RTX 30 series.
"""

import logging
import os
import re

import torch
import triton
import triton.language as tl

from xtier.arch import e4m3_f32

logger = logging.getLogger(__name__)
MODELS = 'sglang.srt.models.deepseek_v4'
APPLY = f'{MODELS}._apply_wo_a_bf16_matmul'
STREAMING = f'{MODELS}._dequant_fp8_wo_a_streaming'
LOAD = f'{MODELS}.DeepseekV4ForCausalLM.load_weights'
SCALE_NAME = re.compile(r'(?:^|\.)layers\.(\d+)\.attn\.wo_a\.scale$')
LAYER_NAME = re.compile(r'(?:^|\.)layers\.(\d+)\.')
BLOCK = 32                        # the checkpoint's FP8 block (rows and columns)
FAST_SHAPE = (2, 1024, 4096)      # (groups, rank, dim) of SGLang's own BF16 fast paths (TP4): left to them
MAX_T = int(os.environ.get('XTIER_DS_WO_A_MAX_T', '16'))
BLOCK_T, BLOCK_R, BLOCK_H = 16, 32, 128   # 256 CTAs for (8, 1024, 4096): FP8 30 us (BLOCK_R 128 as in lkqmoe: 32.5)
_scales = {}        # FP8 weight data_ptr -> FP32 block scales [rows / 32, cols / 32]
_scratch = {}       # (device, numel) -> BF16 buffer for dequantized larger calls
_capture = [None]   # {layer: E8M0 scale} while the target model loads


def enabled():
    return os.environ.get('XTIER_DS_WO_A', '1') != '0'


@triton.jit
def _einsum_kernel(o_ptr, w_ptr, s_ptr, out_ptr, T, o_st, o_sg, o_sh, w_sg, w_sr, w_sh, s_stride, out_st, out_sg,
                   out_sr, R: tl.constexpr, D: tl.constexpr, FP8: tl.constexpr, SCALE_BN: tl.constexpr,
                   SCALE_BK: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_H: tl.constexpr):
    """out[t, g, r] = sum_h o[t, g, h] * w[g, r, h], w = FP8 x its E8M0 block scale (FP8) or BF16; FP32 accumulator."""
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    g = tl.program_id(2)
    t_mask, r_mask = t < T, r < R
    acc = tl.zeros((BLOCK_T, BLOCK_R), tl.float32)
    for h0 in range(0, D, BLOCK_H):
        h = h0 + tl.arange(0, BLOCK_H)
        h_mask = h < D
        a = tl.load(o_ptr + t[:, None] * o_st + g * o_sg + h[None, :] * o_sh,
                    mask=t_mask[:, None] & h_mask[None, :], other=0.0)
        w = tl.load(w_ptr + g * w_sg + r[:, None] * w_sr + h[None, :] * w_sh,
                    mask=r_mask[:, None] & h_mask[None, :], other=0.0)
        if FP8:   # one scale per SCALE_BK columns of a SCALE_BN-row block
            k = h0 // SCALE_BK + tl.arange(0, BLOCK_H // SCALE_BK)
            s = tl.load(s_ptr + ((g * R + r) // SCALE_BN)[:, None] * s_stride + k[None, :],
                        mask=r_mask[:, None] & (k * SCALE_BK < D)[None, :], other=0.0)
            w = tl.reshape(tl.reshape(e4m3_f32(w), (BLOCK_R, BLOCK_H // SCALE_BK, SCALE_BK)) * s[:, :, None],
                           (BLOCK_R, BLOCK_H)).to(tl.bfloat16)
        acc = tl.dot(a, tl.trans(w), acc=acc)
    tl.store(out_ptr + t[:, None] * out_st + g * out_sg + r[None, :] * out_sr, acc.to(out_ptr.dtype.element_ty),
             mask=t_mask[:, None] & r_mask[None, :])


@triton.jit
def _dequant_kernel(w_ptr, s_ptr, out_ptr, rows, cols, s_stride, SCALE_BN: tl.constexpr, SCALE_BK: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    r = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (r < rows)[:, None] & (c < cols)[None, :]
    w = tl.load(w_ptr + r[:, None] * cols + c[None, :], mask=mask, other=0.0)
    s = tl.load(s_ptr + (r // SCALE_BN)[:, None] * s_stride + (c // SCALE_BK)[None, :], mask=mask, other=0.0)
    tl.store(out_ptr + r[:, None] * cols + c[None, :], (e4m3_f32(w) * s).to(tl.bfloat16), mask=mask)


def einsum(o, w, scale=None):
    """o [T, G, D] BF16 x w [G, R, D] (FP8 with FP32 block scales [G R / bn, D / bk], or BF16) -> [T, G, R] BF16."""
    T, G, D = o.shape
    R = w.shape[1]
    out = torch.empty((T, G, R), dtype=o.dtype, device=o.device)
    fp8 = scale is not None
    s = scale if fp8 else w
    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(R, BLOCK_R), G)
    _einsum_kernel[grid](o, w.view(torch.uint8) if fp8 else w, s, out, T, *o.stride(), *w.stride(), s.stride(0),
                         *out.stride(), R=R, D=D, FP8=fp8,
                         SCALE_BN=G * R // scale.shape[0] if fp8 else 1, SCALE_BK=D // scale.shape[1] if fp8 else 1,
                         BLOCK_T=BLOCK_T, BLOCK_R=BLOCK_R, BLOCK_H=BLOCK_H, num_warps=4, num_stages=3)
    return out


def dequantize(w, scale, out):
    """FP8 w [rows, cols] x FP32 block scales -> BF16 out [rows, cols]."""
    rows, cols = w.shape
    _dequant_kernel[(triton.cdiv(rows, 64), triton.cdiv(cols, 128))](
        w.view(torch.uint8), scale, out, rows, cols, scale.stride(0), SCALE_BN=rows // scale.shape[0], SCALE_BK=cols // scale.shape[1],
        BLOCK_M=64, BLOCK_N=128, num_warps=4)
    return out


def _apply(original, o, wo_a, *args, **kwargs):
    """Around SGLang's ``_apply_wo_a_bf16_matmul(o [T, G, D], wo_a [G, R, D], ...)``."""
    scale = _scales.get(wo_a.data_ptr()) if wo_a.dtype == torch.float8_e4m3fn else None
    small = o.dim() == 3 and 0 < o.shape[0] <= MAX_T and o.dtype == torch.bfloat16 and o.is_cuda
    if small and (scale is not None or (wo_a.dtype == torch.bfloat16 and tuple(wo_a.shape) != FAST_SHAPE)):
        return einsum(o, wo_a, scale)
    if scale is not None:
        key = (wo_a.device, wo_a.numel())
        if key not in _scratch:
            _scratch[key] = torch.empty(wo_a.shape, dtype=torch.bfloat16, device=wo_a.device)
        flat = wo_a.reshape(-1, wo_a.shape[-1])
        wo_a = dequantize(flat, scale, _scratch[key].view(flat.shape)).view(wo_a.shape)
    return original(o, wo_a, *args, **kwargs)


def _tap(weights):
    for name, tensor in weights:
        m = SCALE_NAME.search(name) if _capture[0] is not None else None
        if m:
            _capture[0][int(m.group(1))] = tensor.detach().clone()
        yield name, tensor


def _streaming(original, weights, *args, **kwargs):
    """Around SGLang's load-time wo_a dequantization: the target model's E8M0 scales are kept on the way."""
    return original(_tap(weights) if _capture[0] is not None else weights, *args, **kwargs)


def _load_weights(original, self, weights, *args, **kwargs):
    if kwargs.get('is_nextn', args[0] if args else False):
        return original(self, weights, *args, **kwargs)
    _capture[0] = {}
    try:
        result = original(self, weights, *args, **kwargs)
        captured = _capture[0]
    finally:
        _capture[0] = None
    if captured:
        _convert(self, captured)
    return result


def _relative(a, b):
    return ((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-30)).item()


def _convert(model, captured):
    """Each captured layer's BF16 wo_a back to its exact FP8 form, checked; layers that do not check keep BF16."""
    done, kept, freed = 0, [], 0
    for name, mod in model.named_modules():
        m = LAYER_NAME.search(name + '.')
        if not (m and int(m.group(1)) in captured and all(hasattr(mod, a) for a in ('wo_a', 'o_lora_rank',
                                                                                   'n_local_groups'))):
            continue
        layer, w = int(m.group(1)), getattr(mod.wo_a, 'weight', None)
        if w is None or w.dtype != torch.bfloat16 or not w.is_cuda or w.dim() != 2:
            continue
        G, R = mod.n_local_groups, mod.o_lora_rank
        s = captured[layer].to(device=w.device).float().contiguous()
        sn, sk = s.shape
        ok = (G, R, w.shape[1]) != FAST_SHAPE and w.shape == (sn * BLOCK, sk * BLOCK) and G * R == w.shape[0]
        if ok:
            blocks = w.float().view(sn, BLOCK, sk, BLOCK)
            fp8 = (blocks / s[:, None, :, None]).view_as(w).to(torch.float8_e4m3fn)
            ok = torch.equal((fp8.float().view(sn, BLOCK, sk, BLOCK) * s[:, None, :, None]).view_as(w).to(w.dtype), w)
            del blocks
        if ok:
            o = torch.randn(6, G, w.shape[1], device=w.device, dtype=torch.bfloat16)
            ok = _relative(torch.einsum('tgd,grd->tgr', o, w.view(G, R, -1)), einsum(o, fp8.view(G, R, -1), s)) <= 1e-3
        if not ok:
            kept.append(layer)
            continue
        mod.wo_a.weight = torch.nn.Parameter(fp8, requires_grad=False)
        _scales[fp8.data_ptr()] = s
        freed += w.numel()
        done += 1
    torch.cuda.empty_cache()
    logger.info('xtier: DeepSeek wo_a: %d layers in FP8 (checked, %.2f GiB of BF16 freed)%s', done, freed / 2**30,
                f', layers {kept} kept BF16' if kept else '')


def install(registry, hook_type):
    if not enabled():
        return
    registry.register(APPLY, _apply, hook_type.AROUND)
    if os.environ.get('XTIER_DS_WO_A_FP8', '1') != '0':
        registry.register(STREAMING, _streaming, hook_type.AROUND)
        registry.register(LOAD, _load_weights, hook_type.AROUND)
