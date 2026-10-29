"""DeepSeek V4.1 Engram rows below SM89 (RTX 30 series): SGLang's engram_gather with the FP8 payload decoded in software.

SGLang's Triton gather of Engram rows (FP8 E4M3 payload, E8M0 block scales; also the lookup of XTier's NVMe row table)
loads the payload through a float8e4nv pointer, which Triton cannot type below SM89. This is the same kernel with the
payload read as bytes and decoded by xtier.arch.e4m3_f32 (bit for bit the hardware conversion,
tests/fp8_decode_check.py); the arithmetic is SGLang's: fp32(row) * 2^(exp - 127), rounded to BF16 once.
"""
import torch
import triton
import triton.language as tl

from xtier.arch import e4m3_f32

ENGRAM_GATHER = 'sglang.kernels.ops.embeddings.engram_gather.engram_gather'


@triton.jit
def _gather(w_ptr, s_ptr, ids_ptr, out_ptr, row_lo, row_hi, DIM: tl.constexpr, BLK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    owned = (idx >= row_lo) & (idx < row_hi)   # ids outside the table's rows come out as zeros
    local = tl.where(owned, idx - row_lo, 0)
    w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    s = s_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    offs = tl.arange(0, DIM)
    vals = e4m3_f32(tl.load(w + local * DIM + offs, mask=owned, other=0))
    exps = tl.load(s + local * (DIM // BLK) + offs // BLK, mask=owned, other=0).to(tl.int32)
    scale = tl.where(exps == 0, 0x00400000, exps << 23).to(tl.float32, bitcast=True)   # E8M0: 2^(e - 127)
    tl.store(out_ptr + row * DIM + offs, tl.where(owned, vals * scale, 0.0).to(tl.bfloat16))


def engram_gather(weight_ptr: int, scale_ptr: int, ids: torch.Tensor, out: torch.Tensor, dim: int, block_size: int,
                  row_lo: int = 0, row_hi: int = 2**62) -> torch.Tensor:
    """SGLang's engram_gather: rows ``ids`` of the FP8 table at ``weight_ptr`` (E8M0 scales at ``scale_ptr``, global
    rows [row_lo, row_hi)) into ``out`` [N, dim] BF16."""
    assert dim > 0 and dim & (dim - 1) == 0 and block_size > 0 and dim % block_size == 0, (dim, block_size)
    assert ids.is_cuda and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)
    assert out.is_contiguous() and out.dtype == torch.bfloat16
    assert out.device == ids.device and out.shape == (ids.numel(), dim), out.shape
    if ids.numel():
        _gather[(ids.numel(),)](weight_ptr, scale_ptr, ids, out, row_lo, row_hi, DIM=dim, BLK=block_size)
    return out


def replace(original, *args, **kwargs):
    """AROUND hook of SGLang's engram_gather (registered below SM89)."""
    return engram_gather(*args, **kwargs)
