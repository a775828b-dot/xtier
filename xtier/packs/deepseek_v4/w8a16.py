"""DeepSeek V4.1's block FP8 projections below SM89 (RTX 30 series): FP8 weights x BF16 activations.

SGLang serves 32 x 32 block FP8 linears off SM90 with its Triton W8A8 kernel (activations quantized to FP8, float8e4nv:
SM89 and newer). Below SM89 its FP8 linear method takes Marlin's weight-only kernel instead, whose FP8 variants exist
for per-channel scales and 128-wide groups only: a 32-wide group finds no thread config (the first forward fails).
Here such layers keep SGLang's own layout (no Marlin repacking) and run weight-only: up to wo_a.MAX_T rows on the
pack's FP8 einsum (wo_a.py with one group: FP8 x block scale formed in BF16, FP32 accumulation), larger calls on the
weight dequantized into a reused BF16 buffer and cuBLAS. The FP8 bytes decode exactly (xtier.arch.e4m3_f32) and the
activations stay BF16, where production quantizes them to FP8.
"""
import torch

from xtier.packs.deepseek_v4 import wo_a

FP8_METHOD = 'sglang.srt.layers.quantization.fp8.Fp8LinearMethod'
BLOCK_LINEAR = 'sglang.srt.layers.quantization.fp8_utils.triton_w8a8_block_fp8_linear'
BLOCK = [32, 32]
_buffers = {}   # device -> BF16 buffer of the largest dequantized weight so far
_scales = {}    # (data_ptr, shape) of non-FP32 block scales -> their FP32 copy


def no_marlin(result, self, *args, **kwargs):
    """AFTER hook of Fp8LinearMethod.__init__ (registered below SM89): 32 x 32 block layers stay off Marlin."""
    from sglang.srt.utils import get_bool_env_var
    if (self.use_marlin and self.block_quant and list(self.weight_block_size or ()) == BLOCK
            and not get_bool_env_var('SGLANG_FORCE_FP8_MARLIN')):
        self.use_marlin = False
    return result


def _fp32(scale):
    if scale.dtype == torch.float32:
        return scale
    key = (scale.data_ptr(), tuple(scale.shape))
    if key not in _scales:
        _scales[key] = scale.float()
    return _scales[key]


def _buffer(device, numel):
    buf = _buffers.get(device)
    if buf is None or buf.numel() < numel:
        buf = _buffers[device] = torch.empty(numel, dtype=torch.bfloat16, device=device)
    return buf[:numel]


def linear(original, input, weight, block_size, weight_scale, input_scale=None, bias=None, *args, **kwargs):
    """AROUND hook of SGLang's triton_w8a8_block_fp8_linear (registered below SM89)."""
    n, k = weight.shape
    if (input_scale is not None or list(block_size) != BLOCK or weight.dtype != torch.float8_e4m3fn
            or input.dtype != torch.bfloat16 or not weight.is_contiguous() or n % 32 or k % 32):
        return original(input, weight, block_size, weight_scale, input_scale, bias, *args, **kwargs)
    x = input.reshape(-1, k)
    scale = _fp32(weight_scale)
    if x.shape[0] <= wo_a.MAX_T:
        out = wo_a.einsum(x.unsqueeze(1), weight.view(1, n, k), scale).view(-1, n)
    else:
        out = torch.nn.functional.linear(x, wo_a.dequantize(weight, scale, _buffer(weight.device, n * k).view(n, k)))
    if bias is not None:
        out = out + bias
    return out.view(*input.shape[:-1], n)
