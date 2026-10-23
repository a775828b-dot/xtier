"""Checkpoint expert formats: which parameters the checkpoint's own MoE method creates, which of them must be loaded
before the layer can go to the CPU engine, and how they become the engine's inputs (w13 [E, 2I, H/2], w2 [E, H, I/2],
block scales s13 / s2 as bytes, global scales g13 [E, 2] / g2 [E]).
"""

import torch


class Format:
    name = ''          # engine scale format
    required = {}      # parameter -> 'w13' (needs the w1 and w3 shards, or the fused w13) or 'w2', per expert
    derived = ()       # parameters the method derives from the empty tensors at creation (dropped)
    # The required parameters are released once the engine has its copy; small ones (input scales) stay.

    def engine_inputs(self, layer, up_first: bool):
        raise NotImplementedError


def shard_complete(kind: str, shards: set) -> bool:
    return 'w2' in shards if kind == 'w2' else ('w13' in shards or {'w1', 'w3'} <= shards)


def _gate_up(t: torch.Tensor, up_first: bool) -> torch.Tensor:
    """[E, 2I, ...] in [gate; up] row order (the loader stored [up; gate] when the GPU method asked for it)."""
    if not up_first:
        return t.contiguous()
    half = t.shape[1] // 2
    return torch.cat([t[:, half:], t[:, :half]], dim=1).contiguous()


class ModelOptNvFp4(Format):
    """ModelOpt NVFP4 (and W4A16_NVFP4): E2M1 weights, E4M3 scales per 16, FP32 scale per expert and projection."""
    name = 'nvfp4'
    required = {'w13_weight': 'w13', 'w2_weight': 'w2', 'w13_weight_scale': 'w13', 'w2_weight_scale': 'w2',
                'w13_weight_scale_2': 'w13', 'w2_weight_scale_2': 'w2'}
    derived = ('w13_blockscale_swizzled', 'w2_blockscale_swizzled')

    def engine_inputs(self, layer, up_first):
        g13 = layer.w13_weight_scale_2.data.float()   # per-tensor scales always stay [gate, up]
        if g13.dim() == 1:
            g13 = g13[:, None].expand(-1, 2)
        return (_gate_up(layer.w13_weight.data, up_first), layer.w2_weight.data.contiguous(),
                _gate_up(layer.w13_weight_scale.data.view(torch.uint8), up_first),
                layer.w2_weight_scale.data.view(torch.uint8).contiguous(), g13.contiguous(),
                layer.w2_weight_scale_2.data.float().contiguous())


class CompressedTensorsNvFp4(Format):
    """compressed-tensors NVFP4 W4A4 (MiniMax M3.1's routed experts): packed E2M1 weights, E4M3 scales per 16 and an
    FP32 global scale per expert and projection that divides (ModelOpt's scale_2 is its reciprocal). The activations'
    FP4 global scales are not needed: the engine and XTier's GPU kernels multiply BF16 activations."""
    name = 'nvfp4'
    required = {'w13_weight_packed': 'w13', 'w2_weight_packed': 'w2', 'w13_weight_scale': 'w13', 'w2_weight_scale': 'w2',
                'w13_weight_global_scale': 'w13', 'w2_weight_global_scale': 'w2'}

    def engine_inputs(self, layer, up_first):
        g13 = layer.w13_weight_global_scale.data.float().reciprocal()   # per-tensor scales always stay [gate, up]
        if g13.dim() == 1:
            g13 = g13[:, None].expand(-1, 2)
        return (_gate_up(layer.w13_weight_packed.data, up_first), layer.w2_weight_packed.data.contiguous(),
                _gate_up(layer.w13_weight_scale.data.view(torch.uint8), up_first),
                layer.w2_weight_scale.data.view(torch.uint8).contiguous(), g13.contiguous(),
                layer.w2_weight_global_scale.data.float().reciprocal().contiguous())


def _ue8m0_bytes(t: torch.Tensor) -> torch.Tensor:
    """UE8M0 block scales as bytes: SGLang keeps them as float32 powers of two (float8_e8m0fnu with aiter / XPU); the
    exponent field of a float32 2^(e-127) is e, including 2^-127 (exponent field 0)."""
    if t.dtype == torch.uint8:
        return t.contiguous()
    if t.dtype == torch.float8_e8m0fnu:
        return t.view(torch.uint8).contiguous()
    if t.dtype == torch.float32:
        bits = t.contiguous().view(torch.int32)
        if bool(((bits & 0x7FFFFF) != 0).logical_and((bits >> 23) != 0).any()):
            raise ValueError('xtier: an MXFP4 block scale is not a power of two')
        return ((bits >> 23) & 0xFF).to(torch.uint8)
    raise TypeError(f'xtier: MXFP4 block scales of dtype {t.dtype}')


class Fp8Fp4Experts(Format):
    """DeepSeek V4 MXFP4 experts (quantization config fp8, expert_dtype fp4): E2M1 weights, UE8M0 scales per 32, no
    global scales."""
    name = 'mxfp4'
    required = {'w13_weight': 'w13', 'w2_weight': 'w2', 'w13_weight_scale_inv': 'w13', 'w2_weight_scale_inv': 'w2'}

    def engine_inputs(self, layer, up_first):
        E = layer.w13_weight.data.shape[0]
        return (_gate_up(layer.w13_weight.data.view(torch.uint8), up_first), layer.w2_weight.data.view(torch.uint8).contiguous(),
                _gate_up(_ue8m0_bytes(layer.w13_weight_scale_inv.data), up_first),
                _ue8m0_bytes(layer.w2_weight_scale_inv.data), torch.ones((E, 2), dtype=torch.float32),
                torch.ones((E,), dtype=torch.float32))


def detect(gpu_method) -> Format:
    name = type(gpu_method).__name__
    if name == 'ModelOptNvFp4FusedMoEMethod':
        return ModelOptNvFp4()
    if name == 'Fp8MoEMethod' and getattr(gpu_method, 'is_fp4_expert', False):
        return Fp8Fp4Experts()
    if name == 'Mxfp4FlashinferCutlassMoEMethod' and getattr(getattr(gpu_method, '_fp8', None), 'is_fp4_expert', False):
        return Fp8Fp4Experts()   # SGLang's SM90 / SM120 choice for these experts: Fp8MoEMethod's parameters, E8M0 scales
    if name == 'CompressedTensorsFusedMoEMethod':
        return CompressedTensorsNvFp4()   # the scheme sits on the layer: checked once the parameters exist
    raise NotImplementedError(f'xtier: no CPU engine format for the MoE method {name}')
