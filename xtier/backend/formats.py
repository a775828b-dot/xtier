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


def detect(gpu_method) -> Format:
    name = type(gpu_method).__name__
    if name == 'ModelOptNvFp4FusedMoEMethod':
        return ModelOptNvFp4()
    raise NotImplementedError(f'xtier: no CPU engine format for the MoE method {name}')
