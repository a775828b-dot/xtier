"""Checkpoint expert formats: which parameters the checkpoint's own MoE method creates, which of them must be loaded
before the layer can go to the CPU engine, and how they become the engine's inputs (w13 [E, 2I, H/2], w2 [E, H, I/2],
block scales s13 / s2 as bytes, global scales g13 [E, 2] / g2 [E]).
"""

import logging
import time

import torch

logger = logging.getLogger(__name__)


class Format:
    name = ''          # engine scale format
    required = {}      # parameter -> 'w13' (needs the w1 and w3 shards, or the fused w13) or 'w2', per expert
    bulk = ()          # the parameters behind w13, w2, s13, s2: bytes as the checkpoint stores them (direct.py)
    derived = ()       # parameters the method derives from the empty tensors at creation (dropped)
    # The required parameters are released once the engine has its copy; small ones (input scales) stay.

    def engine_inputs(self, layer, up_first: bool):
        raise NotImplementedError

    def globals(self, layer):
        """g13 [E, 2] and g2 [E] (float32), without touching the bulk parameters."""
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
    bulk = ('w13_weight', 'w2_weight', 'w13_weight_scale', 'w2_weight_scale')
    derived = ('w13_blockscale_swizzled', 'w2_blockscale_swizzled')

    def engine_inputs(self, layer, up_first):
        return (_gate_up(layer.w13_weight.data, up_first), layer.w2_weight.data.contiguous(),
                _gate_up(layer.w13_weight_scale.data.view(torch.uint8), up_first),
                layer.w2_weight_scale.data.view(torch.uint8).contiguous()) + self.globals(layer)

    def globals(self, layer):
        g13 = layer.w13_weight_scale_2.data.float()   # per-tensor scales always stay [gate, up]
        if g13.dim() == 1:
            g13 = g13[:, None].expand(-1, 2)
        return g13.contiguous(), layer.w2_weight_scale_2.data.float().contiguous()


class CompressedTensorsNvFp4(Format):
    """compressed-tensors NVFP4 W4A4 (MiniMax M3.1's routed experts): packed E2M1 weights, E4M3 scales per 16 and an
    FP32 global scale per expert and projection that divides (ModelOpt's scale_2 is its reciprocal). The activations'
    FP4 global scales are not needed: the engine and XTier's GPU kernels multiply BF16 activations."""
    name = 'nvfp4'
    required = {'w13_weight_packed': 'w13', 'w2_weight_packed': 'w2', 'w13_weight_scale': 'w13', 'w2_weight_scale': 'w2',
                'w13_weight_global_scale': 'w13', 'w2_weight_global_scale': 'w2'}
    bulk = ('w13_weight_packed', 'w2_weight_packed', 'w13_weight_scale', 'w2_weight_scale')

    def engine_inputs(self, layer, up_first):
        return (_gate_up(layer.w13_weight_packed.data, up_first), layer.w2_weight_packed.data.contiguous(),
                _gate_up(layer.w13_weight_scale.data.view(torch.uint8), up_first),
                layer.w2_weight_scale.data.view(torch.uint8).contiguous()) + self.globals(layer)

    def globals(self, layer):
        g13 = layer.w13_weight_global_scale.data.float().reciprocal()   # per-tensor scales always stay [gate, up]
        if g13.dim() == 1:
            g13 = g13[:, None].expand(-1, 2)
        return g13.contiguous(), layer.w2_weight_global_scale.data.float().reciprocal().contiguous()


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
    bulk = ('w13_weight', 'w2_weight', 'w13_weight_scale_inv', 'w2_weight_scale_inv')

    def engine_inputs(self, layer, up_first):
        return (_gate_up(layer.w13_weight.data.view(torch.uint8), up_first), layer.w2_weight.data.view(torch.uint8).contiguous(),
                _gate_up(_ue8m0_bytes(layer.w13_weight_scale_inv.data), up_first),
                _ue8m0_bytes(layer.w2_weight_scale_inv.data)) + self.globals(layer)

    def globals(self, layer):
        E = layer.w13_weight.data.shape[0]
        return torch.ones((E, 2), dtype=torch.float32), torch.ones((E,), dtype=torch.float32)


E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)   # midpoints of the E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6


def nvfp4_encode(w: torch.Tensor):
    """ModelOpt NVFP4 of a matrix w [rows, K], K a multiple of 16: (E2M1 pairs [rows, K/2], the even element in the low
    nibble; E4M3 block scales [rows, K/16] as bytes; the FP32 global scale amax / (6 * 448)), w ~ e2m1 x scale x global.
    Rounding to the nearest E2M1 value of w / (scale x global)."""
    w = w.float()
    g = (w.abs().amax() / (6 * 448)).clamp_min(1e-12)
    blocks = w.unflatten(-1, (-1, 16))
    s = (blocks.abs().amax(-1) / (6 * g)).to(torch.float8_e4m3fn)   # at most 448: the global scale's choice
    q = (blocks / (s.float() * g).clamp_min(1e-30).unsqueeze(-1)).flatten(-2)   # a zero scale dequantizes to 0
    code = torch.bucketize(q.abs().clamp(max=6.0), torch.tensor(E2M1_BOUNDS, device=w.device)).to(torch.uint8)
    code |= ((q < 0) & (code > 0)).to(torch.uint8) << 3
    return code[:, 0::2] | (code[:, 1::2] << 4), s.view(torch.uint8), g


class Bf16AsNvFp4(Format):
    """A draft model's BF16 experts (GLM-5.3's NextN layer in NVIDIA's ModelOpt checkpoint, excluded from its NVFP4)
    quantized to ModelOpt NVFP4 at the hand-off, so the CPU engine runs them as it runs the target's: the draft's
    precision only, the target model verifies every token the draft proposes. Per expert on the GPU when there is one;
    a global scale per expert and projection."""
    name = 'nvfp4'
    required = {'w13_weight': 'w13', 'w2_weight': 'w2'}

    def engine_inputs(self, layer, up_first):
        t0 = time.perf_counter()
        w13, w2 = layer.w13_weight.data, layer.w2_weight.data   # [E, 2I, H], [E, H, I] BF16
        E, I2, H = w13.shape
        dev = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else w13.device
        q13 = torch.empty((E, I2, H // 2), dtype=torch.uint8)
        s13 = torch.empty((E, I2, H // 16), dtype=torch.uint8)
        q2 = torch.empty((E, H, I2 // 4), dtype=torch.uint8)
        s2 = torch.empty((E, H, I2 // 32), dtype=torch.uint8)
        g13 = torch.empty((E, 2), dtype=torch.float32)
        g2 = torch.empty((E,), dtype=torch.float32)
        for e in range(E):
            m = _gate_up(w13[e:e + 1], up_first)[0].to(dev)
            for j in range(2):   # gate rows, up rows: a global scale each
                rows = slice(j * I2 // 2, (j + 1) * I2 // 2)
                q, s, g = nvfp4_encode(m[rows])
                q13[e, rows], s13[e, rows], g13[e, j] = q.cpu(), s.cpu(), g.item()
            q, s, g = nvfp4_encode(w2[e].to(dev))
            q2[e], s2[e], g2[e] = q.cpu(), s.cpu(), g.item()
        logger.info('xtier: %d BF16 draft experts quantized to NVFP4 for the CPU engine in %.1f s (the draft\'s '
                    'precision only: the target model verifies every token)', E, time.perf_counter() - t0)
        return q13, q2, s13, s2, g13, g2


def detect(gpu_method, draft=False) -> Format:
    name = type(gpu_method).__name__
    if name == 'ModelOptNvFp4FusedMoEMethod':
        return ModelOptNvFp4()
    if name == 'Fp8MoEMethod' and getattr(gpu_method, 'is_fp4_expert', False):
        return Fp8Fp4Experts()
    if name == 'Mxfp4FlashinferCutlassMoEMethod' and getattr(getattr(gpu_method, '_fp8', None), 'is_fp4_expert', False):
        return Fp8Fp4Experts()   # SGLang's SM90 / SM120 choice for these experts: Fp8MoEMethod's parameters, E8M0 scales
    if name == 'CompressedTensorsFusedMoEMethod':
        return CompressedTensorsNvFp4()   # the scheme sits on the layer: checked once the parameters exist
    if name == 'UnquantizedFusedMoEMethod' and draft:
        return Bf16AsNvFp4()   # a target model's BF16 experts stay with SGLang: no precision change there
    raise NotImplementedError(f'xtier: no CPU engine format for the MoE method {name}')
