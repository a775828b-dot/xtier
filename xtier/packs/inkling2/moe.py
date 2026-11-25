"""XTier hybrid CPU+GPU MoE backend for Thinking Machines Inkling-2 (``Inkling2Experts``).

Inkling-2-Small has 48 MoE layers (384 experts, top-8, H=4608, I=2304, soft-capped SiTU(4, 25)):
- Layers 0..39: NVFP4 (uint8 packed weights, E4M3FN scales / 16, per-expert ``amax / outer_scale_denominator``)
- Layers 40..47: MXFP8 (float8_e4m3fn weights, UE8M0 scales / 32, unit global scales)
- Column-interleaved ``w13`` (even rows = gate, odd rows = up), deinterleaved to XTier's ``[gate; up]`` on load.

Each layer registers into ``XTierMoEMethod.layers`` so NUMA localization, pinned shard registration, hot-pool fill,
adaptive swapping, elastic VRAM resizing, multi-format NVMe cold storage, and DSpark decode-path token skipping
work identically to FusedMoE models.
"""

from functools import partial
import gc
import logging
import os
import re
import threading
import time

import torch
from torch import nn

from xtier import config, planner, platform
from xtier.backend import kernels
from xtier.backend.method import (
    DECODE_ROWS,
    MAX_TOP_K,
    XTierMoEMethod,
    slot_width,
)
from xtier.backend.prefill import Prefill
from xtier.cache import tail
from xtier.engine import CpuPlan, Mailbox, placement

logger = logging.getLogger(__name__)

EXPERTS_CLS = 'sglang.srt.models.inkling_common.inkling2_moe.Inkling2Experts'
MODEL_CLS = 'sglang.srt.models.inkling2.Inkling2ForConditionalGeneration'
_LAYER_RE = re.compile(r'\.layers\.(\d+)\.')


def _layer_id_from_prefix(prefix: str) -> int | None:
    m = _LAYER_RE.search(prefix)
    return int(m.group(1)) if m else None


class Inkling2LayerBackend:
    """Per-layer XTier MoE runtime attached to an ``Inkling2Experts`` module."""

    _hand_off_lock = threading.Lock()

    def __init__(self, cfg, layer_id, experts, hidden, inter, scale_format, gate_cap, up_cap, top_k):
        self.cfg = cfg
        self.layer_id = int(layer_id)
        self.experts = int(experts)
        self.hidden = int(hidden)
        self.inter = int(inter)
        self.scale_format = scale_format
        self.gate_cap = float(gate_cap)
        self.up_cap = float(up_cap)
        self.top_k = int(top_k)
        self.nvfp4 = scale_format == 'nvfp4'

        self.expected = (
            'w13_weight', 'w13_scale', 'w2_weight', 'w2_scale',
            'w13_amax', 'w13_outer_scale_denominator', 'w2_amax', 'w2_outer_scale_denominator',
        ) if self.nvfp4 else (
            'w13_weight', 'w13_scale', 'w2_weight', 'w2_scale',
        )
        self._staged = {}
        self._received = set()
        self._source = None
        self._cold_plan = None
        self._cold_max_width = None
        self._tail_checked = False

        self.plan = None
        self.prefill = None
        self.mailbox = None
        self.pool = None
        self.calls = dict(decode=0, cpu_prefill=0, gpu_prefill=0, hybrid_prefill=0, split_prefill=0)
        self.last_split = None

    def _choose_source(self):
        from xtier.cache import cold

        if (cold.enabled() and not self.cfg.draft
                and 0 < self.cfg.cache_slots < self.experts
                and os.environ.get('XTIER_PREFILL_FROM_SHARDS', '1') == '1'):
            self._cold_plan = cold.LayerPlan(self.layer_id, self.experts, self.cfg.cache_slots)
            if cold.load_enabled() and cold.loadable(
                self._cold_plan, self.hidden, self.inter, self.scale_format, self.cfg.threads
            ):
                return 'store'
        if not self.cfg.draft:
            cold.require_checkpoint(self.layer_id)
        return 'copy'

    def load_tensor(self, name, param, loaded):
        if self._source is None:
            self._source = self._choose_source()
        if self._source == 'store' and name in ('w13_weight', 'w2_weight', 'w13_scale', 'w2_scale'):
            # Lazy safetensors mmap view: never touch or fault its pages when reloading from the cold store.
            self._received.add(name)
        else:
            if name in ('w13_weight', 'w2_weight'):
                platform.bulk_copy()
            if name in ('w13_weight', 'w13_scale'):
                # Deinterleave [E, 2*I, K] (even=gate, odd=up) directly from mmap into [gate; up] uint8.
                raw = loaded.view(torch.uint8)
                if raw.device.type != 'cpu':
                    raw = raw.cpu()
                I = self.inter
                out = torch.empty(raw.shape, dtype=torch.uint8, device='cpu')
                out[:, :I, :] = raw[:, 0::2, :]
                out[:, I:, :] = raw[:, 1::2, :]
                self._staged[name] = out
            elif name in ('w2_weight', 'w2_scale'):
                raw = loaded.view(torch.uint8)
                self._staged[name] = raw.cpu().contiguous() if raw.device.type != 'cpu' else raw.contiguous().clone()
            else:
                t = loaded.to(dtype=torch.float32, device='cpu').contiguous().clone()
                self._staged[name] = t
            self._received.add(name)

        if len(self._received) == len(self.expected):
            with Inkling2LayerBackend._hand_off_lock:
                self._finalize()

    def _build_globals(self):
        E = self.experts
        if not self.nvfp4:
            return torch.ones((E, 2), dtype=torch.float32), torch.ones((E,), dtype=torch.float32)
        w13_amax = self._staged['w13_amax'].reshape(E)
        w13_den = float(self._staged['w13_outer_scale_denominator'].reshape(-1)[0].item())
        w2_amax = self._staged['w2_amax'].reshape(E)
        w2_den = float(self._staged['w2_outer_scale_denominator'].reshape(-1)[0].item())
        g13_1d = (w13_amax / w13_den).clamp_min(1e-30).to(torch.float32)
        g13 = torch.stack([g13_1d, g13_1d], dim=1).contiguous()
        g2 = (w2_amax / w2_den).clamp_min(1e-30).to(torch.float32).contiguous()
        return g13, g2

    def _finalize(self):
        from xtier.cache import cold

        source = self._source or 'copy'
        g13, g2 = self._build_globals()
        if source == 'copy':
            inputs = (
                self._staged['w13_weight'],
                self._staged['w2_weight'],
                self._staged['w13_scale'],
                self._staged['w2_scale'],
                g13,
                g2,
            )
        else:
            inputs = (None, None, None, None, g13, g2)
        self._staged.clear()

        placement.apply(self.cfg.threads, 0, 1)
        t0 = time.perf_counter()
        slots = self.experts if self.cfg.cache_slots < 0 else self.cfg.cache_slots
        full_gpu = (not cold.enabled()) and slots >= self.experts and source == 'copy'
        if full_gpu:
            # Zero-CPU-RAM full-GPU layer (Plan B / XTIER_CACHE_FULL with XTIER_COLD=0):
            # upload directly from staged CPU tensors into LayerPool.arena and drop host copies immediately.
            from xtier.cache.pool import LayerPool

            self.plan = None
            self.mailbox = None
            self.prefill = Prefill(
                self.experts,
                (None, None, None, None, g13.data_ptr(), g2.data_ptr()),
                native=None,
                hidden=self.hidden,
                inter=self.inter,
                swiglu_limit=0.0,
                situ_gate_cap=self.gate_cap,
                situ_up_cap=self.up_cap,
                scale_format=self.scale_format,
            )
            self.prefill.order = self.layer_id
            self.pool = LayerPool(self.prefill, self.experts, None, None, self.layer_id)
            self.pool.tp_size = 1
            self.prefill._upload_chunk = self._upload_from_cache
            dev = self.pool.arena[0].device
            for dst, src in zip(self.pool.arena, inputs[:4]):
                dst.copy_(src, non_blocking=False)
            self.pool.table.copy_(torch.arange(self.experts, dtype=torch.int32, device=dev))
            self.pool.owner.copy_(torch.arange(self.experts, dtype=torch.int32, device=dev))
            self.pool.filled = True
            del inputs
            gc.collect()
            XTierMoEMethod.layers.append(self)
            logger.info(
                'xtier: inkling2 layer %d: %d %s experts (H=%d, I=%d, SiTU=%.1f/%.1f), full-GPU zero-CPU-RAM (%.2f GiB VRAM) in %.2f s',
                self.layer_id,
                self.experts,
                self.scale_format,
                self.hidden,
                self.inter,
                self.gate_cap,
                self.up_cap,
                sum(a.numel() for a in self.pool.arena) / (1 << 30),
                time.perf_counter() - t0,
            )
            return

        pointers = [None if t is None else t.data_ptr() for t in inputs]
        lp = None
        if cold.enabled() and not self.cfg.draft and 0 < self.cfg.cache_slots < self.experts:
            lp = self._cold_plan or cold.LayerPlan(self.layer_id, self.experts, self.cfg.cache_slots)

        self.plan = CpuPlan(
            self.experts,
            self.cfg.threads,
            pointers,
            hidden=self.hidden,
            inter=self.inter,
            swiglu_limit=0.0,
            scale_format=self.scale_format,
            resident=lp.resident if lp else None,
            situ_gate_cap=self.gate_cap,
            situ_up_cap=self.up_cap,
        )
        how = ''
        if source == 'store':
            prefill_cold = cold.attach_loaded(self.plan, lp, self.scale_format)
            how = ', being read from the cold store'
        else:
            prefill_cold = cold.attach(self.plan, lp, inputs, self.scale_format) if lp else None
        if lp is not None:
            w_def = slot_width(self.top_k)
            min_w = min(16, w_def)
            min_need = min(self.experts, min_w * self.top_k)
            if self.plan.slots < min_need:
                raise RuntimeError(
                    f'xtier cold: layer {self.layer_id} keeps {self.plan.slots} experts in RAM, a decode call of '
                    f'{min_w} tokens can route to {min_need}: lower XTIER_COLD_FRACTION'
                )
            if self.plan.slots < min(self.experts, w_def * self.top_k):
                self._cold_max_width = max(1, self.plan.slots // self.top_k)

        self.prefill = Prefill(
            self.experts,
            pointers,
            native=self.plan,
            hidden=self.hidden,
            inter=self.inter,
            swiglu_limit=0.0,
            situ_gate_cap=self.gate_cap,
            situ_up_cap=self.up_cap,
        )
        self.prefill.order = self.layer_id
        self.prefill.cold = prefill_cold
        del inputs, pointers
        gc.collect()

        self.mailbox = Mailbox(self.plan)
        self.pool = None
        slots = self.experts if self.cfg.cache_slots < 0 else self.cfg.cache_slots
        if slots > 0:
            from xtier.cache.pool import LayerPool

            self.pool = LayerPool(self.prefill, slots, self.plan, self.mailbox, self.layer_id)
            self.pool.tp_size = 1
            if self.pool.full:
                self.prefill._upload_chunk = self._upload_from_cache
        self.plan.pool = self.pool
        if prefill_cold is not None:
            prefill_cold.pool = self.pool
        XTierMoEMethod.layers.append(self)
        logger.info(
            'xtier: inkling2 layer %d: %d %s experts (H=%d, I=%d, SiTU=%.1f/%.1f), %d threads, %s cache slots in %.2f s%s',
            self.layer_id,
            self.experts,
            self.scale_format,
            self.hidden,
            self.inter,
            self.gate_cap,
            self.up_cap,
            self.cfg.threads,
            'all' if self.pool and self.pool.full else (self.pool.S if self.pool else 0),
            time.perf_counter() - t0,
            how,
        )

    def _upload_from_cache(self, slot, e0, e1, stream, up_first=False, virtual=True, indices=None):
        if up_first:
            raise NotImplementedError('xtier: FP4 prefill of a fully cached layer')
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            for i, (dst, src) in enumerate(zip(slot, self.pool.arena)):
                if indices is None or i in indices:
                    dst[:e1 - e0].copy_(src[e0:e1], non_blocking=True)

    def _decode(self, x, ids, weights, m, k):
        H = self.hidden
        if m > DECODE_ROWS:
            raise ValueError(f'xtier: a {m}-token call on the decode path (decode buffer: {DECODE_ROWS} tokens)')
        out = torch.empty((m, H), dtype=torch.bfloat16, device=x.device)
        stream = torch.cuda.current_stream().cuda_stream
        active = XTierMoEMethod._active
        width = min(slot_width(k), self._cold_max_width) if self._cold_max_width else slot_width(k)
        for start in range(0, m, width):
            n = min(width, m - start)
            xs = x.data_ptr() + start * H * 2
            ids_s = ids.data_ptr() + start * k * 4
            ws = weights.data_ptr() + start * k * 4
            act = active.data_ptr() + start * 4 if active is not None else 0
            out_ptr = out.data_ptr() + start * H * 2
            if self.pool is not None:
                self.pool.decode(stream, n, k, xs, ids_s, ws, out_ptr, act, out_bf16=True, scale=1.0)
                continue
            slot = self.mailbox.slot(stream, n, k)
            kernels.pack(slot, n, k, xs, ids_s, ws, H, act)
            self.mailbox.notify(stream, n, k)
            kernels.unpack(slot, n, out_ptr, H, out_bf16=True, scale=1.0)
        return out

    def _cpu_prefill(self, x, ids, weights, m, k):
        H = self.hidden
        x_h = torch.empty((m, H), dtype=torch.bfloat16, pin_memory=True)
        ids_h = torch.empty((m, k), dtype=torch.int32, pin_memory=True)
        w_h = torch.empty((m, k), dtype=torch.float32, pin_memory=True)
        out_h = torch.empty((m, H), dtype=torch.float32, pin_memory=True)
        x_h.copy_(x, non_blocking=True)
        ids_h.copy_(ids, non_blocking=True)
        w_h.copy_(weights, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        batch = config.cpu_prefill_batch()
        for start in range(0, m, batch):
            n = min(batch, m - start)
            self.plan.forward(
                n, k,
                x_h.data_ptr() + start * H * 2,
                ids_h.data_ptr() + start * k * 4,
                w_h.data_ptr() + start * k * 4,
                out_h.data_ptr() + start * H * 4,
            )
        return out_h.to(x.device, non_blocking=True)

    def forward(self, x, weights, ids, combine=True):
        from sglang.srt.models.inkling_common.inkling2_moe import RoutedOutput

        m = x.shape[0]
        if m == 0:
            out = x.new_zeros(x.shape)
            return out if combine else RoutedOutput(out, None, 0)
        if x.dtype != torch.bfloat16:
            raise TypeError(f'xtier: BF16 activations required, got {x.dtype}')
        if not XTierMoEMethod._localized:
            XTierMoEMethod._localize_all()
        ids_i = ids if ids.dtype == torch.int32 else ids.to(torch.int32)
        w_f = weights if weights.dtype == torch.float32 else weights.float()
        x_c, ids_c, w_c = x.contiguous(), ids_i.contiguous(), w_f.contiguous()
        k = ids_c.shape[1]
        if not 1 <= k <= MAX_TOP_K:
            raise ValueError(f'xtier: top-k {k} not supported')
        if not self._tail_checked:
            if not self.cfg.draft:
                tail.check(k)
            self._tail_checked = True

        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing and XTierMoEMethod._registered < len(XTierMoEMethod.layers):
            XTierMoEMethod._register_all()
        if not capturing and m > config.decode_max() and tail.eval_rank() and not self.cfg.draft:
            w_c = tail.prefill_mask(self.pool, ids_c, w_c, tail.eval_rank())
        if not capturing and m > config.decode_max() and self.pool is not None and not self.pool.full:
            from xtier.cache import pool as cache_pool

            cache_pool.record_prompt(self.pool, ids_c)

        timed = config.timing() and not capturing and m > config.decode_max()
        if timed:
            torch.cuda.current_stream().synchronize()
            t0 = time.perf_counter()

        if m <= config.decode_max() or capturing:
            if not capturing:
                XTierMoEMethod._cache_start()
            out = self._decode(x_c, ids_c, w_c, m, k)
            self.calls['decode'] += 1
        elif self.pool is not None and self.pool.full and config.resident_prefill():
            from xtier.cache.prefill import resident_prefill

            out = resident_prefill(self.prefill, self.pool.arena, x_c, ids_c, w_c)
            self.calls['gpu_prefill'] += 1
        elif self.pool is not None and not self.pool.full and m < config.hybrid_prefill_max():
            from xtier.cache.prefill import hybrid_prefill

            XTierMoEMethod._cache_start()
            out = hybrid_prefill(self.pool, self.plan, x_c, ids_c, w_c, config.cpu_prefill_batch()).to(x.dtype)
            self.calls['hybrid_prefill'] += 1
        elif 0 < config.gpu_prefill_min() <= m or (self.pool is not None and self.pool.full):
            if m >= 16384 and (self.scale_format == 'mxfp8' or self.layer_id == max(b.layer_id for b in XTierMoEMethod.layers)):
                try:
                    from sglang.kernels.ops.attention.flash_attn.cute.interface import clear_shear_bias_workspace
                    clear_shear_bias_workspace()
                except Exception:
                    pass
            XTierMoEMethod._cache_start()
            out = torch.empty((m, self.hidden), dtype=torch.bfloat16, device=x.device)
            self.prefill.run(
                x_c.data_ptr(), out.data_ptr(), ids_c.data_ptr(), w_c.data_ptr(),
                m, k, torch.cuda.current_stream().cuda_stream,
            )
            self.calls['gpu_prefill'] += 1
        else:
            out = self._cpu_prefill(x_c, ids_c, w_c, m, k).to(x.dtype)
            self.calls['cpu_prefill'] += 1

        if timed:
            torch.cuda.current_stream().synchronize()
            XTierMoEMethod._time(self, m, time.perf_counter() - t0)
        return out if combine else RoutedOutput(out, None, 0)


def _experts_init(original, self, config_obj, num_experts, checkpoint_prefix, quant_config):
    from sglang.srt.layers.utils.common import set_weight_attrs
    from sglang.srt.runtime_context import get_parallel

    if not config.enabled() or quant_config is None:
        return original(self, config_obj, num_experts, checkpoint_prefix, quant_config)
    parallel = get_parallel()
    layer_id = _layer_id_from_prefix(checkpoint_prefix)
    if parallel.moe_ep_size != 1 or parallel.moe_tp_size != 1 or layer_id is None:
        return original(self, config_obj, num_experts, checkpoint_prefix, quant_config)
    quantize_w13 = quant_config.quantizes_input(checkpoint_prefix + 'w13_weight')
    quantize_w2 = quant_config.quantizes_input(checkpoint_prefix + 'w2_weight')
    if not (quantize_w13 and quantize_w2):
        return original(self, config_obj, num_experts, checkpoint_prefix, quant_config)
    cfg = planner.layer_config(layer_id)
    if cfg is None:
        return original(self, config_obj, num_experts, checkpoint_prefix, quant_config)

    nvfp4_w13 = (checkpoint_prefix + 'w13_weight') in quant_config.nvfp4_weights
    nvfp4_w2 = (checkpoint_prefix + 'w2_weight') in quant_config.nvfp4_weights
    if nvfp4_w13 != nvfp4_w2:
        raise ValueError(f'xtier: mixed NVFP4/MXFP8 inside layer {layer_id} is not supported')
    scale_format = 'nvfp4' if nvfp4_w13 else 'mxfp8'

    nn.Module.__init__(self)
    self.tp_rank, self.tp_size = 0, 1
    self.ep_rank, self.ep_size = 0, 1
    self.num_global_experts = num_experts
    self.quantize_w13 = True
    self.quantize_w2 = True
    self._nvfp4_scale_shapes = {}
    self.gate_cap, self.up_cap = config_obj.mlp_gate_softcap, config_obj.mlp_up_softcap
    self.sm90 = False
    self.output_sizes = {'w13': 2 * config_obj.intermediate_size, 'w2': config_obj.hidden_size}
    self._marlin = {}
    self.top_k = config_obj.num_experts_per_tok
    self.route_scale = config_obj.route_scale
    self.mega = None
    self.w13_cutlass_scale = None
    self.w2_cutlass_scale = None
    self._routed_kernels = None
    self._lacks_routed_kernels = False

    backend = Inkling2LayerBackend(
        cfg, layer_id, num_experts, config_obj.hidden_size, config_obj.intermediate_size,
        scale_format, config_obj.mlp_gate_softcap, config_obj.mlp_up_softcap, config_obj.num_experts_per_tok,
    )
    self._xtier = backend

    dev = torch.get_default_device()
    w_dtype = torch.uint8 if nvfp4_w13 else torch.float8_e4m3fn
    s_dtype = torch.float8_e4m3fn if nvfp4_w13 else torch.uint8
    for name in ('w13', 'w2'):
        w_param = nn.Parameter(torch.empty(0, dtype=w_dtype, device=dev), requires_grad=False)
        self.register_parameter(f'{name}_weight', w_param)
        set_weight_attrs(w_param, {'weight_loader': partial(backend.load_tensor, f'{name}_weight')})

        s_param = nn.Parameter(torch.empty(0, dtype=s_dtype, device=dev), requires_grad=False)
        self.register_parameter(f'{name}_scale', s_param)
        set_weight_attrs(s_param, {'weight_loader': partial(backend.load_tensor, f'{name}_scale')})

        if nvfp4_w13:
            for suffix in ('amax', 'outer_scale_denominator'):
                p = nn.Parameter(torch.empty(0, dtype=torch.float32, device=dev), requires_grad=False)
                self.register_parameter(f'{name}_{suffix}', p)
                set_weight_attrs(p, {'weight_loader': partial(backend.load_tensor, f'{name}_{suffix}')})


def _noop_if_xtier(original, self, *args, **kwargs):
    if getattr(self, '_xtier', None) is not None:
        return None
    return original(self, *args, **kwargs)


def _experts_forward(original, self, x, weights, ids, combine=True):
    backend = getattr(self, '_xtier', None)
    if backend is not None:
        return backend.forward(x, weights, ids, combine=combine)
    return original(self, x, weights, ids, combine=combine)


def _after_load_weights(result, self, *args, **kwargs):
    from sglang.srt.models.inkling_common.inkling2_moe import Inkling2Experts

    experts_mods = [m for m in self.modules() if isinstance(m, Inkling2Experts)]
    ours = [m._xtier for m in experts_mods if getattr(m, '_xtier', None) is not None]
    if ours:
        unplanned = [b.layer_id for b in ours if b.plan is None and not (b.pool and b.pool.full)]
        if unplanned:
            raise RuntimeError(f'xtier: Inkling-2 MoE layers {unplanned} have no CPU engine plan after loading')
        from xtier.cache import cold
        cold.relink()
        if XTierMoEMethod._registered == 0:
            XTierMoEMethod.layers.sort(key=lambda b: b.cfg.layer_id)
        nvfp4_n = sum(1 for b in ours if b.nvfp4)
        mxfp8_n = len(ours) - nvfp4_n
        logger.info('xtier: Inkling-2 model: %d of %d MoE layers on XTier (%d NVFP4, %d MXFP8)',
                    len(ours), len(experts_mods), nvfp4_n, mxfp8_n)
    return result


def install(registry, hook_type):
    registry.register(f'{EXPERTS_CLS}.__init__', _experts_init, hook_type.AROUND)
    registry.register(f'{EXPERTS_CLS}.swizzle_nvfp4_scales', _noop_if_xtier, hook_type.AROUND)
    registry.register(f'{EXPERTS_CLS}.prepare_mxfp8_cutlass_scales', _noop_if_xtier, hook_type.AROUND)
    registry.register(f'{EXPERTS_CLS}.prepare_mega_moe', _noop_if_xtier, hook_type.AROUND)
    registry.register(f'{EXPERTS_CLS}.prepare_routed_exchange', _noop_if_xtier, hook_type.AROUND)
    registry.register(f'{EXPERTS_CLS}.forward', _experts_forward, hook_type.AROUND)
    registry.register(f'{MODEL_CLS}.load_weights', _after_load_weights, hook_type.AFTER)
