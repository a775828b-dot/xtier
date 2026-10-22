"""XTier's FusedMoE method: the routed experts of one MoE layer on the CPU engine.

It wraps the method the checkpoint's quantization config chose for the layer (the "GPU method") only for its weight
layout: the parameters are created in host memory with the GPU method's names, shapes and loader (FusedMoE's loader
follows ``layer.weight_load_method``). As soon as the last expert tensor of the layer is loaded, the layer goes to
the CPU engine (NUMA-sharded copies) and the host parameters are released in place, so host memory peaks at one copy
of the experts plus the layers still loading, and SGLang's post-load staging (which moves a module's parameters to
the GPU around ``process_weights_after_loading``) finds nothing left to move.

Forward, per call of m tokens:
- m <= XTIER_DECODE_MAX (and always inside CUDA-graph capture): through the mailbox, in slots of up to 8 tokens.
  A packing kernel stores the inputs into the slot's mapped pinned memory, stream memory operations hand the request
  to the CPU dispatcher and wait for it, an unpacking kernel reads the FP32 output. Capturable, no host sync.
- prefill of at least XTIER_GPU_PREFILL_MIN tokens: on the GPU (backend/prefill.py: expert chunks uploaded from the
  engine's pinned shards, grouped BF16 GEMMs, FP32 route sums);
- other prefill: synchronously on the CPU in batches of XTIER_CPU_PREFILL_BATCH tokens.
Numerics are the engine's or the GPU prefill's (BF16 operands, FP32 accumulation); the output is rounded to the
input dtype.
"""

import dataclasses
import logging
import time

import torch
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

from xtier import config
from xtier.backend import formats, kernels
from xtier.backend.prefill import Prefill
from xtier.engine import CpuPlan, Mailbox, placement

logger = logging.getLogger(__name__)
SLOT_TOKENS = 8      # tokens per mailbox slot (engine DECODE_M)
MAX_TOP_K = 10
DECODE_ROWS = 256    # rows of the shared decode output buffer (fixed: captured graphs keep its address)


@dataclasses.dataclass(frozen=True)
class LayerConfig:
    """Planner decision for one FusedMoE layer (returned where SGLang asks for a KTransformers config)."""
    layer_id: int
    threads: int
    cache_slots: int = 0   # GPU expert cache slots (xtier.cache); -1: every expert of the layer


class XTierMoEMethod(FusedMoEMethodBase):
    layers = []          # every XTier layer of the process, in creation order
    _decode_out = {}     # hidden size -> FP32 decode output [DECODE_ROWS, H], shared (layers run one after another)
    _localized = False
    _registered = False

    def __init__(self, gpu_method, layer_config: LayerConfig):
        self.gpu_method = gpu_method
        self.cfg = layer_config
        self.format = formats.detect(gpu_method)
        self.plan = None
        self.prefill = None
        self.mailbox = None
        self.calls = dict(decode=0, cpu_prefill=0, gpu_prefill=0)

    # -- construction ---------------------------------------------------------------------------------------------
    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        rc = layer.moe_runner_config
        unsupported = [what for what, bad in (
            ('activation', rc.activation != 'silu'), ('non-gated experts', not rc.is_gated),
            ('expert bias', bool(extra_weight_attrs.get('with_bias'))),
            ('router weight on input', rc.apply_router_weight_on_input),
            ('fused shared experts', layer.num_fused_shared_experts > 0),
            ('gemm1 alpha / beta', rc.gemm1_alpha is not None or rc.gemm1_beta is not None),
            ('expert or tensor parallel MoE', layer.moe_ep_size > 1 or layer.moe_tp_size > 1),
            ('params dtype', params_dtype != torch.bfloat16)) if bad]
        if unsupported:
            raise NotImplementedError(f'xtier: layer {self.cfg.layer_id}: unsupported {", ".join(unsupported)}')
        self.hidden, self.inter, self.experts = hidden_size, intermediate_size_per_partition, num_experts
        self.swiglu_limit = float(rc.swiglu_limit or rc.gemm1_clamp_limit or 0.0)
        extra_weight_attrs['weight_loader'] = self._tracking_loader(layer, extra_weight_attrs['weight_loader'])
        with torch.device('cpu'):
            self.gpu_method.create_weights(layer, num_experts, hidden_size, intermediate_size_per_partition,
                                           params_dtype, **extra_weight_attrs)
        for name in self.format.derived:   # derived from the empty tensors at creation, never used
            if getattr(layer, name, None) is not None:
                setattr(layer, name, None)
        # FusedMoE's loader follows this method's layout (row order, scale shapes, shard handling).
        layer.weight_load_method = self.gpu_method
        own = {id(p): n for n, p in layer.named_parameters(recurse=False)}
        self._names = {i: n for i, n in own.items() if n in self.format.required}
        self._shards = {n: [set() for _ in range(num_experts)] for n in self.format.required}   # loaded shard ids
        self._missing = {n: num_experts for n in self.format.required}                        # incomplete experts
        self._layer = layer
        XTierMoEMethod.layers.append(self)

    def _tracking_loader(self, layer, original):
        def weight_loader(param, loaded_weight, *args, **kwargs):
            name = self._names.get(id(param))
            if name is not None and self.plan is not None:
                raise RuntimeError(f'xtier: layer {self.cfg.layer_id}: {name} loaded again after the hand-off')
            result = original(param, loaded_weight, *args, **kwargs)
            if name is not None:
                shard = kwargs.get('shard_id', args[1] if len(args) > 1 else None)
                expert = kwargs.get('expert_id', args[2] if len(args) > 2 else None)
                self._record(name, shard, expert)
            return result
        return weight_loader

    def _record(self, name, shard, expert):
        kind = self.format.required[name]
        for e in (range(self.experts) if expert is None else (int(expert),)):
            if not 0 <= e < self.experts:
                continue
            shards = self._shards[name][e]
            if formats.shard_complete(kind, shards):
                continue
            shards.add(shard)
            if formats.shard_complete(kind, shards):
                self._missing[name] -= 1
        if not any(self._missing.values()):
            self._hand_off(self._layer)

    def create_moe_runner(self, layer, moe_runner_config):
        # No GPU runner. The GPU method's weight layout may depend on its runner config (load_up_proj_weight_first).
        self.gpu_method.moe_runner_config = moe_runner_config

    def get_moe_runner_backend(self, layer):
        return None   # a direct kernel consuming standard TopK output with local expert ids

    def _hand_off(self, layer):
        """All expert tensors are loaded: build the engine's copy and release the host parameters in place."""
        up_first = bool(getattr(self.gpu_method, 'load_up_proj_weight_first', False))
        inputs = self.format.engine_inputs(layer, up_first)
        if any(t.device.type != 'cpu' for t in inputs):
            inputs = [t.cpu() for t in inputs]
        placement.apply(self.cfg.threads)
        t0 = time.perf_counter()
        pointers = [t.data_ptr() for t in inputs]
        self.plan = CpuPlan(self.experts, self.cfg.threads, pointers, hidden=self.hidden, inter=self.inter,
                            swiglu_limit=self.swiglu_limit, scale_format=self.format.name)
        # uploads from the engine's shards; copies the global scales (pointers 4, 5) now
        self.prefill = Prefill(self.experts, pointers, native=self.plan, hidden=self.hidden, inter=self.inter,
                               swiglu_limit=self.swiglu_limit)
        self.prefill.order = self.cfg.layer_id   # the cross-layer prefetch links layers in this order
        del inputs, pointers
        self.mailbox = Mailbox(self.plan)
        self.pool = None
        slots = self.experts if self.cfg.cache_slots < 0 else self.cfg.cache_slots
        if slots > 0:   # allocated now, so the KV cache is sized after it
            from xtier.cache.pool import LayerPool
            self.pool = LayerPool(self.prefill, slots, self.plan, self.mailbox, self.cfg.layer_id)
            if self.pool.full:   # the prefill reads the experts from the slots instead of host memory
                self.prefill._upload_chunk = self._upload_from_cache
        self.plan.pool = self.pool   # read by the prefill (residency of the FP4 path)
        for name in self.format.required:
            p = getattr(layer, name, None)
            if p is not None:
                p.data = torch.empty(0, dtype=p.dtype)   # the model's parameter dict may still hold p
        logger.info('xtier: layer %d: %d experts (H %d, I %d, %s) on the CPU engine in %.2f s', self.cfg.layer_id,
                    self.experts, self.hidden, self.inter, self.format.name, time.perf_counter() - t0)

    def process_weights_after_loading(self, layer):
        if self.plan is None:   # some expert tensor never came through the loader
            logger.warning('xtier: layer %d: loading did not complete (experts missing per parameter: %s); handing '
                           'the experts over now', self.cfg.layer_id, {n: c for n, c in self._missing.items() if c})
            self._hand_off(layer)

    def _upload_from_cache(self, slot, e0, e1, stream, up_first=False):
        """Prefill staging of a layer whose experts are all in the cache: device copies of slots e0..e1 (filled in
        expert order, never swapped), bytes unchanged."""
        if up_first:
            raise NotImplementedError('xtier: FP4 prefill of a fully cached layer')
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            for dst, src in zip(slot, self.pool.arena):
                dst[:e1 - e0].copy_(src[e0:e1], non_blocking=True)

    # -- forward ----------------------------------------------------------------------------------------------------
    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        m = x.shape[0]
        if m == 0:
            return StandardCombineInput(hidden_states=x.new_zeros(x.shape))
        if x.dtype != torch.bfloat16:
            raise TypeError(f'xtier: BF16 activations required, got {x.dtype}')
        if not XTierMoEMethod._localized:
            XTierMoEMethod._localize_all()
        ids = topk.topk_ids if topk.topk_ids.dtype == torch.int32 else topk.topk_ids.to(torch.int32)
        weights = topk.topk_weights if topk.topk_weights.dtype == torch.float32 else topk.topk_weights.float()
        ids, weights, x = ids.contiguous(), weights.contiguous(), x.contiguous()
        k = ids.shape[1]
        if not 1 <= k <= MAX_TOP_K:
            raise ValueError(f'xtier: top-k {k} not supported')
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing and not XTierMoEMethod._registered:
            XTierMoEMethod._register_all()
        if m <= config.decode_max() or capturing:
            out = self._decode(x, ids, weights, m, k)
            self.calls['decode'] += 1
        elif 0 < config.gpu_prefill_min() <= m or (self.pool is not None and self.pool.full):
            self._cache_start()
            out = torch.empty_like(x)
            self.prefill.run(x.data_ptr(), out.data_ptr(), ids.data_ptr(), weights.data_ptr(), m, k,
                             torch.cuda.current_stream().cuda_stream)
            self.calls['gpu_prefill'] += 1
        else:
            self._cache_start()
            out = self._cpu_prefill(x, ids, weights, m, k)
            self.calls['cpu_prefill'] += 1
        out = out.to(x.dtype)
        rsf = layer.moe_runner_config.routed_scaling_factor
        if rsf is not None and rsf != 1.0 and not layer.should_fuse_routed_scaling_factor_in_topk:
            out.mul_(rsf)
        return StandardCombineInput(hidden_states=out)

    def _decode(self, x, ids, weights, m, k):
        H = self.hidden
        if m > DECODE_ROWS:
            raise ValueError(f'xtier: {m} tokens exceed the decode buffer ({DECODE_ROWS}); lower XTIER_DECODE_MAX')
        out = XTierMoEMethod._decode_out.get(H)
        if out is None:
            out = XTierMoEMethod._decode_out[H] = torch.empty((DECODE_ROWS, H), dtype=torch.float32, device=x.device)
        stream = torch.cuda.current_stream().cuda_stream
        for start in range(0, m, SLOT_TOKENS):
            n = min(SLOT_TOKENS, m - start)
            xs, ids_s, ws = x.data_ptr() + start * H * 2, ids.data_ptr() + start * k * 4, weights.data_ptr() + start * k * 4
            if self.pool is not None:
                self.pool.decode(stream, n, k, xs, ids_s, ws, out.data_ptr() + start * H * 4)
                continue
            slot = self.mailbox.slot(stream, n, k)
            kernels.pack(slot, n, k, xs, ids_s, ws, H)
            self.mailbox.notify(stream, n, k)
            kernels.unpack(slot, n, out.data_ptr() + start * H * 4, H)
        return out[:m]

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
            self.plan.forward(n, k, x_h.data_ptr() + start * H * 2, ids_h.data_ptr() + start * k * 4,
                              w_h.data_ptr() + start * k * 4, out_h.data_ptr() + start * H * 4)
        # pinned blocks are reused only after the copy completes (the caching host allocator records the stream)
        return out_h.to(x.device, non_blocking=True)

    @classmethod
    def _localize_all(cls):
        """Once, at the first MoE call (the model is fully loaded): move the shard pages that loading pressure placed
        on another NUMA node back to their node."""
        cls._localized = True
        t0, failed = time.perf_counter(), 0
        for method in cls.layers:
            if method.plan is not None:
                failed += method.plan.localize() or 0
        logger.info('xtier: NUMA localize of %d layers in %.1f s, %d ranges not fully moved', len(cls.layers),
                    time.perf_counter() - t0, failed)

    @classmethod
    def _register_all(cls):
        """Once, at the first MoE call outside graph capture (SGLang's warm-up request): pin every layer's shards for
        the GPU prefill upload, so the first user prompt does not pay for it. Pinning is not allowed during capture
        and must follow the NUMA localize (pinned pages cannot migrate)."""
        cls._registered = True
        cached = any(method.pool is not None for method in cls.layers)
        if not config.gpu_prefill_min() and not cached:
            return
        t0 = time.perf_counter()
        for method in cls.layers:
            method.prefill.ensure_registered()
        logger.info('xtier: pinned the shards of %d layers for GPU prefill in %.1f s', len(cls.layers),
                    time.perf_counter() - t0)
        if cached:
            from xtier.cache import pool
            pool.fill_all()

    @staticmethod
    def _cache_start():
        """At a prefill: start the cache's adaptive policy once the decode graphs exist."""
        from xtier.cache import pool
        pool.maybe_start()

    def stats(self):
        return dict(layer=self.cfg.layer_id, **self.calls)
