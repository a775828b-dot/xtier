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
- prefill of a fully cached layer: on its slots in place (cache/prefill.py);
- prefill of a partly cached layer below XTIER_HYBRID_PREFILL_MAX tokens: cached routes on the GPU, the others on the
  CPU at the same time (cache/prefill.py, hybrid_prefill);
- other prefill of at least XTIER_GPU_PREFILL_MIN tokens: on the GPU (backend/prefill.py: expert chunks uploaded
  from the engine's pinned shards, grouped BF16 GEMMs, FP32 route sums); up to XTIER_SPLIT_PREFILL_MAX tokens on a
  layer uploaded in expert chunks, the GPU takes the first g experts and the CPU the others at the same time
  (_split_prefill);
- the rest: synchronously on the CPU in batches of XTIER_CPU_PREFILL_BATCH tokens.
Numerics are the engine's or the GPU prefill's (BF16 operands, FP32 accumulation); the output is rounded to the
input dtype.
"""

import dataclasses
import logging
import os
import threading
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
    draft: bool = False    # a layer of a speculative draft model


class XTierMoEMethod(FusedMoEMethodBase):
    layers = []          # every XTier layer of the process, in creation order
    _decode_out = {}     # hidden size -> FP32 decode output [DECODE_ROWS, H], shared (layers run one after another)
    _localized = False
    _registered = 0      # layers whose shards are pinned and caches filled

    def __init__(self, gpu_method, layer_config: LayerConfig):
        self.gpu_method = gpu_method
        self.cfg = layer_config
        self.format = formats.detect(gpu_method)
        self.plan = None
        self.prefill = None
        self.mailbox = None
        self.calls = dict(decode=0, cpu_prefill=0, gpu_prefill=0, hybrid_prefill=0, split_prefill=0)

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
            ('expert parallel MoE', layer.moe_ep_size > 1),
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

    _hand_off_lock = threading.Lock()

    def _hand_off(self, layer):
        """All expert tensors are loaded: build the engine's copy and release the host parameters in place. One layer
        at a time: a layer holds its loaded parameters and its new shards at once (DeepSeek V4.1: 2 x 7 GB), and
        SGLang's loader threads complete several layers together; on the 377 GB host with DSpark that peak pushed
        ~20 GB into swap, and pinning the shards then took 554 s instead of 72 s."""
        with XTierMoEMethod._hand_off_lock:
            self._convert(layer)

    def _convert(self, layer):
        up_first = bool(getattr(self.gpu_method, 'load_up_proj_weight_first', False))
        inputs = self.format.engine_inputs(layer, up_first)
        if any(t.device.type != 'cpu' for t in inputs):
            inputs = [t.cpu() for t in inputs]
        # TP: this rank holds the I / tp slice of every expert; SGLang all-reduces the partial outputs
        placement.apply(self.cfg.threads, getattr(layer, 'moe_tp_rank', 0), getattr(layer, 'moe_tp_size', 1))
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
            if self.pool.full:   # a staged prefill reads the experts from the slots instead of host memory
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
        """Staged prefill of a fully cached layer (XTIER_CACHE_PREFILL=staged): device copies of slots e0..e1 (filled
        in expert order, never swapped), bytes unchanged. Partly cached layers upload whole chunks from host memory:
        skipping their cached experts would cut every chunk into short runs of 48 copies each (8 NUMA shards x 6
        tensors)."""
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
        if not capturing and XTierMoEMethod._registered < len(XTierMoEMethod.layers):
            XTierMoEMethod._register_all()
        timed = config.timing() and not capturing and m > config.decode_max()
        if timed:
            torch.cuda.current_stream().synchronize()
            t0 = time.perf_counter()
        if m <= config.decode_max() or capturing:
            if not capturing:   # short prompts never reach a prefill branch: the policy must start here too
                self._cache_start()
            out = self._decode(x, ids, weights, m, k)
            self.calls['decode'] += 1
        elif self.pool is not None and self.pool.full and config.resident_prefill():
            from xtier.cache.prefill import resident_prefill
            out = resident_prefill(self.prefill, self.pool.arena, x, ids, weights)
            self.calls['gpu_prefill'] += 1
        elif self.pool is not None and not self.pool.full and m < config.hybrid_prefill_max():
            from xtier.cache.prefill import hybrid_prefill
            self._cache_start()
            out = hybrid_prefill(self.pool, self.plan, x, ids, weights, config.cpu_prefill_batch())
            self.calls['hybrid_prefill'] += 1
        elif 0 < config.gpu_prefill_min() <= m or (self.pool is not None and self.pool.full):
            self._cache_start()
            split = config.split_prefill_max()
            if (self.pool is None or not self.pool.full) and m <= split and self.prefill.chunk < self.experts:
                out = self._split_prefill(x, ids, weights, m, k)
                self.calls['split_prefill'] += 1
            else:
                out = torch.empty_like(x)
                self.prefill.run(x.data_ptr(), out.data_ptr(), ids.data_ptr(), weights.data_ptr(), m, k,
                                 torch.cuda.current_stream().cuda_stream)
                self.calls['gpu_prefill'] += 1
        else:
            self._cache_start()
            out = self._cpu_prefill(x, ids, weights, m, k)
            self.calls['cpu_prefill'] += 1
        if timed:
            torch.cuda.current_stream().synchronize()
            self._time(m, time.perf_counter() - t0)
        out = out.to(x.dtype)
        rsf = layer.moe_runner_config.routed_scaling_factor
        if rsf is not None and rsf != 1.0 and not layer.should_fuse_routed_scaling_factor_in_topk:
            out.mul_(rsf)
        return StandardCombineInput(hidden_states=out)

    def _decode(self, x, ids, weights, m, k):
        H = self.hidden
        if m > DECODE_ROWS:
            raise ValueError(f'xtier: a {m}-token call on the decode path (decode buffer: {DECODE_ROWS} tokens); prefill '
                             'cannot be captured in a CUDA graph (--cuda-graph-backend-prefill disabled), and '
                             'XTIER_DECODE_MAX must stay below the buffer')
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

    _split_cost = None   # {gpu_ms per expert, cpu_ms per expert and token}, shared by the layers of the process

    def _split_experts(self, m):
        """Experts [0, g) for the GPU: g balances the upload time per expert against the CPU time per expert and
        token, in whole upload chunks; both costs are refined from measured calls."""
        E, CH = self.experts, self.prefill.chunk
        cost = XTierMoEMethod._split_cost
        if cost is None:
            scale = self.hidden * self.inter / (5120 * 2304)
            per_expert = 3 * self.hidden * self.inter * 9 // 16   # FP4 weights and scales
            cost = XTierMoEMethod._split_cost = dict(
                gpu_ms=per_expert / 55e9 * 1e3,   # upload-bound GPU
                cpu_ms=float(os.environ.get('XTIER_SPLIT_CPU_US', '0.285')) * scale / 1e3)
        cpu = cost['cpu_ms'] * m
        g = E * cpu / (cost['gpu_ms'] + cpu)
        return max(CH, min(E, int(round(g / CH)) * CH))

    def _split_prefill(self, x, ids, weights, m, k):
        """Expert-split prefill (lkqmoe 0.5.6's hybrid prefill): the routes to experts [0, g) through the GPU
        prefill (g whole upload chunks; the other routes get id -1), the others on the CPU engine in a thread at the
        same time; the FP32 partial outputs add up."""
        H, E = self.hidden, self.experts
        g = self._split_experts(m)
        stream = torch.cuda.current_stream().cuda_stream
        if g >= E:
            out = torch.empty_like(x)
            self.prefill.run(x.data_ptr(), out.data_ptr(), ids.data_ptr(), weights.data_ptr(), m, k, stream)
            return out
        gpu = ids < g
        ids_g = torch.where(gpu, ids, -1).to(torch.int32).contiguous()
        w_g = torch.where(gpu, weights, 0.0).contiguous()
        x_h = torch.empty((m, H), dtype=torch.bfloat16, pin_memory=True)
        ids_h = torch.empty((m, k), dtype=torch.int32, pin_memory=True)
        w_h = torch.empty((m, k), dtype=torch.float32, pin_memory=True)
        out_c = torch.empty((m, H), dtype=torch.float32, pin_memory=True)
        x_h.copy_(x, non_blocking=True)
        ids_h.copy_(torch.where(gpu, -1, ids), non_blocking=True)
        w_h.copy_(torch.where(gpu, 0.0, weights), non_blocking=True)
        staged = torch.cuda.Event()
        staged.record()
        timing, batch = {}, config.cpu_prefill_batch()

        def cpu_part():
            staged.synchronize()
            t0 = time.perf_counter()
            for start in range(0, m, batch):
                n = min(batch, m - start)
                self.plan.forward(n, k, x_h.data_ptr() + start * H * 2, ids_h.data_ptr() + start * k * 4,
                                  w_h.data_ptr() + start * k * 4, out_c.data_ptr() + start * H * 4)
            timing['cpu'] = time.perf_counter() - t0

        worker = threading.Thread(target=cpu_part, name='xtier-split-cpu')
        worker.start()
        out_g = torch.empty((m, H), dtype=torch.float32, device=x.device)
        t0 = time.perf_counter()
        try:
            self.prefill.run(x.data_ptr(), out_g.data_ptr(), ids_g.data_ptr(), w_g.data_ptr(), m, k, stream,
                             output_dtype=torch.float32)
            gpu_s = time.perf_counter() - t0   # the GPU part only; the join may wait for the CPU
        finally:
            worker.join()
        out = out_g + out_c.to(x.device, non_blocking=True)
        cost = XTierMoEMethod._split_cost
        if 'cpu' in timing:
            cost['cpu_ms'] = 0.8 * cost['cpu_ms'] + 0.2 * timing['cpu'] * 1e3 / ((E - g) * m)
        cost['gpu_ms'] = 0.8 * cost['gpu_ms'] + 0.2 * gpu_s * 1e3 / g
        self.last_split = dict(m=m, gpu_experts=g, cpu_ms=round(timing.get('cpu', 0) * 1e3, 1),
                               gpu_ms=round(gpu_s * 1e3, 1))
        return out

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
        """At the first MoE call outside graph capture after layers were added (SGLang's warm-up runs; a speculative
        draft loads after the target model): pin the new layers' shards for the GPU prefill upload, so the first user
        prompt does not pay for it, and fill their expert caches. Pinning is not allowed during capture and must
        follow the NUMA localize (pinned pages cannot migrate)."""
        new, cls._registered = cls.layers[cls._registered:], len(cls.layers)
        cached = any(method.pool is not None for method in new)
        if not config.gpu_prefill_min() and not cached:
            return
        t0 = time.perf_counter()
        for method in new:
            method.prefill.ensure_registered()
        logger.info('xtier: pinned the shards of %d layers for GPU prefill in %.1f s', len(new),
                    time.perf_counter() - t0)
        if cached:
            from xtier.cache import pool
            pool.fill_all()

    @staticmethod
    def _cache_start():
        """At any call outside graph capture: start the cache's adaptive policy once the decode graphs exist (with
        speculative decoding and short prompts every call before the first graph replay is a decode-sized one)."""
        from xtier.cache import pool
        pool.maybe_start()

    _timing = {}

    def _time(self, m, seconds):
        """XTIER_TIMING=1: prefill MoE time per kind of layer, logged once per forward (at the last layer)."""
        kind = 'cpu' if self.pool is None else 'full' if self.pool.full else 'cached'
        t = XTierMoEMethod._timing.setdefault(kind, [0, 0.0])
        t[0] += 1
        t[1] += seconds
        if self is XTierMoEMethod.layers[-1] or self.cfg.layer_id == max(x.cfg.layer_id for x in XTierMoEMethod.layers):
            logger.info('xtier timing: %d tokens: %s', m, ', '.join(f'{k} {n} layers {s * 1e3:.1f} ms'
                                                                    for k, (n, s) in XTierMoEMethod._timing.items()))
            XTierMoEMethod._timing = {}

    def stats(self):
        return dict(layer=self.cfg.layer_id, **self.calls)
