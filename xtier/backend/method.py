"""XTier's FusedMoE method: the routed experts of one MoE layer on the CPU engine.

It wraps the method the checkpoint's quantization config chose for the layer (the "GPU method") only for its weight
layout: the parameters are created in host memory with the GPU method's names, shapes and loader (FusedMoE's loader
follows ``layer.weight_load_method``). As soon as the last expert tensor of the layer is loaded, the layer goes to
the CPU engine (NUMA-sharded copies) and the host parameters are released in place, so host memory peaks at one copy
of the experts plus the layers still loading, and SGLang's post-load staging (which moves a module's parameters to
the GPU around ``process_weights_after_loading``) finds nothing left to move. With XTIER_FAST_LOAD=1 the loader
leaves the expert tensors alone and the shards are read from the checkpoint files instead (backend/direct.py); a
layer whose cold-store records match the checkpoint fills its RAM slots from the store.

Forward, per call of m tokens:
- m <= XTIER_DECODE_MAX (and always inside CUDA-graph capture): through the mailbox, in slots of up to 8 tokens.
  A packing kernel stores the inputs into the slot's mapped pinned memory, stream memory operations hand the request
  to the CPU dispatcher and wait for it, an unpacking kernel reads the FP32 output. Capturable, no host sync.
  With token flags (active_flags) a caller can drop tokens of these calls: their routes become -1 in the packing.
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

from xtier import config, platform
from xtier.backend import direct, formats, kernels
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
    _active = None       # int32 [DECODE_ROWS] token flags of the decode path (active_flags)
    _localized = False
    _registered = 0      # layers whose shards are pinned and caches filled

    def __init__(self, gpu_method, layer_config: LayerConfig):
        self.gpu_method = gpu_method
        self.cfg = layer_config
        self.format = formats.detect(gpu_method, draft=layer_config.draft)
        self.plan = None
        self.prefill = None
        self.mailbox = None
        self.calls = dict(decode=0, cpu_prefill=0, gpu_prefill=0, hybrid_prefill=0, split_prefill=0)
        self._source = None     # where the bulk parameters come from: 'copy' (the loader), 'direct', 'store'
        self._reads = []        # 'direct': the noted reads of the layer's expert tensors (direct.locate)
        self._cold_plan = None  # 'store': the layer's cold.LayerPlan

    # -- construction ---------------------------------------------------------------------------------------------
    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        rc = layer.moe_runner_config
        unsupported = [what for what, bad in (
            ('activation', rc.activation != 'silu'), ('non-gated experts', not rc.is_gated),
            ('expert bias', bool(extra_weight_attrs.get('with_bias'))),
            ('router weight on input', rc.apply_router_weight_on_input),
            ('fused shared experts', layer.num_fused_shared_experts > 0),
            ('expert parallel MoE', layer.moe_ep_size > 1),
            ('params dtype', params_dtype != torch.bfloat16)) if bad]
        if unsupported:
            raise NotImplementedError(f'xtier: layer {self.cfg.layer_id}: unsupported {", ".join(unsupported)}')
        self.hidden, self.inter, self.experts = hidden_size, intermediate_size_per_partition, num_experts
        self.swiglu_limit = float(rc.swiglu_limit or rc.gemm1_clamp_limit or 0.0)
        # GPT-OSS SwiGLU (MiniMax M3: alpha 1.702, beta 1): gate * sigmoid(alpha * gate) * (up + beta)
        self.swiglu_alpha = 1.0 if rc.gemm1_alpha is None else float(rc.gemm1_alpha)
        self.swiglu_beta = 0.0 if rc.gemm1_beta is None else float(rc.gemm1_beta)
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
        missing = sorted(set(self.format.required) - set(own.values()))
        if missing:   # e.g. a compressed-tensors scheme other than NVFP4: its experts would never complete
            scheme = type(getattr(layer, 'scheme', self.gpu_method)).__name__
            raise NotImplementedError(f'xtier: layer {self.cfg.layer_id}: {scheme} creates no {", ".join(missing)} '
                                      f'({self.format.name} format)')
        self._names = {i: n for i, n in own.items() if n in self.format.required}
        self._shards = {n: [set() for _ in range(num_experts)] for n in self.format.required}   # loaded shard ids
        self._missing = {n: num_experts for n in self.format.required}                        # incomplete experts
        self._layer = layer
        XTierMoEMethod.layers.append(self)

    def _tracking_loader(self, layer, original):
        def weight_loader(param, loaded_weight, *args, **kwargs):
            name = self._names.get(id(param))
            if name is None:
                return original(param, loaded_weight, *args, **kwargs)
            if self.plan is not None:
                raise RuntimeError(f'xtier: layer {self.cfg.layer_id}: {name} loaded again after the hand-off')
            shard = kwargs.get('shard_id', args[1] if len(args) > 1 else None)
            expert = kwargs.get('expert_id', args[2] if len(args) > 2 else None)
            result = None
            if not self._bypass(layer, name, loaded_weight, shard, expert):
                result = original(param, loaded_weight, *args, **kwargs)
                platform.bulk_copy()   # this checkpoint file is mostly XTier experts: read the next ones ahead
            self._record(name, shard, expert)
            return result
        weight_loader.xtier_inline = True   # runs on the loading thread (backend.hooks)
        return weight_loader

    def _bypass(self, layer, name, w, shard, expert):
        """Whether the loader leaves checkpoint tensor `w` alone: the layer's shards are filled from the checkpoint
        files (the read noted here for direct.fill) or from the cold store (cold.attach_loaded)."""
        if name not in self.format.bulk:
            return False   # global scales: through the loader
        if self._source is None:
            self._source = self._choose_source(layer)
        if self._source == 'store':
            return True
        if self._source != 'direct':
            return False
        read = direct.locate(self.format.bulk.index(name), shard, expert, w, self.hidden, self.inter, self.experts,
                             self.format.name)
        if read is None:
            if self._reads:   # the reads' [gate; up] rows and a loader's [up; gate] rows must not mix in one layer
                raise RuntimeError(f'xtier: layer {self.cfg.layer_id}: {name} of expert {expert} ({shard}) cannot be '
                                   'read as it is after other expert tensors were (XTIER_FAST_LOAD=0 loads them)')
            self._source = 'copy'
            return False
        self._reads.append(read)
        return True

    def _choose_source(self, layer):
        from xtier.cache import cold
        if os.environ.get('XTIER_PREFILL_FROM_SHARDS', '1') != '1':
            return 'copy'   # the prefill copies the inputs then
        if cold.enabled() and not self.cfg.draft and 0 < self.cfg.cache_slots < self.experts:
            lp = cold.LayerPlan(self.cfg.layer_id, self.experts, self.cfg.cache_slots)
            if cold.load_enabled() and cold.loadable(lp, self.hidden, self.inter, self.format.name, self.cfg.threads):
                self._cold_plan = lp
                return 'store'
            return 'copy'   # attach writes or checks the records from the loaded experts
        if direct.enabled() and getattr(layer, 'moe_tp_size', 1) == 1:
            return 'direct'
        return 'copy'

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
        at a time: a layer holds its loaded parameters and its new shards at once (DeepSeek V4.1: 2 x 7 GB). The
        expert copies run on the loading thread, in checkpoint order (backend.hooks), so layers complete one after
        the other."""
        with XTierMoEMethod._hand_off_lock:
            self._convert(layer)

    def _convert(self, layer):
        source = self._source or 'copy'
        if source == 'copy':
            up_first = bool(getattr(self.gpu_method, 'load_up_proj_weight_first', False))
            inputs = self.format.engine_inputs(layer, up_first)
        else:   # the bulk parameters were never filled: the shards are, below
            inputs = (None,) * 4 + self.format.globals(layer)
        if any(t is not None and t.device.type != 'cpu' for t in inputs):
            inputs = [t if t is None else t.cpu() for t in inputs]
        # TP: this rank holds the I / tp slice of every expert; SGLang all-reduces the partial outputs
        placement.apply(self.cfg.threads, getattr(layer, 'moe_tp_rank', 0), getattr(layer, 'moe_tp_size', 1))
        t0 = time.perf_counter()
        pointers = [None if t is None else t.data_ptr() for t in inputs]
        from xtier.cache import cold
        lp = None   # cold experts (XTIER_COLD=1): partly cached layers of the model keep part of the experts in RAM
        if cold.enabled() and not self.cfg.draft and 0 < self.cfg.cache_slots < self.experts:
            lp = self._cold_plan or cold.LayerPlan(self.cfg.layer_id, self.experts, self.cfg.cache_slots)
        self.plan = CpuPlan(self.experts, self.cfg.threads, pointers, hidden=self.hidden, inter=self.inter,
                            swiglu_limit=self.swiglu_limit, scale_format=self.format.name,
                            swiglu_alpha=self.swiglu_alpha, swiglu_beta=self.swiglu_beta,
                            resident=lp.resident if lp else None)
        how = ''
        if source == 'direct':   # read while the loader goes on; done and checked at the end of loading (direct.finish)
            nbytes = direct.start(self.plan, self._reads, self.format.name, self.cfg.layer_id)
            how = f', {nbytes / 1e9:.1f} GB being read from the checkpoint'
            self._reads = []
        if source == 'store':   # read while the loader goes on, as a direct layer's
            prefill_cold = cold.attach_loaded(self.plan, lp, self.format.name)
            how = ', being read from the cold store'
        else:
            prefill_cold = cold.attach(self.plan, lp, inputs, self.format.name) if lp else None
        # uploads from the engine's shards; copies the global scales (pointers 4, 5) now
        self.prefill = Prefill(self.experts, pointers, native=self.plan, hidden=self.hidden, inter=self.inter,
                               swiglu_limit=self.swiglu_limit, swiglu_alpha=self.swiglu_alpha,
                               swiglu_beta=self.swiglu_beta)
        self.prefill.order = self.cfg.layer_id   # the cross-layer prefetch links layers in this order
        self.prefill.cold = prefill_cold
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
        if prefill_cold is not None:   # cold experts the pool holds are copied from it, not read
            prefill_cold.pool = self.pool
        for name in self.format.required:
            p = getattr(layer, name, None)
            if p is not None:
                p.data = torch.empty(0, dtype=p.dtype)   # the model's parameter dict may still hold p
        logger.info('xtier: layer %d: %d experts (H %d, I %d, %s) on the CPU engine in %.2f s%s', self.cfg.layer_id,
                    self.experts, self.hidden, self.inter, self.format.name, time.perf_counter() - t0, how)
        if direct.digest_enabled() and source == 'copy':   # the others' once their reads are done (direct.finish)
            logger.info('xtier: layer %d: shards digest %s', self.cfg.layer_id, direct.digest(self.plan))

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
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        if TopKOutputChecker.format_is_bypassed(topk):   # routing left to the MoE kernel: the standard ids, as KT's
            topk = topk.to_standard()
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
        if not capturing and m > config.decode_max() and self.pool is not None and not self.pool.full:
            from xtier.cache import pool as cache_pool
            cache_pool.record_prompt(self.pool, ids)   # XTIER_CACHE_PROMPT=1: the prompt warms the expert cache
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
        active = XTierMoEMethod._active
        for start in range(0, m, SLOT_TOKENS):
            n = min(SLOT_TOKENS, m - start)
            xs, ids_s, ws = x.data_ptr() + start * H * 2, ids.data_ptr() + start * k * 4, weights.data_ptr() + start * k * 4
            act = active.data_ptr() + start * 4 if active is not None else 0
            if self.pool is not None:
                self.pool.decode(stream, n, k, xs, ids_s, ws, out.data_ptr() + start * H * 4, act)
                continue
            slot = self.mailbox.slot(stream, n, k)
            kernels.pack(slot, n, k, xs, ids_s, ws, H, act)
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
        table = None
        cold = self.prefill.cold
        if cold is None:
            gpu = ids < g
        else:   # cold experts: the CPU takes RAM slots [lo, hi), whole upload chunks below the cold staging; its
            # experts are in RAM, so its forward reads nothing and moves no slot while the GPU part uploads
            CH = self.prefill.chunk
            hi = cold.C // CH * CH
            lo = max(0, hi - (E - g) // CH * CH)
            if lo >= hi:
                out = torch.empty_like(x)
                self.prefill.run(x.data_ptr(), out.data_ptr(), ids.data_ptr(), weights.data_ptr(), m, k, stream)
                return out
            table = self.plan.slot_table()   # before the CPU part runs (slot_table refuses during a forward)
            slot = torch.tensor(table, dtype=torch.int64, device=ids.device)[ids.clamp(min=0).long()]
            gpu = (slot < lo) | (slot >= hi)
            g = E - (hi - lo)   # for the cost model: the CPU's experts
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
                             output_dtype=torch.float32, cold_table=table)
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


def active_flags(device) -> torch.Tensor:
    """Token flags of the decode path, int32 [DECODE_ROWS], all 1: a caller that sets row i to 0 on the stream before
    a forward makes the i-th token of each decode-path MoE call skip its routed experts (GPU slots and CPU engine; a
    zero output row), e.g. the tokens a truncated speculative verify will not use, and sets it back to 1 after. The
    first call creates the buffer; only graphs captured after it read the flags (they keep its address)."""
    if XTierMoEMethod._active is None:
        XTierMoEMethod._active = torch.ones(DECODE_ROWS, dtype=torch.int32, device=device)
        logger.info('xtier: decode-path token flags on (tokens can skip their routed experts)')
    return XTierMoEMethod._active
