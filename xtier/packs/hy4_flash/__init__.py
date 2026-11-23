"""Acceleration & SM120 compatibility pack for Hunyuan-4-Flash (``HYV4FlashForConditionalGeneration``).

- ModelOpt NVFP4 checkpoint support (``atakans/Hy4-flash-NVFP4``):
  1. Upstream ``HYV4FlashDecoderLayer`` passes ``quant_config=None`` to layer 0's dense MLP and
     ``HYV4FlashMoE`` passes ``quant_config=None`` to ``shared_experts``; for ModelOpt NVFP4
     checkpoints where those MLPs are not excluded in ``hf_quant_config.json``, rebuild them with
     ``quant_config`` so the packed NVFP4 weights and scales load cleanly.
  2. Upstream ``HYV4FlashForConditionalGeneration.load_weights`` unconditionally renames every
     ``.weight_scale`` tensor to ``.weight_scale_inv`` (FP8 block-scale convention), which breaks
     ModelOpt NVFP4 loaders that expect ``.weight_scale``. Undo that rename for ``modelopt_fp4``
     Hy4-Flash models before ``DeepseekV2WeightLoaderMixin.do_load_weights``.
- SM120 sparse MLA with learnable attention sinks + fast Triton decode + optional NVFP4 KV:
  1. Allow ``HYV4FlashForConditionalGeneration`` through ``_validate_flashinfer_sparse_mla_backend``
     on SM120 with FP8/NVFP4 DSA KV cache.
  2. Permit ``attn_sink`` on ``flashinfer_sparse_mla`` in ``DeepseekSparseAttnBackend.forward_extend``
     and ``forward_decode`` and forward it as ``sinks=[attn_sink]`` to FlashInfer's
     ``trtllm_batch_decode_with_kv_cache_mla``, or fuse it in our split-K Triton decode kernel
     (``kv_fp4.py``) for lower decode latency and/or ``XTIER_HY4_KV_FP4=1`` (416 B/tok NVFP4 KV).
- Hybrid SWA + DSA KV pool (``swa.py``, ``XTIER_HY4_HYBRID_SWA=1`` default on):
  Uses SGLang's native ``SWAKVPool`` so the 30 sliding-window layers (``window=512``) allocate a
  compact sliding pool without indexer keys instead of 1M full DSA slots (~21.9 GiB VRAM freed at
  1M tokens).
- Engram FP8 tables (64.7 GB across layers 1, 11, 21, 31) on NVMe via XTier RowStore with
  preloaded BF16 per-row scales and ordered side-stream prefetch (``engram.py``, when
  ``XTIER_ENGRAM_ROWS=nvme``).
"""

import logging

import torch

logger = logging.getLogger(__name__)

DECODER_LAYER_INIT = 'sglang.srt.models.hunyuan_v4_flash.HYV4FlashDecoderLayer.__init__'
MOE_INIT = 'sglang.srt.models.hunyuan_v4_flash.HYV4FlashMoE.__init__'
MOE_FWD = 'sglang.srt.models.hunyuan_v4_flash.HYV4FlashMoE.forward'
DO_LOAD_WEIGHTS = 'sglang.srt.models.deepseek_common.deepseek_weight_loader.DeepseekV2WeightLoaderMixin.do_load_weights'
VALIDATE_SPARSE_MLA = 'sglang.kernels.ops.attention.flash_mla_sm120._validate_flashinfer_sparse_mla_backend'
FLASHINFER_SPARSE_MLA = 'sglang.kernels.ops.attention.flash_mla_sm120.flashinfer_sparse_mla_forward'
DSA_EXTEND = 'sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_extend'
DSA_DECODE = 'sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_decode'
RUN_POST_PROCESS = 'sglang.srt.arg_groups.model_hook.run_post_process_pass'

_pending_sink = [None]


def _is_modelopt_fp4(quant_config) -> bool:
    return quant_config is not None and quant_config.get_name() == 'modelopt_fp4'


def _decoder_layer_init(result, self, config, layer_id, quant_config, prefix, *args, **kwargs):
    """Quantize layer 0 dense MLP when the ModelOpt NVFP4 checkpoint quantized it."""
    if (config.mlp_layer_types[layer_id] == 'dense'
            and _is_modelopt_fp4(quant_config)
            and not quant_config.is_layer_excluded(f'{prefix}.mlp.gate_proj')):
        from sglang.srt.models.hunyuan_v4_flash import HYV4FlashDenseMLP
        self.mlp = HYV4FlashDenseMLP(config, quant_config, f'{prefix}.mlp')
    return result


def _moe_init(result, self, config, layer_id, quant_config, prefix, *args, **kwargs):
    """Quantize shared_experts when the ModelOpt NVFP4 checkpoint quantized them."""
    if (_is_modelopt_fp4(quant_config)
            and not quant_config.is_layer_excluded(f'{prefix}.shared_experts.gate_proj')):
        from sglang.srt.models.deepseek_v2 import DeepseekV2MLP
        self.shared_experts = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f'{prefix}.shared_experts',
        )
    return result


def _moe_forward(original, self, hidden_states, forward_batch=None, apply_out_norm=True):
    """During decode (rows <= 16), overlap ``shared_experts`` with the CPU MoE mailbox wait.

    ``LayerPool.decode`` launches GPU cached experts while the CPU computes missed experts, then
    blocks at ``mailbox.signal(2)``. Running ``shared_experts(hidden_states)`` between ``_kernels``
    and ``mailbox.signal(2)`` hides ~0.73 ms/step of GPU work behind CPU tail latency at bs=4 with
    bit-identical numerics and full CUDA-graph safety.
    """
    if (hidden_states.shape[0] == 0
            or hidden_states.shape[0] > 16
            or getattr(self, 'shared_experts', None) is None):
        return original(self, hidden_states, forward_batch, apply_out_norm=apply_out_norm)
    from sglang.srt.models.hunyuan_v4_flash import _maybe_all_reduce
    from xtier.cache import pool as xtier_pool

    router_logits = self.gate(hidden_states)
    topk_output = self.topk(hidden_states, router_logits)
    routed_input = (
        self.routed_latent_down(hidden_states)[0]
        if self.routed_latent_down is not None
        else hidden_states
    )
    shared_box = [None]
    xtier_pool._between[0] = lambda: shared_box.__setitem__(0, self.shared_experts(hidden_states))
    try:
        routed = self.experts(routed_input, topk_output)
    finally:
        xtier_pool._between[0] = None
    if self.routed_latent_up is not None:
        routed = self.routed_latent_up(routed)[0]
    shared = shared_box[0] if shared_box[0] is not None else self.shared_experts(hidden_states)
    out = _maybe_all_reduce(routed + shared)
    if apply_out_norm and self.combine_output_norm is not None:
        out = self.combine_output_norm(out)
    return out


def _do_load_weights(original, self, weights, *args, **kwargs):
    """Restore ``.weight_scale`` for ModelOpt NVFP4 Hy4-Flash checkpoints."""
    if (type(self).__name__ == 'HYV4FlashForConditionalGeneration'
            and _is_modelopt_fp4(getattr(self, 'quant_config', None))):
        def restore_scales():
            for name, loaded_weight in weights:
                if name.endswith('.weight_scale_inv'):
                    name = name[:-4]
                yield name, loaded_weight
        return original(self, restore_scales(), *args, **kwargs)
    return original(self, weights, *args, **kwargs)


def _validate_sparse_mla(original, *args, **kwargs):
    selected = {kwargs.get('prefill_impl'), kwargs.get('decode_impl')}
    if kwargs.get('model_arch') == 'HYV4FlashForConditionalGeneration' and 'flashinfer_sparse_mla' in selected:
        kwargs = dict(kwargs, model_arch='GlmMoeDsaForCausalLM')
    return original(*args, **kwargs)


def _dsa_extend(original, self, *args, **kwargs):
    attn_sink = kwargs.get('attn_sink')
    if attn_sink is not None:
        forward_batch = kwargs.get('forward_batch', args[4] if len(args) > 4 else None)
        use_decode = (forward_batch is not None and (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ))
        dsa_impl = self.dsa_decode_impl if use_decode else self.dsa_prefill_impl
        if dsa_impl == 'flashinfer_sparse_mla':
            kwargs = dict(kwargs, attn_sink=None)
            _pending_sink[0] = attn_sink
            try:
                return original(self, *args, **kwargs)
            finally:
                _pending_sink[0] = None
    return original(self, *args, **kwargs)


def _dsa_decode(original, self, *args, **kwargs):
    attn_sink = kwargs.get('attn_sink')
    if attn_sink is not None and self.dsa_decode_impl == 'flashinfer_sparse_mla':
        kwargs = dict(kwargs, attn_sink=None)
        _pending_sink[0] = attn_sink
        try:
            return original(self, *args, **kwargs)
        finally:
            _pending_sink[0] = None
    return original(self, *args, **kwargs)


def _flashinfer_sparse_mla_forward(
    original,
    q,
    kv_cache,
    indices,
    seq_lens,
    workspace_buffer,
    *,
    page_size,
    kv_cache_dim,
    qk_nope_head_dim,
    kv_lora_rank,
    qk_rope_head_dim,
    sm_scale,
    skip_softmax_threshold_scale_factor,
):
    from xtier.packs.hy4_flash import kv_fp4

    sink = _pending_sink[0]
    rows = q.shape[0]
    topk = indices.shape[1]

    # NVFP4 KV cache (416 B/tok): gather-unpack into L2-hot FP8 scratch + FlashInfer C++ for decode
    # (0.67 ms across 42 layers vs 4.80 ms for pure Triton), full active-slot unpack + FlashInfer for prefill.
    if kv_cache_dim == kv_fp4.NVFP4_KV_DIM:
        if rows <= kv_fp4.TRITON_DECODE_MAX_ROWS and kv_fp4.triton_fp8_decode_enabled():
            return kv_fp4.triton_sparse_mla_decode(q, kv_cache, indices, sink, sm_scale, kv_cache_dim)
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
        if rows <= kv_fp4.TRITON_DECODE_MAX_ROWS:
            fp8_buf, block_tables = kv_fp4.gather_unpack_nvfp4_to_fp8(kv_cache, indices)
            kv_arg = fp8_buf
            bt_arg = block_tables.unsqueeze(1)
        else:
            fp8_buf = kv_fp4.unpack_nvfp4_to_fp8(kv_cache, indices, page_size)
            kv_arg = fp8_buf.view(-1, page_size, kv_fp4.FP8_KV_DIM).unsqueeze(1)
            bt_arg = indices.unsqueeze(1)
        result = trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_arg,
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=bt_arg,
            seq_lens=seq_lens,
            max_seq_len=topk,
            sparse_mla_top_k=topk,
            bmm1_scale=float(sm_scale),
            bmm2_scale=1.0,
            sinks=[sink] if sink is not None else None,
            kv_scale_format='arbitrary_fp32',
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        )
        return result.squeeze(1)

    # FP8 KV cache (656 B/tok): use faster split-K Triton decode on small batches where FlashInfer
    # SM120 incurs ~160 us fixed overhead (especially 30 SWA layers with topk=512).
    if (kv_cache_dim == kv_fp4.FP8_KV_DIM
            and rows <= kv_fp4.TRITON_DECODE_MAX_ROWS
            and sink is not None
            and kv_fp4.triton_fp8_decode_enabled()
            and topk <= kv_fp4.triton_fp8_max_topk()):
        return kv_fp4.triton_sparse_mla_decode(q, kv_cache, indices, sink, sm_scale, kv_cache_dim)

    if sink is None:
        return original(
            q, kv_cache, indices, seq_lens, workspace_buffer,
            page_size=page_size, kv_cache_dim=kv_cache_dim,
            qk_nope_head_dim=qk_nope_head_dim, kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim, sm_scale=sm_scale,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        )
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    result = trtllm_batch_decode_with_kv_cache_mla(
        query=q.unsqueeze(1),
        kv_cache=kv_cache.view(torch.uint8).view(-1, page_size, kv_cache_dim).unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=indices.unsqueeze(1),
        seq_lens=seq_lens,
        max_seq_len=topk,
        sparse_mla_top_k=topk,
        bmm1_scale=float(sm_scale),
        bmm2_scale=1.0,
        sinks=[sink],
        kv_scale_format='arbitrary_fp32',
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
    )
    return result.squeeze(1)


def _run_post_process_pass(original, server_args, fn):
    """Allow FP8/NVFP4 DSA KV cache + flashinfer_sparse_mla with learnable sinks on SM120 for Hy4-Flash."""
    if getattr(fn, '__name__', '') in ('_dsa_kv_cache_dtype_default', '_dsa_split_backend_resolution'):
        from sglang.srt.arg_groups.overrides import ResolvedView, _declaration_overlay, model_config_of
        view = ResolvedView(server_args, overlay=_declaration_overlay(server_args))
        hf_config = model_config_of(view).hf_config
        archs = getattr(hf_config, 'architectures', None) or ()
        if ('HYV4FlashForConditionalGeneration' in archs
                and getattr(view, 'kv_cache_dtype', None) == 'fp8_e4m3'
                and getattr(view, 'dsa_prefill_backend', None) == 'flashinfer_sparse_mla'
                and getattr(view, 'dsa_decode_backend', None) == 'flashinfer_sparse_mla'
                and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12):
            return None
    return original(server_args, fn)


def install_serving(registry, hook_type):
    from xtier.packs.hy4_flash import swa

    registry.register(RUN_POST_PROCESS, _run_post_process_pass, hook_type.AROUND)
    swa.install_serving(registry, hook_type)
    return True


def install(registry, hook_type):
    from xtier.packs.hy4_flash import engram, indexpool_fp8, kv_fp4, swa

    registry.register(DECODER_LAYER_INIT, _decoder_layer_init, hook_type.AFTER)
    registry.register(MOE_INIT, _moe_init, hook_type.AFTER)
    registry.register(MOE_FWD, _moe_forward, hook_type.AROUND)
    registry.register(DO_LOAD_WEIGHTS, _do_load_weights, hook_type.AROUND)
    registry.register(VALIDATE_SPARSE_MLA, _validate_sparse_mla, hook_type.AROUND)
    registry.register(DSA_EXTEND, _dsa_extend, hook_type.AROUND)
    registry.register(DSA_DECODE, _dsa_decode, hook_type.AROUND)
    registry.register(FLASHINFER_SPARSE_MLA, _flashinfer_sparse_mla_forward, hook_type.AROUND)
    registry.register(RUN_POST_PROCESS, _run_post_process_pass, hook_type.AROUND)
    swa.install(registry, hook_type)
    kv_fp4.install(registry, hook_type)
    indexpool_fp8.install(registry, hook_type)
    engram.install(registry, hook_type)
