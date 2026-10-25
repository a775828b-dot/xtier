"""Pack for SGLang's MiniMax M3 models (MiniMax-M3.1-Flash: NVFP4 routed experts, MXFP8 projections, MSA sparse
attention), on SGLang's MiniMax M3.1 branch (PR #45925).

- SM120 runs the generic M3.1 stack (needed to run at all; the RTX PRO 5000 is SM120). The branch tests
  ``is_blackwell`` where it means SM100: on any Blackwell GPU an NVFP4 checkpoint gets the native stack (MSA-v1 Q8KV4
  attention on a packed NVFP4 KV pool, MegaMoE or TRT-LLM experts, an FP8 KV cache with TRT-LLM attention), whose
  kernels are SM100 / SM103 only (tcgen05; MegaMoE; TRT-LLM's FMHAv2 prefill on SM120 rejects an FP8 KV cache, so the
  server stops while resolving its arguments). On SM120 the stack is the one SGLang gives Hopper instead: MiniMax's
  sparse attention backend on its Triton path (MSA's fmha_sm100 is SM100-only there as well) over a dense TRT-LLM MHA
  backend (Hopper's FA3 is SM90-only), page size 128, the experts' scheme as on Hopper (Marlin, not MegaMoE: XTier
  runs them anyway). The KV cache is still the checkpoint's FP8 E4M3 (its kv_cache_quant_algo, which the Triton
  sparse path dequantizes on load), only not named in the arguments, where it would trip the FMHAv2 check: every M3
  layer is sparse, so the dense backend never runs a prefill. The MXFP8 projections stay MXFP8 (FlashInfer CUTLASS
  on SM120) and DSpark keeps its settings (BF16 draft KV, TRT-LLM MHA draft attention). Argument resolution happens
  in the process that serves the API, so that hook is a serving hook.
- NVFP4 KV cache on SM120 (``XTIER_MINIMAX_KV_FP4``, default on; memory: 648 instead of 1280 bytes per token and
  layer, a 512K context next to the hot experts): the KV pool and its cell size are the native stack's (SGLang's
  MSA-v1 NVFP4 pool, its stores through SGLang's fused quantize+store kernel), while the attention stays the generic
  Triton path, reading that cache through ``kv_fp4`` (decode: kernels decoding the cache in registers; prefill:
  SGLang's kernels on BF16 copies of the batch's contexts). The DSpark draft KV stays BF16.
- DSpark's target verify on SM120 as SGLang runs it on ROCm: the generic sparse backend verifies only on NPU and ROCm
  (on CUDA, DSpark comes with the native stack), so a verify batch reached its prefill path and failed on the first
  forward (no extend lengths). Each draft row becomes one causal decode query on the Triton sparse decode, with the
  accepted prefix plus its position as its length (metadata ops captured with the graph, KV stored once), and the
  captured graphs bound the KV length by the context length.
- DSpark draft LM head in FP8 (``XTIER_DRAFT_LMHEAD_FP8``, default on; speed): the draft projects every step through
  the target's BF16 LM head (200064 x 6144, 2.46 GB). The DeepSeek pack's row-scaled FP8 copy and its
  ``project_through_lm_head`` hook serve SGLang's generic DSpark draft as well; here the copy is made when that draft
  attaches the head. Draft proposals only: the target verifies every token with its BF16 head.
- On an SGLang without the M3.1 branch (upstream main, where DeepSeek, GLM and Qwen run) the pack installs nothing:
  its hook targets are not there (``m31_branch``).
"""

import copy
import functools
import importlib.util
import logging
import os

logger = logging.getLogger(__name__)
NATIVE_STACK = 'sglang.srt.configs.model_config.uses_minimax_m3_nvfp4_native_stack'
INVOKE_OVERRIDE = 'sglang.srt.arg_groups.model_override_base._invoke_provider'
MOE_SCHEME = 'sglang.srt.layers.quantization.minimax_m3_nvfp4.MiniMaxM3Nvfp4MoE'
SPARSE = 'sglang.srt.layers.attention.minimax_sparse_backend.MiniMaxSparseAttnBackend'
DSPARK_DRAFT = 'sglang.srt.models.dspark.DSparkDraftMixin'
DRAFT_PROJECT = 'sglang.srt.models.dspark.project_through_lm_head'
KV_POOL = 'sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_minimax_sparse_kv_pool'
KV_CELL = 'sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator._compute_cell_size'
OVERRIDES = '_minimax_m3_overrides'
_kv_pool_scope = [False]


@functools.lru_cache
def sm120():
    from sglang.srt.runtime_context import get_platform   # no CUDA context in the serving process
    return bool(get_platform().is_sm120)


@functools.lru_cache
def m31_branch():
    """SGLang's MiniMax M3.1 branch (PR #45925) is installed: its NVFP4 MoE scheme module is there. By file, without
    importing SGLang's modules (hooks are installed while it starts). Found through ``sglang.srt``: started from a
    directory that holds an SGLang checkout (``python -m`` puts the working directory first on sys.path), ``sglang``
    itself is a namespace package of that checkout's top directory, while its modules come from the installed SGLang
    (an editable install's finder)."""
    try:
        spec = importlib.util.find_spec('sglang.srt')
    except (ImportError, ValueError):
        return False
    roots = list(spec.submodule_search_locations or []) if spec else []
    return any(os.path.exists(os.path.join(r, 'layers', 'quantization', 'minimax_m3_nvfp4.py')) for r in roots)


def kv_fp4_enabled():
    return os.environ.get('XTIER_MINIMAX_KV_FP4', '1') != '0'


def _native_stack(original, quantization, *args, **kwargs):
    if sm120() and not _kv_pool_scope[0]:
        return False
    return original(quantization, *args, **kwargs)


def _in_kv_pool_scope(original, *args, **kwargs):
    """The KV pool's cell size and class as on the native stack (SGLang's MSA-v1 NVFP4 pool)."""
    _kv_pool_scope[0] = True
    try:
        return original(*args, **kwargs)
    finally:
        _kv_pool_scope[0] = False


def _kv_pool(original, *args, **kwargs):
    pool = _in_kv_pool_scope(original, *args, **kwargs)
    if sm120() and getattr(getattr(pool, 'main_pool', None), 'is_minimax_m3_nvfp4_main', False):
        from xtier.packs.minimax_m3 import kv_fp4 as fp4

        fp4.attach_scales(pool)
        fp4.enable_fused_store(pool)
    return pool


def _invoke_override(original, fn, server_args, hf_config, *args, **kwargs):
    """Around SGLang's call of a model's argument provider: M3's NVFP4 Blackwell choices made SM120-capable."""
    declared = original(fn, server_args, hf_config, *args, **kwargs)
    if getattr(fn, '__name__', '') != OVERRIDES or not sm120():
        return declared
    from sglang.srt.arg_groups.model_override_base import resolving_view

    cfg = resolving_view(server_args)
    if declared.get('quantization', cfg.quantization) != 'minimax_m3_nvfp4':
        return declared
    declared = dict(declared)
    changed = {}
    if declared.get('moe_a2a_backend') == 'megamoe':
        changed['moe_a2a_backend'] = declared.pop('moe_a2a_backend')
    if declared.get('moe_runner_backend') == 'deep_gemm':
        declared['moe_runner_backend'] = 'marlin'
        changed['moe_runner_backend'] = 'deep_gemm'
    if declared.get('kv_cache_dtype') == 'fp8_e4m3' and cfg.kv_cache_dtype == 'auto':
        changed['kv_cache_dtype'] = declared.pop('kv_cache_dtype')
    logger.info('xtier: MiniMax M3.1 NVFP4 on SM120: the generic stack (not %s), %s', changed, declared)
    return declared


def _moe_scheme_init(original, self, *args, **kwargs):
    """The experts' scheme as SGLang builds it off Blackwell (Marlin; XTier takes the layers)."""
    if not sm120():
        return original(self, *args, **kwargs)
    from sglang.srt.layers.quantization.compressed_tensors.schemes import CompressedTensorsW4A4Nvfp4MoE

    CompressedTensorsW4A4Nvfp4MoE.__init__(self)
    self.use_marlin, self.use_flashinfer_trtllm, self.use_mega_moe_prefill = True, False, False


def _verify(forward_batch):
    return sm120() and forward_batch.forward_mode.is_target_verify()


def _metadata_in_graph(original, self, forward_batch, *args, **kwargs):
    if _verify(forward_batch):
        return self._init_rocm_linear_verify_metadata(forward_batch)   # per-row lengths, captured ops
    return original(self, forward_batch, *args, **kwargs)


def _metadata_out_graph(result, self, forward_batch, *args, **kwargs):
    if _verify(forward_batch):
        in_capture = kwargs.get('in_capture', args[0] if args else False)
        if in_capture:   # tiny dummy lengths at capture; replays run longer sequences
            self._max_seqlen_k = self.max_context_len
        else:
            self._max_seqlen_k += self.speculative_num_draft_tokens
    return result


def _forward_extend(original, self, q, k, v, layer, forward_batch, *args, **kwargs):
    if not _verify(forward_batch):
        return original(self, q, k, v, layer, forward_batch, *args, **kwargs)
    meta = self._linear_verify_meta
    if meta is None or meta.seq_lens.numel() != q.shape[0]:
        raise RuntimeError('xtier: MiniMax M3 verify metadata missing on SM120')
    save_kv_cache = args[0] if args else kwargs.pop('save_kv_cache', True)
    verify_batch = copy.copy(forward_batch)   # the dense backend keeps the per-request metadata
    verify_batch.seq_lens, verify_batch.req_pool_indices = meta.seq_lens, meta.req_pool_indices
    return self.forward_decode(q, k, v, layer, verify_batch, save_kv_cache, **kwargs)


def install(registry, hook_type):
    if not m31_branch():
        return
    registry.register(NATIVE_STACK, _native_stack, hook_type.AROUND)
    registry.register(f'{MOE_SCHEME}.__init__', _moe_scheme_init, hook_type.AROUND)
    registry.register(f'{SPARSE}.init_forward_metadata_in_graph', _metadata_in_graph, hook_type.AROUND)
    registry.register(f'{SPARSE}.init_forward_metadata_out_graph', _metadata_out_graph, hook_type.AFTER)
    registry.register(f'{SPARSE}.forward_extend', _forward_extend, hook_type.AROUND)
    if kv_fp4_enabled():
        from xtier.packs.minimax_m3 import kv_fp4 as fp4

        registry.register(KV_CELL, _in_kv_pool_scope, hook_type.AROUND)
        registry.register(KV_POOL, _kv_pool, hook_type.AROUND)
        fp4.install(registry, hook_type)
    if os.environ.get('XTIER_DRAFT_LMHEAD_FP8', '1') != '0':
        from xtier import packs
        from xtier.packs import deepseek_v4

        registry.register(f'{DSPARK_DRAFT}.attach_shared_modules', deepseek_v4._attach_shared, hook_type.AFTER)
        if 'deepseek_v4' not in packs.selected():   # otherwise that pack already hooks the projection
            registry.register(DRAFT_PROJECT, deepseek_v4._draft_project, hook_type.AROUND)


def install_serving(registry, hook_type):
    """Hooks of the process that resolves the server arguments and serves the API."""
    if not m31_branch():
        return False
    registry.register(NATIVE_STACK, _native_stack, hook_type.AROUND)
    registry.register(INVOKE_OVERRIDE, _invoke_override, hook_type.AROUND)
    return True
