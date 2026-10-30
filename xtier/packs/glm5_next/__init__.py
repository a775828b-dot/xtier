"""Acceleration pack for SGLang's ``glm5_next`` models (GLM-5.3 Flash).

- ModelOpt exclusions of fused modules: the GLM-5.3 Flash NVFP4 checkpoint keeps its attention projections in BF16
  and lists them unfused (``*.self_attn.q_proj``, ``k_proj``, ``v_proj``, ...), while SGLang loads them fused
  (``self_attn.qkv_proj``). ``ModelOptQuantConfig.is_layer_excluded`` only knows the fused MLA names, so the fused
  projection is created as NVFP4 and loading fails on the BF16 tensors. A fused module whose parts (the model's
  ``packed_modules_mapping``) are all excluded is excluded. Generic; a candidate for upstream.
- DSA attention on SM120: SGLang's FlashInfer sparse MLA backend (the SM120 path; the default TRT-LLM MLA kernels are
  SM100 only) accepts only the GLM-5 DSA architectures. GLM-5.3 Flash (``Glm5NextForConditionalGeneration`` and its
  NextN draft) uses the same DSA attention and passes the check as GLM-5.
- NextN draft quantization: ``DeepseekModelNextN.__init__`` drops a ``modelopt_fp4`` quantization config (DeepSeek
  NVFP4 checkpoints keep their NextN layer in BF16), but GLM-5.3's NextN experts are NVFP4: built as BF16, loading
  fails on the packed tensors. For a GLM-5.3 draft the check does not fire (the config keeps its methods), unless the
  checkpoint excludes the NextN layer from its quantization (NVIDIA's ModelOpt checkpoint: BF16); with XTIER_DRAFT=1
  the CPU engine then takes those experts as NVFP4 (formats.Bf16AsNvFp4).
- FP8 storage of the BF16 attention and dense weights (XTIER_FP8_WEIGHTS=1, fp8_weights.py): ~15 GB read every
  decode step become ~7.5 GB; W8A16 decode kernel, W8A8 prefill for XTIER_FP8_W8A8_PREFILL.
- DSA on a BF16 KV cache with SGLang's TileLang backend on GPUs with less than Hopper's shared memory per block
  (RTX 30 / 40 / 50 series, RTX PRO; 99-101 KB): its CUDA kernel for that case (sparse_attention_fwd_kernel_v2) is
  built for Hopper's 227 KB and asks for 256 KB. Those GPUs take SGLang's split kernels instead, the ones its ROCm
  and FP8-KV paths use (partial attention over top-k blocks, then the log-sum-exp merge; blocks of 32, 128 threads,
  at least two blocks per program, so Q is not staged in shared memory): the same values up to the order of sums.
- The DSA indexer's key pool below SM89 (kpool_fp8.py, RTX 30 series): SGLang's four kernels that write it store FP8
  through float8e4nv; rebuilt from its source with that store encoding the bits in software.
- The DSA indexer below SM90 (indexer_torch.py): SGLang's k-pool indexer calls DeepGEMM directly (its prefill and
  decode logits, their schedule, the SM count), which asserts "Unsupported architecture" there; those four in torch,
  the scores in FP32.
"""

import functools
import os

import torch

EXCLUDED = 'sglang.srt.layers.quantization.modelopt_quant.ModelOptQuantConfig.is_layer_excluded'
SPARSE_MLA = 'sglang.kernels.ops.attention.flash_mla_sm120._validate_flashinfer_sparse_mla_backend'
AS_GLM5_DSA = {'Glm5NextForConditionalGeneration': 'GlmMoeDsaForCausalLM',
               'Glm5NextForConditionalGenerationNextN': 'GlmMoeDsaForCausalLMNextN'}


def _validate_sparse_mla(original, *args, **kwargs):
    selected = {kwargs.get('prefill_impl'), kwargs.get('decode_impl')}
    if 'model_arch' in kwargs and 'flashinfer_sparse_mla' in selected:   # other backends keep their own checks
        kwargs['model_arch'] = AS_GLM5_DSA.get(kwargs['model_arch'], kwargs['model_arch'])
    return original(*args, **kwargs)


def _is_layer_excluded(original, self, prefix, *args, **kwargs):
    if original(self, prefix, *args, **kwargs):
        return True
    head, _, tail = prefix.rpartition('.')
    parts = (getattr(self, 'packed_modules_mapping', None) or {}).get(tail)
    return bool(parts and head) and all(original(self, f'{head}.{p}', *args, **kwargs) for p in parts)


NEXTN_MODEL = 'sglang.srt.models.deepseek_nextn.DeepseekModelNextN.__init__'
TILELANG_DSA = 'sglang.kernels.ops.attention.dsa.tilelang_kernel.tilelang_sparse_fwd'
HOPPER_SMEM = 232448   # bytes of shared memory per block (opt-in) the BF16 kernel's tiles are sized for


def _is_glm5_next(config):
    archs = ' '.join(getattr(config, 'architectures', None) or [])
    return 'Glm5Next' in archs or str(getattr(config, 'model_type', '')).startswith('glm5_next')


def _nextn_model_init(original, self, config, quant_config=None, *args, **kwargs):
    if quant_config is None or quant_config.get_name() != 'modelopt_fp4' or not _is_glm5_next(config):
        return original(self, config, quant_config, *args, **kwargs)
    if quant_config.is_layer_excluded(f'model.layers.{config.num_hidden_layers}.mlp.experts'):
        return original(self, config, quant_config, *args, **kwargs)   # a BF16 NextN layer: SGLang drops the config
    quant_config.get_name = lambda: 'modelopt_fp4 (GLM-5.3 NextN)'   # shadows the method while the draft is built
    try:
        return original(self, config, quant_config, *args, **kwargs)
    finally:
        del quant_config.get_name


@functools.lru_cache(maxsize=None)
def _small_smem(device):
    return torch.cuda.get_device_properties(device).shared_memory_per_block_optin < HOPPER_SMEM


def _tilelang_sparse_fwd(original, q, kv, indices, sm_scale, d_v=512):
    if kv.dtype not in (torch.bfloat16, torch.float16) or not _small_smem(q.device):
        return original(q, kv, indices, sm_scale, d_v)
    from sglang.kernels.ops.attention.dsa import tilelang_kernel as tk
    num_heads, tail_dim, topk = q.shape[1], q.shape[2] - d_v, indices.shape[-1]
    block_I, threads, block_per_cu = 32, 128, 1
    ni = topk // block_I
    # at least 2 top-k blocks per program: with one the kernel keeps Q (64 heads x 576) in shared memory, 100 KB with
    # its KV tiles when built for sm_89
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    inner_iter = max(2, tk._pick_inner_iter(q.shape[0], ni, sms, block_per_cu))
    partial = tk.sparse_mla_fwd_decode_partial(num_heads, d_v, tail_dim, topk, sm_scale=sm_scale, block_I=block_I,
                                               inner_iter=inner_iter, threads=threads)
    partial_o, partial_lse = partial(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))
    combine = tk.sparse_mla_fwd_decode_combine(num_heads, d_v, ni // inner_iter * block_I, head_per_block=4,
                                               block_I=block_I, threads=threads)
    return combine(partial_o, partial_lse)


def install(registry, hook_type):
    registry.register(EXCLUDED, _is_layer_excluded, hook_type.AROUND)
    registry.register(SPARSE_MLA, _validate_sparse_mla, hook_type.AROUND)
    registry.register(NEXTN_MODEL, _nextn_model_init, hook_type.AROUND)
    registry.register(TILELANG_DSA, _tilelang_sparse_fwd, hook_type.AROUND)   # BF16 KV on small-shared-memory GPUs only
    if torch.cuda.get_device_capability()[0] < 9:
        from xtier.packs.glm5_next import indexer_torch
        for target, hook in indexer_torch.HOOKS.items():
            registry.register(target, hook, hook_type.AROUND)
    from xtier.arch import HW_FP8
    if not HW_FP8:
        from xtier.packs.glm5_next import kpool_fp8
        for name in kpool_fp8.LAUNCHERS:
            registry.register(f'{kpool_fp8.MODULE}.{name}', kpool_fp8.launcher, hook_type.AROUND)
    if os.environ.get('XTIER_FP8_WEIGHTS') == '1':
        from xtier.packs.glm5_next import fp8_weights
        method = 'sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod'
        registry.register(f'{method}.apply', fp8_weights.apply, hook_type.AROUND)
        registry.register(f'{method}.apply_into', fp8_weights.apply_into, hook_type.AROUND)
        registry.register(f'{method}.apply_with_addend', fp8_weights.apply_with_addend, hook_type.AROUND)
        registry.register('sglang.srt.model_loader.loader.DefaultModelLoader.load_weights_and_postprocess',
                          fp8_weights.load_weights_and_postprocess, hook_type.AROUND)
