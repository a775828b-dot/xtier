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
  fails on the packed tensors. For a GLM-5.3 draft the check does not fire (the config keeps its methods).
- FP8 storage of the BF16 attention and dense weights (XTIER_FP8_WEIGHTS=1, fp8_weights.py): ~15 GB read every
  decode step become ~7.5 GB; W8A16 decode kernel, W8A8 prefill for XTIER_FP8_W8A8_PREFILL.
"""

import os

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


def _is_glm5_next(config):
    archs = ' '.join(getattr(config, 'architectures', None) or [])
    return 'Glm5Next' in archs or str(getattr(config, 'model_type', '')).startswith('glm5_next')


def _nextn_model_init(original, self, config, quant_config=None, *args, **kwargs):
    if quant_config is None or quant_config.get_name() != 'modelopt_fp4' or not _is_glm5_next(config):
        return original(self, config, quant_config, *args, **kwargs)
    quant_config.get_name = lambda: 'modelopt_fp4 (GLM-5.3 NextN)'   # shadows the method while the draft is built
    try:
        return original(self, config, quant_config, *args, **kwargs)
    finally:
        del quant_config.get_name


def install(registry, hook_type):
    registry.register(EXCLUDED, _is_layer_excluded, hook_type.AROUND)
    registry.register(SPARSE_MLA, _validate_sparse_mla, hook_type.AROUND)
    registry.register(NEXTN_MODEL, _nextn_model_init, hook_type.AROUND)
    if os.environ.get('XTIER_FP8_WEIGHTS') == '1':
        from xtier.packs.glm5_next import fp8_weights
        method = 'sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod'
        registry.register(f'{method}.apply', fp8_weights.apply, hook_type.AROUND)
        registry.register(f'{method}.apply_into', fp8_weights.apply_into, hook_type.AROUND)
        registry.register(f'{method}.apply_with_addend', fp8_weights.apply_with_addend, hook_type.AROUND)
        registry.register('sglang.srt.model_loader.loader.DefaultModelLoader.load_weights_and_postprocess',
                          fp8_weights.load_weights_and_postprocess, hook_type.AROUND)
