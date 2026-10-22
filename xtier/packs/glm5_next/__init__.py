"""Acceleration pack for SGLang's ``glm5_next`` models (GLM-5.3 Flash).

- ModelOpt exclusions of fused modules: the GLM-5.3 Flash NVFP4 checkpoint keeps its attention projections in BF16
  and lists them unfused (``*.self_attn.q_proj``, ``k_proj``, ``v_proj``, ...), while SGLang loads them fused
  (``self_attn.qkv_proj``). ``ModelOptQuantConfig.is_layer_excluded`` only knows the fused MLA names, so the fused
  projection is created as NVFP4 and loading fails on the BF16 tensors. A fused module whose parts (the model's
  ``packed_modules_mapping``) are all excluded is excluded. Generic; a candidate for upstream.
- DSA attention on SM120: SGLang's FlashInfer sparse MLA backend (the SM120 path; the default TRT-LLM MLA kernels are
  SM100 only) accepts only the GLM-5 DSA architectures. GLM-5.3 Flash (``Glm5NextForConditionalGeneration`` and its
  NextN draft) uses the same DSA attention and passes the check as GLM-5.
"""

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


def install(registry, hook_type):
    registry.register(EXCLUDED, _is_layer_excluded, hook_type.AROUND)
    registry.register(SPARSE_MLA, _validate_sparse_mla, hook_type.AROUND)
