"""Acceleration pack for SGLang's ``kimi_k3`` models (Kimi K3.1 Pro: ``KimiK31ForConditionalGeneration``).

- DSA backends on SM120 (RTX 50 series, RTX PRO Blackwell) and below SM90: SGLang pins Kimi-K3.1's DSA to FlashMLA
  (``flashmla_kv``, the reference fp8_ds_mla numerics), which has no build for those GPUs ("no kernel image is
  available"). There the pin becomes SGLang's Triton backend, the one it pins on ROCm: it reads the same fp8_ds_mla
  rows and runs BF16 dots. Resolved with the server arguments, in the process that serves the API; the GPU is asked
  through NVML there (no CUDA context in that process).
- Triton sparse MLA tiles (prefill and decode): SGLang sizes BLOCK_K for the AMD staging (one KV row and its index
  per key). On CUDA the two pipeline stages and the Q tile take more: 64 keys need 110,336 bytes on SM120, whose limit
  is 101,376. The shared memory left after a 24 KB Q tile is halved for the two stages (SM120: 32 keys; Hopper and
  SM100 keep 64).
- Long prompts: SGLang's single-pass sparse MLA prefill kernel holds every head of a token in one CTA (a power of two
  of them): meant for a few heads per rank, it needs 150 KB at Kimi's 96 heads (TP1, padded to 128). Above 16 heads
  such prompts go to the split-K kernel with one split instead (heads in blocks of 16), as SGLang does on gfx950.
- 4-bit dense projections, NVFP4 or MXFP4 (a checkpoint variant from tools/kimi_mxfp4_dense.py, marked ``xtier_dense_mxfp4`` in its
  quantization config): SGLang's W4A16 linear schemes name their packed weight ``weight_packed``, but Kimi's
  loader renames every ``weight_packed`` tensor to ``weight`` (meant for the routed experts): the dense parameter is
  registered as ``weight`` and named back before its post-processing. The variant keeps the source shards, whose BF16
  originals of the converted projections are dropped from the loader's stream. With ``latent`` the quantization
  config allows the latent MoE projections (SGLang keeps them BF16 unless it says so).
"""

import functools
import logging

logger = logging.getLogger(__name__)

OVERRIDES = 'sglang.srt.arg_groups.overrides'
TSM = 'sglang.kernels.ops.attention.dsa.triton_sparse_mla'
TRITON = 'triton'
Q_TILE_BYTES = 24 * 1024


@functools.lru_cache(maxsize=None)
def _compute_capability():
    """(major, minor) of GPU 0 without a CUDA context (NVML), else from torch; None without a GPU."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            return tuple(pynvml.nvmlDeviceGetCudaComputeCapability(pynvml.nvmlDeviceGetHandleByIndex(0)))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available() and torch.version.hip is None:
            return torch.cuda.get_device_capability(0)
    except Exception:
        pass
    return None


def _no_flashmla():
    cap = _compute_capability()
    return cap is not None and (cap[0] < 9 or cap[0] == 12)


def _dsa_backends(original, *args, **kwargs):
    import importlib
    overrides = importlib.import_module(OVERRIDES)
    pinned = overrides._KIMI_K31_DSA_BACKEND
    if not _no_flashmla() or pinned == TRITON:
        return original(*args, **kwargs)
    overrides._KIMI_K31_DSA_BACKEND = TRITON
    try:
        logger.info('xtier kimi_k3: DSA on the Triton backend (FlashMLA has no build for sm_%s%s)',
                    *_compute_capability())
        return original(*args, **kwargs)
    finally:
        overrides._KIMI_K31_DSA_BACKEND = pinned


def _block_k_for_shared_memory(original, kv_dim, element_size, max_shared_memory):
    import torch
    if max_shared_memory is not None and torch.version.hip is None:
        max_shared_memory = max(0, (max_shared_memory - Q_TILE_BYTES) // 2)
    return original(kv_dim, element_size, max_shared_memory)


def _single_pass_prefill(original, q_nope, q_rope, kv, indices, sm_scale, d_v=512):
    if q_nope.shape[1] <= 16:
        return original(q_nope, q_rope, kv, indices, sm_scale, d_v)
    import importlib
    tsm = importlib.import_module(TSM)
    return tsm._triton_sparse_mla_fwd_splitk(q_nope, q_rope, kv, indices, sm_scale, d_v, kv_splits=1)


SCHEMES = 'sglang.srt.layers.quantization.compressed_tensors.schemes'
DENSE_SCHEMES = (f'{SCHEMES}.compressed_tensors_w4a16_mxfp4.CompressedTensorsW4A16Mxfp4',
                 f'{SCHEMES}.compressed_tensors_w4a16_nvfp4.CompressedTensorsW4A16Fp4')
KIMI = 'sglang.srt.models.kimi_k3'


_VARIANT = []


def _variant():
    """(the variant's settings, its converted projections) when the served model is a dense-MXFP4 variant, else None.
    Asked once the runtime context is published (SGLang's get_server_args); an answer is kept, a failure is not."""
    if _VARIANT:
        return _VARIANT[0]
    import json
    import os
    try:
        from sglang.srt.runtime_context import get_server_args
        root = get_server_args().model_path
    except Exception:
        return None
    qc = json.load(open(os.path.join(root, 'config.json'))).get('text_config', {}).get('quantization_config', {})
    if 'xtier_dense_mxfp4' not in qc:
        _VARIANT.append(None)
        return None
    wm = json.load(open(os.path.join(root, 'model.safetensors.index.json')))['weight_map']
    converted = frozenset(n[:-len('.weight_packed')] for n in wm
                          if n.endswith('.weight_packed') and '.experts.' not in n)
    logger.info('xtier kimi_k3: dense MXFP4 variant, %d projections (%s)', len(converted), qc['xtier_dense_mxfp4'])
    _VARIANT.append((qc['xtier_dense_mxfp4'], converted))
    return _VARIANT[0]


def _dense_create(result, self, layer, *args, **kwargs):
    if _variant() is not None and 'weight_packed' in layer._parameters:
        layer.register_parameter('weight', layer._parameters.pop('weight_packed'))
    return result


def _dense_process(original, self, layer):
    if 'weight_packed' not in layer._parameters and 'weight' in layer._parameters:
        layer.register_parameter('weight_packed', layer._parameters.pop('weight'))
    return original(self, layer)


def _load_weights(original, self, weights, *args, **kwargs):
    v = _variant()
    if v is None:
        return original(self, weights, *args, **kwargs)
    converted = v[1]

    def stream():
        dropped = 0
        for name, w in weights:
            if name.endswith('.weight') and name[:-len('.weight')] in converted and w.dtype != __import__('torch').uint8:
                dropped += 1
                continue
            yield name, w
        logger.info('xtier kimi_k3: %d BF16 originals of MXFP4 projections not loaded', dropped)

    return original(self, stream(), *args, **kwargs)


def _moe_init(original, self, config, *args, **kwargs):
    v = _variant()
    qc = kwargs.get('quant_config', args[0] if args else None)
    if v is not None and v[0].get('latent') and qc is not None:
        qc.supports_kimi_k3_quantized_latent_projections = True
    return original(self, config, *args, **kwargs)


def install(registry, hook_type):
    registry.register(f'{TSM}._block_k_for_shared_memory', _block_k_for_shared_memory, hook_type.AROUND)
    registry.register(f'{TSM}._triton_sparse_mla_fwd_single', _single_pass_prefill, hook_type.AROUND)
    for scheme in DENSE_SCHEMES:
        registry.register(f'{scheme}.create_weights', _dense_create, hook_type.AFTER)
        registry.register(f'{scheme}.process_weights_after_loading', _dense_process, hook_type.AROUND)
    registry.register(f'{KIMI}.KimiK3ForConditionalGeneration.load_weights', _load_weights, hook_type.AROUND)
    registry.register(f'{KIMI}.KimiK3MoE.__init__', _moe_init, hook_type.AROUND)


def install_serving(registry, hook_type):
    registry.register(f'{OVERRIDES}._kimi_k31_dsa_backends', _dsa_backends, hook_type.AROUND)
    return True
