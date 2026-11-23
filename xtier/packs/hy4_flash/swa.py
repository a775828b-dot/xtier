"""Hybrid SWA + DSA MLA KV pool for Hunyuan-4-Flash (``XTIER_HY4_HYBRID_SWA=1``, default on).

Hunyuan-4-Flash has 12 ``deepseek_sparse_attention`` layers and 30 ``sliding_attention`` layers
(``sliding_window = 512``). Upstream SGLang only activates its hybrid ``SWAKVPool`` when
``layer_types`` mixes ``full_attention`` and ``sliding_attention``, so all 30 sliding layers were
allocated with full 1M KV slots AND full 1M indexer-key buffers (23.1 GiB wasted out of 32.3 GiB
at 1M tokens).

This module opts ``HYV4FlashTextConfig`` into SGLang's existing hybrid MLA/DSA ``SWAKVPool`` +
``SWATokenToKVPoolAllocator`` + ``UnifiedRadixCache`` (``SWAComponent``):
1. ``HYV4FlashTextConfig.__init__`` (AFTER, serving + scheduler): set ``is_hybrid_swa = True`` and
   ``hybrid_layer_pattern`` (1 for ``sliding_attention``, 0 for ``deepseek_sparse_attention``).
2. ``SWAKVPool.__init__`` (AROUND) & ``register_mapping`` (AFTER): propagate ``use_dsa=True`` and
   ``override_kv_cache_dim`` from ``full_kv_pool_kwargs`` into ``swa_kv_pool_kwargs`` so sliding
   MLA layers use the same FP8/NVFP4 byte layout as the DSA layers, mirror ``index_page_size`` onto
   ``SWAKVPool``, and attach ``full_to_swa_index_mapping`` onto ``swa_kv_pool``.
3. ``KVLayer._write_loc`` (AROUND): ``DeepseekSparseAttnBackend`` passes
   ``KVWriteLoc.for_layer(forward_batch, layer)`` with ``window_locs=None``; translate window-layer
   write slots through ``full_to_swa_index_mapping`` (fixed GPU tensor updated in place by the
   allocator, CUDA-graph safe).
4. ``FlashWindowIndexer.forward`` (AROUND): remap full-pool window slots returned by
   ``window_indices`` (in PAGED mode) through ``full_to_swa_index_mapping`` (trailing ``-1``
   sentinel preserves masked ``-1`` entries in one gather) so attention indexes ``swa_kv_pool``.

When ``XTIER_HY4_HYBRID_SWA=0``, ``dsa_layer_skips_topk`` still elides the unused ``index_key_cache``
on the 30 sliding-window layers (+3.86 GiB saved at 1M tokens).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

TEXT_CONFIG_INIT = 'sglang.srt.configs.hy_v4_flash.HYV4FlashTextConfig.__init__'
SWA_POOL_INIT = 'sglang.srt.mem_cache.swa_memory_pool.SWAKVPool.__init__'
SWA_REGISTER_MAPPING = 'sglang.srt.mem_cache.swa_memory_pool.SWAKVPool.register_mapping'
KV_LAYER_WRITE_LOC = 'sglang.srt.mem_cache.kv.router.KVLayer._write_loc'
WINDOW_INDEXER_FWD = 'sglang.srt.models.hunyuan_v4_flash.FlashWindowIndexer.forward'
DSA_SKIPS_TOPK = 'sglang.srt.configs.model_config.dsa_layer_skips_topk'


def _hybrid_enabled() -> bool:
    return os.environ.get('XTIER_HY4_HYBRID_SWA', '1') != '0'


def _text_config_init(result, self, *args, **kwargs):
    if _hybrid_enabled():
        layer_types = getattr(self, 'layer_types', None) or ()
        if 'sliding_attention' in layer_types and 'deepseek_sparse_attention' in layer_types:
            self.is_hybrid_swa = True
            self.hybrid_layer_pattern = [1 if lt == 'sliding_attention' else 0 for lt in layer_types]
    return result


def _swa_pool_init(original, self, *args, **kwargs):
    full_cls = kwargs.get('full_kv_pool_class')
    full_kw = kwargs.get('full_kv_pool_kwargs')
    swa_kw = kwargs.get('swa_kv_pool_kwargs')
    if (full_cls is not None and isinstance(full_kw, dict) and isinstance(swa_kw, dict)
            and 'kv_cache_dim' in full_kw and 'override_kv_cache_dim' not in swa_kw):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
        if isinstance(full_cls, type) and issubclass(full_cls, DSATokenToKVPool):
            swa_kw = dict(swa_kw, use_dsa=True, override_kv_cache_dim=full_kw['kv_cache_dim'])
            kwargs = dict(kwargs, swa_kv_pool_kwargs=swa_kw)
    res = original(self, *args, **kwargs)
    if hasattr(self.full_kv_pool, 'index_page_size'):
        self.index_page_size = self.full_kv_pool.index_page_size
    if hasattr(self.full_kv_pool, 'kv_scale_ue8m0') and hasattr(self.swa_kv_pool, 'kv_scale_ue8m0'):
        self.swa_kv_pool.kv_scale_ue8m0 = self.full_kv_pool.kv_scale_ue8m0
    return res


def _swa_register_mapping(result, self, full_to_swa_index_mapping):
    if getattr(self, 'swa_kv_pool', None) is not None:
        self.swa_kv_pool._xtier_full_to_swa = full_to_swa_index_mapping
    return result


def _kv_layer_write_loc(original, self, layer, loc_info):
    if not self.owner:
        from sglang.srt.mem_cache.memory_pool import (
            KVWriteLoc,
            unwrap_write_loc,
            write_loc_is_physical,
        )
        loc, window_locs, _ = unwrap_write_loc(loc_info)
        if window_locs is None or self.route.sub_pool not in window_locs:
            lut = getattr(self.view, '_xtier_full_to_swa', None)
            if lut is None:
                from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
                lut = getattr(get_token_to_kv_pool(), 'full_to_swa_index_mapping', None)
            if lut is not None:
                return KVWriteLoc(lut[loc], physical=write_loc_is_physical(loc_info))
    return original(self, layer, loc_info)


def _window_indexer_forward(
    original, self, x, q_lora, positions, forward_batch, layer_id, return_indices=True
):
    idx = original(self, x, q_lora, positions, forward_batch, layer_id, return_indices=return_indices)
    if idx.numel() == 0:
        return idx
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
    from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool
    if get_attn_backend().get_topk_transform_method(forward_batch.forward_mode) == TopkTransformMethod.RAGGED:
        return idx
    lut = getattr(get_token_to_kv_pool(), 'full_to_swa_index_mapping', None)
    if lut is not None:
        # lut carries a trailing -1 sentinel so masked -1 entries stay -1 in one gather.
        idx = lut[idx].to(torch.int32)
    return idx


def _dsa_layer_skips_topk(original, config, layer_id):
    layer_types = getattr(config, 'layer_types', None)
    if layer_types is not None and 0 <= layer_id < len(layer_types) and layer_types[layer_id] == 'sliding_attention':
        return True
    return original(config, layer_id)


def install_serving(registry, hook_type):
    if _hybrid_enabled():
        registry.register(TEXT_CONFIG_INIT, _text_config_init, hook_type.AFTER)


def install(registry, hook_type):
    registry.register(DSA_SKIPS_TOPK, _dsa_layer_skips_topk, hook_type.AROUND)
    if _hybrid_enabled():
        registry.register(TEXT_CONFIG_INIT, _text_config_init, hook_type.AFTER)
        registry.register(SWA_POOL_INIT, _swa_pool_init, hook_type.AROUND)
        registry.register(SWA_REGISTER_MAPPING, _swa_register_mapping, hook_type.AFTER)
        registry.register(KV_LAYER_WRITE_LOC, _kv_layer_write_loc, hook_type.AROUND)
        registry.register(WINDOW_INDEXER_FWD, _window_indexer_forward, hook_type.AROUND)
