"""Acceleration pack for SGLang's ``qwen4_exp`` models (Qwen3.8 Flash Next).

- PLE host table: the pinned backend allocates the n-gram embedding table (320M rows x 160 FP8, 48 GiB) as an
  exact-size, NUMA-interleaved, CUDA-registered mapping (xtier.engine.memory) instead of the caching host
  allocator's power-of-two block on one node, which cannot be placed on an NPS4 host.
- PLE rows on NVMe (``XTIER_PLE_ROWS=nvme``, ple_rows.py): no table in memory at all, the rows read from the
  checkpoint shards through a row table with a bounded cache.
- Hyper-connection mix of decode batches: the overlapped persistent kernel of hc_mix.py replaces SGLang's
  ``fused_hc_mix`` (XTIER_HC_MIX_OVERLAP=0 keeps SGLang's).
- PLE layer of a long prefill chunk in token blocks (ple_blocks.py, XTIER_PLE_BLOCK, default 4096 tokens): SGLang's
  whole-chunk PLE layer keeps several [tokens x hc_hidden] BF16 tensors alive at once (about 5 GB for a 32K chunk),
  where 32K prefill chunks run out of GPU memory; per block, with the convolution state carried across blocks, the
  same values need about 2 GB.
- QSA indexer prefill logits with the keys split over CTAs (qsa_prefill.py, XTIER_QSA_KEYS_PER_CTA, default 4096): deep
  in a long prompt SGLang's kernel runs ~350 rows per launch on 11 CTAs; the split gives hundreds, same logits bitwise
  (352 rows x 95K keys: 2.40 -> 0.61 ms).
- QSA attention on an NVFP4 KV cache (qsa_fp4.py, ``--kv-cache-dtype nvfp4`` as production, XTIER_QSA_FP4=0 off):
  SGLang stores the cache but its QSA backend reads it as plain K/V; the pack's decode / verify gather and prefill
  packing dequantize straight from the packed rows (equal to SGLang's kernels on the dequantized pool, bitwise).
- QSA decode / verify attention below SM90 (qsa_decode.py; RTX 30 / 40 series): SGLang falls back to flash_attn's
  varlen kernel there, which has no head_dim 256 tile that fits their shared memory; split-key Triton kernels with the
  flash kernels' precision take its place.
"""

import logging
import os

logger = logging.getLogger(__name__)
PLE = 'sglang.srt.models.qwen4_exp_ple_table.allocate_ple_host_table'
HC_MIX = 'sglang.kernels.ops.gemm.hc_mix.fused_hc_mix'
MODELS = 'sglang.srt.models.qwen4_exp'
QSA_PREFILL = 'sglang.srt.layers.attention.qsa.mqa.tilelang_qsa_mqa_prefill'
QSA_MODULE = 'sglang.srt.layers.attention.qwen_sparse_attn_backend'
QSA_BACKEND = f'{QSA_MODULE}.QwenSparseAttnBackend'
SPARSE_ATTN = 'sglang.srt.layers.attention.qsa.sparse_attn'
KV_FP4 = 'sglang.srt.layers.quantization.fp4_kv_cache_quant_method'
KV_FP4_UTIL = 'sglang.srt.layers.quantization.kvfp4_tensor'


def _allocate_ple_host_table(original, shape, dtype, backend='pinned', table_dir=None, tag=None, *args, **kwargs):
    if backend != 'pinned' or args or kwargs:
        return original(shape, dtype, backend, table_dir, tag, *args, **kwargs)
    from xtier.engine.memory import pinned_interleaved
    return pinned_interleaved(shape, dtype)


def _fused_hc_mix(original, *args, **kwargs):
    from xtier.packs.qwen4_exp.hc_mix import fused_hc_mix
    return fused_hc_mix(original, *args, **kwargs)


def install(registry, hook_type):
    registry.register(PLE, _allocate_ple_host_table, hook_type.AROUND)
    registry.register(HC_MIX, _fused_hc_mix, hook_type.AROUND)
    if os.environ.get('XTIER_PLE_BLOCK', '4096') != '0':
        from xtier.packs.qwen4_exp import ple_blocks
        registry.register(f'{MODELS}.Qwen4ExpPLELayer.forward', ple_blocks.ple_forward, hook_type.AROUND)
    if os.environ.get('XTIER_QSA_KEYS_PER_CTA', '4096') != '0':
        from xtier.packs.qwen4_exp import qsa_prefill
        registry.register(QSA_PREFILL, qsa_prefill.qsa_mqa_prefill, hook_type.AROUND)
    if os.environ.get('XTIER_QSA_FP4', '1') != '0':   # each hook acts only on an NVFP4 KV pool
        from xtier.packs.qwen4_exp import qsa_fp4
        registry.register(f'{QSA_BACKEND}.forward_extend', qsa_fp4.forward_extend, hook_type.AROUND)
        registry.register(f'{QSA_BACKEND}._forward_paged_attention', qsa_fp4.paged_attention, hook_type.AROUND)
        registry.register(f'{SPARSE_ATTN}.pack_qsa_prefill_kv', qsa_fp4.pack_qsa_prefill_kv, hook_type.AROUND)
        registry.register(f'{SPARSE_ATTN}.qwen_sparse_kv_extraction_compact_triton', qsa_fp4.extraction,
                          hook_type.AROUND)
        registry.register(f'{KV_FP4}.NVFP4KVCacheMethod.create_buffers', qsa_fp4.create_buffers, hook_type.AFTER)
        registry.register(f'{KV_FP4_UTIL}.NVFP4KVQuantizeUtil.quantize', qsa_fp4.quantize, hook_type.AROUND)
    from xtier.packs.qwen4_exp import qsa_decode   # acts only below SM90
    registry.register(f'{QSA_MODULE}._resolve_flash_attn_varlen_func', qsa_decode.resolve, hook_type.AROUND)
    if os.environ.get('XTIER_PLE_ROWS', '') == 'nvme':
        from xtier.packs.qwen4_exp import ple_rows
        registry.register(f'{MODELS}.Qwen4ExpNGramEmbedding.__init__', ple_rows.ngram_init, hook_type.AROUND)
        registry.register(f'{MODELS}.Qwen4ExpPinnedHostEmbedding.__init__', ple_rows.host_embedding_init,
                          hook_type.AROUND)
        registry.register(f'{MODELS}.Qwen4ExpPinnedHostEmbedding.gather', ple_rows.gather, hook_type.REPLACE)
        registry.register(f'{MODELS}.Qwen4ExpForConditionalGeneration.load_weights', ple_rows.load_weights,
                          hook_type.AROUND)
