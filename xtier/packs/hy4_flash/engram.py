"""Hunyuan-4-Flash Engram rows on NVMe (XTIER_ENGRAM_ROWS=nvme) with async side-stream prefetch.

Hy4-Flash places four FP8 Engram tables (~165M x 96 F8_E4M3 rows + per-row BF16 weight_scale_inv,
64.7 GB total) on layers 1, 11, 21 and 31 (1-based: 2, 12, 22, 32). Upstream SGLang materializes all
four tables in device VRAM, which exceeds a single 72 GB GPU.

Here each FlashEngramEmbedding opens an XTier RowStore over its checkpoint shard (96 B FP8 row +
2 B BF16 scale), preloads the 1.32 GB of per-row BF16 scales into host RAM, keeps a bounded direct-
mapped row cache (XTIER_ENGRAM_CACHE_GIB split across the four layers), and skips the GPU weight
allocation in FlashEngramEmbedding.load. Immediately after FlashEngramHasher.forward computes all
four layers' hashes [T, 4, 16], a single shared side CUDA stream prefetches layers 1, 11, 21 and 31
in strict layer order so layer 1 finishes during layer 0 and the remaining three overlap layers 1..30
without contending on the row-store I/O lock.
"""

import logging
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from xtier.arch import HW_FP8, e4m3_f32

logger = logging.getLogger(__name__)

ROW_BYTES = 96
SCALE_BYTES = 2
ENGRAM_LAYERS = 4
_FP32_EPS = torch.finfo(torch.float32).eps

_engram_layers = {}        # hash_index (0..3) -> FlashEngram instance
_prefetch_stream = [None]  # single shared side stream: preserves L1 -> L11 -> L21 -> L31 order


def _model_dir():
    path = os.environ.get('XTIER_ENGRAM_MODEL')
    if not path:
        from sglang.srt.runtime_context import get_model
        path = get_model().model_path
    return path


def open_table(model, layer_id, cache_bytes, row_range=None):
    """RowStore for layer `layer_id`'s Hy4-Flash FP8 Engram table in checkpoint `model`."""
    from xtier.tables import RowStore, Segment, tensor_locations

    prefix = f'model.layers.{layer_id}.engram_layer.multi_head_embedding.embedding'
    wname, sname = f'{prefix}.weight', f'{prefix}.weight_scale_inv'
    found = tensor_locations(model, (wname, sname))
    (wpath, wshape, wdtype, woff), (spath, sshape, sdtype, soff) = found[wname], found[sname]
    rows = wshape[0]
    if (wpath != spath or wshape != [rows, ROW_BYTES] or wdtype != 'F8_E4M3'
            or sshape != [rows] or sdtype != 'BF16'):
        raise ValueError(
            f'xtier: Hy4-Flash Engram layer {layer_id}: unexpected table '
            f'{wshape} {wdtype} / {sshape} {sdtype} ({wpath} vs {spath})'
        )
    return RowStore([Segment(wpath, rows, woff, soff)], ROW_BYTES, SCALE_BYTES, cache_bytes, row_range)


@triton.jit
def _fetched_engram_norm_kernel(
    hash_ptr,       # [T, n_hash] int (original hashes; < 0 -> zero row)
    hash_stride,
    rows_ptr,       # [T * HEADS, D] fp8 or uint8 (fetched sequentially by GraphLookup)
    scale_ptr,      # [T * HEADS] bf16 per-row scale
    gamma_ptr,      # [HEADS, D] bf16 lookup_gamma
    out_ptr,        # [T, HEADS * D] bf16
    head_start,
    eps,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SOFT_FP8: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    mask = d < D
    idx = tl.load(hash_ptr + t * hash_stride + head_start + h).to(tl.int64)
    row_i = t * HEADS + h
    if SOFT_FP8:
        x = e4m3_f32(tl.load(rows_ptr + row_i * D + d, mask=mask, other=0))
    else:
        x = tl.load(rows_ptr + row_i * D + d, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + row_i).to(tl.float32)
    x = tl.where(idx < 0, 0.0, x * s)
    gamma = tl.load(gamma_ptr + h * D + d, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / D + eps) * gamma
    tl.store(out_ptr + t * HEADS * D + h * D + d, y.to(tl.bfloat16), mask=mask)


def _lookup_from_store(engram, hashes):
    """Fetch local hash heads from RowStore and apply per-head RMSNorm -> [T, local_heads * 96] BF16."""
    table = engram.multi_head_embedding
    tokens = hashes.shape[0]
    heads = engram.local_heads
    dim = table.dim
    out = torch.empty((tokens, heads * dim), dtype=torch.bfloat16, device=hashes.device)
    if tokens == 0 or heads == 0:
        return out
    local = hashes[:, table.head_start : table.head_end].to(torch.int64)
    # RowStore aborts on negative ids; subtract vocab_start and clamp to [0, num_embeddings - 1]
    # for the I/O lookup, then mask idx < 0 to zero inside _fetched_engram_norm_kernel.
    safe = (local - table.vocab_start).clamp(0, table.num_embeddings - 1).contiguous()
    rows_u8, scales_u8, _ = table._xtier_lookup.fetch(safe)
    rows_view = rows_u8 if not HW_FP8 else rows_u8.view(torch.float8_e4m3fn)
    scales_bf16 = scales_u8.view(torch.bfloat16).view(-1)
    _fetched_engram_norm_kernel[(tokens, heads)](
        hashes,
        hashes.stride(0),
        rows_view,
        scales_bf16,
        engram.lookup_gamma,
        out,
        table.head_start,
        _FP32_EPS,
        HEADS=heads,
        D=dim,
        BLOCK_D=triton.next_power_of_2(dim),
        SOFT_FP8=not HW_FP8,
        num_warps=1,
    )
    return out


def _view_key(t):
    return t.data_ptr(), tuple(t.shape), tuple(t.stride())


def _flash_engram_init(result, self, config, layer_number, prefix, *args, **kwargs):
    """After FlashEngram.__init__: attach an NVMe RowStore + GraphLookup to multi_head_embedding."""
    from xtier.tables.lookup import GraphLookup

    table = self.multi_head_embedding
    if table.dim != ROW_BYTES:
        raise ValueError(f'xtier: Hy4-Flash Engram NVMe rows expect dim {ROW_BYTES}, got {table.dim}')
    layer_id = int(layer_number) - 1
    engram_ids = list(getattr(config, 'engram_layer_ids', None) or ())
    n_layers = max(1, len(engram_ids) or ENGRAM_LAYERS)
    hash_index = engram_ids.index(layer_number)
    total_cache = int(float(os.environ.get('XTIER_ENGRAM_CACHE_GIB', '4')) * (2**30))
    # Layer 1 (hash_index == 0) only overlaps with dense layer 0 (~0.25 ms), whereas layers 11/21/31
    # hide behind 10 MoE layers (~3.5-4.5 ms). Prioritize layer 1 up to a collision-free full cache.
    if n_layers > 1 and total_cache > 0:
        full_l1 = table.num_embeddings * (ROW_BYTES + SCALE_BYTES + 8)
        l1_share = 0.55 if total_cache >= 20 * (2**30) else 0.40
        l1_bytes = min(full_l1, int(total_cache * l1_share))
        cache_bytes = l1_bytes if hash_index == 0 else max(0, total_cache - l1_bytes) // (n_layers - 1)
    else:
        cache_bytes = total_cache // n_layers
    store = open_table(_model_dir(), layer_id, cache_bytes, (table.vocab_start, table.vocab_end))
    if store.rows != table.num_embeddings:
        raise ValueError(
            f'xtier: Hy4-Flash Engram layer {layer_id}: config {table.num_embeddings} rows, '
            f'checkpoint {store.rows}'
        )
    store.preload_scales()
    lookup = GraphLookup(store)
    lookup.warm(
        int(os.environ.get('XTIER_ENGRAM_GRAPH_IDS', '16384')),
        torch.device('cuda', torch.cuda.current_device()),
    )
    table._xtier_rows = store
    table._xtier_lookup = lookup
    _engram_layers[hash_index] = self
    logger.info(
        'xtier: Hy4-Flash Engram layer %d (idx %d): %d rows [%d, %d) on NVMe (%s), '
        'row cache %.2f GiB, scales preloaded',
        layer_id, hash_index, table.num_embeddings, table.vocab_start, table.vocab_end,
        store.describe(), cache_bytes / (2**30),
    )
    return result


def _embedding_load(original, self, leaf, value):
    """Skip GPU allocation of the 64.7 GB Engram tables when backed by XTier RowStore."""
    store = getattr(self, '_xtier_rows', None)
    if store is None:
        return original(self, leaf, value)
    if value.shape[0] != self.num_embeddings:
        raise ValueError(f'Invalid Engram table rows: {tuple(value.shape)}')
    return None


def _hasher_prefetch(hash_ids, self, input_ids, *args, **kwargs):
    """After FlashEngramHasher.forward: prefetch all 4 Engram layers in order on one side stream."""
    if not _engram_layers or hash_ids.dim() != 3 or hash_ids.shape[0] == 0 or not hash_ids.is_cuda:
        return hash_ids
    if _prefetch_stream[0] is None:
        _prefetch_stream[0] = torch.cuda.Stream(device=hash_ids.device)
    side = _prefetch_stream[0]
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    hash_ids.record_stream(side)
    for index in sorted(_engram_layers):
        if index >= hash_ids.shape[1]:
            continue
        engram = _engram_layers[index]
        view = hash_ids[:, index]
        # Run view.contiguous() on `side` (after side.wait_stream(main)) so the copy and lookup are
        # strictly ordered on the same stream with zero cross-stream race.
        with torch.cuda.stream(side):
            hashes_c = view.contiguous()
            x = _lookup_from_store(engram, hashes_c)
            done = torch.cuda.Event()
            done.record(side)
        engram._xtier_prefetched = (_view_key(view), x, done, hashes_c)
    return hash_ids


def _flash_engram_forward(original, self, hidden, hashes):
    """Run FlashEngram.forward using the NVMe RowStore (consuming prefetched output when ready)."""
    table = self.multi_head_embedding
    if getattr(table, '_xtier_rows', None) is None or hidden.shape[0] == 0 or not hidden.is_cuda:
        return original(self, hidden, hashes)

    from sglang.kernels.ops.embeddings.hy4_flash_engram import hy4_flash_engram_combine
    from sglang.srt.models.hunyuan_v4_flash import _maybe_all_reduce

    x = None
    prefetched = getattr(self, '_xtier_prefetched', None)
    if prefetched is not None:
        self._xtier_prefetched = None
        key, out, done, _ = prefetched
        main = torch.cuda.current_stream()
        main.wait_event(done)
        out.record_stream(main)
        if key == _view_key(hashes):
            x = out
    if x is None:
        x = _lookup_from_store(self, hashes.contiguous())

    if self._fused_kv is None:
        self._fused_kv = (
            torch.cat([p.weight for p in self.key_projs] + [self.value_proj.weight]),
            torch.cat([p.bias for p in self.key_projs] + [self.value_proj.bias]),
        )
    kv = _maybe_all_reduce(F.linear(x, *self._fused_kv))
    return hy4_flash_engram_combine(
        kv,
        hidden,
        hashes,
        self.norm_key,
        self.norm_query,
        self.unified_pre_gamma,
        self.unified_per_hc_scales,
    )


def install(registry, hook_type):
    if os.environ.get('XTIER_ENGRAM_ROWS', '') != 'nvme':
        return
    base = 'sglang.srt.models.hunyuan_v4_flash'
    registry.register(f'{base}.FlashEngram.__init__', _flash_engram_init, hook_type.AFTER)
    registry.register(f'{base}.FlashEngramEmbedding.load', _embedding_load, hook_type.AROUND)
    registry.register(f'{base}.FlashEngram.forward', _flash_engram_forward, hook_type.AROUND)
    if os.environ.get('XTIER_ENGRAM_PREFETCH', '1') != '0':
        registry.register(f'{base}.FlashEngramHasher.forward', _hasher_prefetch, hook_type.AFTER)
