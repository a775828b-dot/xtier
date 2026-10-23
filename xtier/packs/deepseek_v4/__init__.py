"""Acceleration pack for SGLang's ``deepseek_v4`` models (DeepSeek V4.1 Flash 1017).

- Engram rows on NVMe (``XTIER_ENGRAM_ROWS=nvme``; docs/design-tables-nvme.md): an ``EngramEmbedding`` keeps no table
  in device or host memory. Its rows stay in the checkpoint file and come through a row table (xtier.tables: O_DIRECT
  reads, a row cache of ``XTIER_ENGRAM_CACHE_GIB`` over all Engram layers, the owned rows' scales preloaded); the
  owned-row gather ``_owned_rows`` becomes the graph-capturable lookup plus SGLang's ``engram_gather``. Hashing,
  gating, the projections and the TP all-reduce stay SGLang's. The checkpoint directory is the served model's
  (``XTIER_ENGRAM_MODEL`` overrides); staging for ``XTIER_ENGRAM_GRAPH_IDS`` ids (default 16384, 24 per token) is
  allocated before any graph capture.
- SM120 candidate indexer: SGLang sends every SM100-or-newer GPU to the sparse candidate table, whose DeepGEMM paged
  sparse MQA logits exist on SM100 only; on SM120 the dense-block candidates (SGLang's path below SM100, DeepGEMM's
  SM120 dense FP8 / FP4 MQA logits) serve prefill and decode.
- Tuned W8A8 block-FP8 GEMM configs (fp8_configs/, from the 1017 pipeline's bench/ds41_fp8_tune.py): SGLang has none
  for DeepSeek V4.1's 32 x 32 block projections on SM120 and runs one default tile for every shape, 1.5-4.6x slower
  at decode sizes (on the RTX PRO 5000 63 ms of a 68 ms decode step). Files of this directory win over SGLang's own.
"""

import functools
import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
ENGRAM = 'sglang.srt.layers.engram.EngramEmbedding'
INDEXER = 'sglang.srt.layers.attention.dsv4.v41_indexer.make_candidate_indexer'
FP8_CONFIGS = 'sglang.kernels.ops.gemm.fp8_kernel.get_w8a8_block_fp8_configs'
FP8_CONFIG_DIR = Path(__file__).with_name('fp8_configs')
ENGRAM_LAYERS = 2   # DeepSeek V4.1 Flash has two Engram layers (1 and 14); the cache budget is split over them
ROW_BYTES, SCALE_BYTES, BLOCK = 256, 8, 32   # an FP8 E4M3 row of 256 with one E8M0 scale per 32


def model_dir():
    path = os.environ.get('XTIER_ENGRAM_MODEL')
    if not path:
        from sglang.srt.runtime_context import get_model
        path = get_model().model_path
    return path


def open_table(model, layer_id, cache_bytes, row_range=None):
    """Row table of layer `layer_id`'s Engram embedding in checkpoint `model`."""
    from xtier.tables import RowStore, Segment, tensor_locations

    weight, scale = (f'layers.{layer_id}.engram.embed.{n}' for n in ('weight', 'scale'))
    found = tensor_locations(model, (weight, scale))
    (wpath, wshape, wdtype, woff), (spath, sshape, sdtype, soff) = found[weight], found[scale]
    rows = wshape[0]
    if wpath != spath or wshape != [rows, ROW_BYTES] or wdtype != 'F8_E4M3' or sshape != [rows, SCALE_BYTES] \
            or sdtype != 'F8_E8M0':
        raise ValueError(f'xtier: Engram layer {layer_id}: unexpected table {wshape} {wdtype} / {sshape} {sdtype}')
    return RowStore([Segment(wpath, rows, woff, soff)], ROW_BYTES, SCALE_BYTES, cache_bytes, row_range)


class EngramLookup:
    """BF16 Engram rows of device ids (zero rows outside the store's owned range), on the current stream."""

    def __init__(self, store):
        from xtier.tables.lookup import GraphLookup
        self.store, self.lookup = store, GraphLookup(store)

    def warm(self, count, device):
        self.lookup.warm(count, device)

    def __call__(self, indices):
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        out = torch.empty(*indices.shape, ROW_BYTES, dtype=torch.bfloat16, device=indices.device)
        if indices.numel():
            rows, scales, sequential = self.lookup.fetch(indices)
            engram_gather(rows.data_ptr(), scales.data_ptr(), sequential, out.view(-1, ROW_BYTES), ROW_BYTES, BLOCK)
        return out


def _check_rows(self, param, loaded_weight):
    """Loader of the empty parameters: the checkpoint tensor is only checked, never read (it stays a lazy mapping)."""
    if loaded_weight.shape[0] != self._xtier_rows.rows:
        raise ValueError(f'xtier: Engram table of {loaded_weight.shape[0]} rows, the row store has '
                         f'{self._xtier_rows.rows}')


def _init(original, self, num_embeddings, dim, layer_id, column_bounds, *args, **kwargs):
    from sglang.srt.layers.engram import shard_range
    from sglang.srt.runtime_context import get_parallel

    if dim != ROW_BYTES:
        raise ValueError(f'xtier: Engram rows on NVMe hold {ROW_BYTES} FP8 values, not {dim}')
    torch.nn.Module.__init__(self)
    self.dim = dim
    self.tp_size = get_parallel().tp_size
    self.row_start, row_end = shard_range(num_embeddings, column_bounds, get_parallel().tp_rank, self.tp_size)
    self.rows = row_end - self.row_start
    self.host_table = None
    cache = int(float(os.environ.get('XTIER_ENGRAM_CACHE_GIB', '4')) * 2**30) // ENGRAM_LAYERS
    store = open_table(model_dir(), layer_id, cache, (self.row_start, row_end))
    if store.rows != num_embeddings:
        raise ValueError(f'xtier: Engram layer {layer_id}: config {num_embeddings} rows, checkpoint {store.rows}')
    store.preload_scales()
    self._xtier_rows = store
    self._xtier_lookup = EngramLookup(store)
    self._xtier_lookup.warm(int(os.environ.get('XTIER_ENGRAM_GRAPH_IDS', '16384')),
                            torch.device('cuda', torch.cuda.current_device()))
    self.weight = torch.nn.Parameter(torch.empty(0, dim, dtype=torch.float8_e4m3fn), requires_grad=False)
    self.scale = torch.nn.Parameter(torch.empty(0, dim // BLOCK, dtype=torch.float8_e8m0fnu), requires_grad=False)
    self.weight.weight_loader = functools.partial(_check_rows, self)
    self.scale.weight_loader = functools.partial(_check_rows, self)
    logger.info('xtier: Engram layer %d: %d rows [%d, %d) on NVMe (%s), row cache %.1f GiB', layer_id,
                num_embeddings, self.row_start, row_end, store.describe(), cache / 2**30)


def _owned_rows(self, indices):
    return self._xtier_lookup(indices)


def _make_candidate_indexer(original, *args, **kwargs):
    from sglang.srt.layers.deep_gemm_wrapper.configurer import DEEPGEMM_PAGED_SPARSE_MQA_LOGITS

    if (DEEPGEMM_PAGED_SPARSE_MQA_LOGITS or torch.cuda.get_device_capability()[0] != 12
            or kwargs.get('candidate_topk_blocks', 1) <= 0):
        return original(*args, **kwargs)
    from sglang.srt.layers.attention.dsv4 import v41_indexer
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import DenseBlocksBackend

    dense = DenseBlocksBackend(token_to_kv_pool=kwargs['token_to_kv_pool'], req_to_token=kwargs['req_to_token'],
                               candidate_topk_blocks=kwargs['candidate_topk_blocks'],
                               candidate_block_size=kwargs['candidate_block_size'],
                               use_deep_gemm_prefill=v41_indexer._use_deep_gemm_prefill())
    return dense, dense


@functools.lru_cache
def _tuned(N, K, block_n, block_k):
    device = torch.cuda.get_device_name().replace(' ', '_')
    path = FP8_CONFIG_DIR / f'N={N},K={K},device_name={device},dtype=fp8_w8a8,block_shape=[{block_n}, {block_k}].json'
    if not path.exists():
        return None
    configs = {int(m): c for m, c in json.loads(path.read_text()).items()}
    if any(c['BLOCK_SIZE_K'] < block_k or c['BLOCK_SIZE_K'] % block_k for c in configs.values()):
        return None   # the kernel steps the scales per BLOCK_SIZE_K, which must cover the block
    logger.info('xtier: W8A8 block FP8 config %s', path.name)
    return configs


def _fp8_configs(original, N, K, block_n, block_k):
    if not torch._dynamo.is_compiling():
        configs = _tuned(N, K, block_n, block_k)
        if configs is not None:
            return configs
    return original(N, K, block_n, block_k)


def install(registry, hook_type):
    registry.register(INDEXER, _make_candidate_indexer, hook_type.AROUND)
    registry.register(FP8_CONFIGS, _fp8_configs, hook_type.AROUND)
    if os.environ.get('XTIER_ENGRAM_ROWS', '') == 'nvme':
        registry.register(f'{ENGRAM}.__init__', _init, hook_type.AROUND)
        registry.register(f'{ENGRAM}._owned_rows', _owned_rows, hook_type.REPLACE)
