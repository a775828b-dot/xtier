"""Acceleration pack for SGLang's ``deepseek_v4`` models (DeepSeek V4.1 Flash 1017).

- Engram rows on NVMe (``XTIER_ENGRAM_ROWS=nvme``; docs/design-tables-nvme.md): an ``EngramEmbedding`` keeps no table
  in device or host memory. Its rows stay in the checkpoint file and come through a row table (xtier.tables: O_DIRECT
  reads, a row cache of ``XTIER_ENGRAM_CACHE_GIB`` over all Engram layers, the owned rows' scales preloaded); the
  owned-row gather ``_owned_rows`` becomes the graph-capturable lookup plus SGLang's ``engram_gather``. Hashing,
  gating, the projections and the TP all-reduce stay SGLang's. The checkpoint directory is the served model's
  (``XTIER_ENGRAM_MODEL`` overrides); staging for ``XTIER_ENGRAM_GRAPH_IDS`` ids (default 16384, 24 per token) is
  allocated before any graph capture.
- Engram rows prefetched (``XTIER_ENGRAM_PREFETCH``, default on; from the 1017 pipeline's DS41_ENGRAM_PREFETCH): every
  forward hashes all tokens up front, so as soon as ``EngramHasher.forward`` returns, each Engram layer's lookup (ids
  to the host, the row table's host function, rows back, dequantize) starts on a side stream of its own and overlaps
  the layers before it (layer 0 for Engram layer 1, layers 2-13 for layer 14); the layer then only waits for its
  event. The rows are the same; a graph captures the fork and the join. Otherwise the GPU waits for the host function
  inside the layer. RTX PRO 5000, DSpark: decode step -2.4 ms (8K and 64K context, with the FP8 draft head below);
  unlike the 1017 pipeline, prefill prefetches too (a 22528-token chunk looks up 540K rows per layer): 64K cold
  prefill +16%.
- DSpark draft LM head in FP8 (``XTIER_DRAFT_LMHEAD_FP8``, default on; DS41_DRAFT_LMHEAD_FP8): the draft projects
  through the target's BF16 LM head (1.32 GB) every step. A copy in FP8 E4M3 with one scale per vocabulary row,
  multiplied by ``torch._scaled_mm`` with per-token activation scales, takes 0.62 instead of 1.11 ms on the RTX PRO
  5000 (+0.66 GB). It is made when the draft attaches the target's head, before any graph capture. Draft proposals
  only: the target verifies every token with its BF16 head, so the output is the target's.
- SM120 candidate indexer: SGLang sends every SM100-or-newer GPU to the sparse candidate table, whose DeepGEMM paged
  sparse MQA logits exist on SM100 only; on SM120 the dense-block candidates (SGLang's path below SM100, DeepGEMM's
  SM120 dense FP8 / FP4 MQA logits) serve prefill and decode.
- Tuned W8A8 block-FP8 GEMM configs (fp8_configs/, from the 1017 pipeline's bench/ds41_fp8_tune.py): SGLang has none
  for DeepSeek V4.1's 32 x 32 block projections on SM120 and runs one default tile for every shape, 1.5-4.6x slower
  at decode sizes (on the RTX PRO 5000 63 ms of a 68 ms decode step). Files of this directory win over SGLang's own.
- MXFP8 dense GEMMs at decode sizes on b12x: with --fp8-gemm-backend flashinfer_cutlass SGLang serves the
  32 x 32 block FP8 projections as MXFP8 on FlashInfer CUTLASS (on auto it keeps the Triton block kernel, which on
  SM120 has no split-K); calls of XTIER_MXFP8_B12X_ROWS rows (default 1, 4, 6: decode and DSpark verify widths,
  as the 1017 pipeline's verify6 fast path) take FlashInfer's b12x small-M kernels instead.
- Attention wo_a at decode sizes on a Triton grouped einsum, the target's weights in the checkpoint's FP8
  (``XTIER_DS_WO_A``, default on; the 1017 pipeline's LKQMOE_DS41_WO_A_TRITON / _FP8): SGLang dequantizes the 32 x 32
  block FP8 wo_a to BF16 and cuBLAS runs the einsum on an SM80 WMMA kernel, 71-75 us per layer at the DSpark
  verify's 6 tokens on the RTX PRO 5000, the FP8 kernel 30 us; see wo_a.py.
- Indexer decode scored from the request table (``XTIER_DS_PAGED_INDEXER``, default on): SGLang takes its paged
  FP4 index logits and fused top-k v2 on SM90 only; on SM120 it gathers every row's slots over the graph capacity
  and selects with torch.topk. Nothing on the paged path is SM90-only; see paged_indexer.py. RTX PRO 5000, DSpark,
  16K context: the indexer's kernels 1.08 -> 0.33 ms per decode step, the main stream's work 15.4 -> 13.7 ms;
  agent_bench long responses 33.0 -> 32.5 ms per step.
- Prompts tokenized as DeepSeek's deepseek-recipe does (``XTIER_DS_RECIPE_TOKENS``, default on; the 1017 pipeline's
  DS41_PLAIN_CONTENT_TOKENS): added-token spellings in message content stay plain text, only the template's control
  tokens are special, and the prompt is encoded in regions instead of one string (a 568K-token prompt 1.3 -> 0.18 s,
  every turn). Hooks of the serving process, not the schedulers; see recipe_tokens.py.
- Decoder replay tail without graphs: with --enable-decoder-swa-bounded-replay SGLang runs the late layers of a
  prefill from row-bucket CUDA graphs captured while serving (up to SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS = 2048
  rows). Their MoE calls of hundreds to thousands of rows cannot go through XTier's capturable decode path (mailbox
  slots of 8 tokens), so the tail runs eagerly on XTier's prefill paths; the replay itself is unchanged.
"""

import functools
import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
ENGRAM = 'sglang.srt.layers.engram.EngramEmbedding'
ENGRAM_LAYER = 'sglang.srt.layers.engram.Engram'
HASHER = 'sglang.srt.layers.engram.EngramHasher'
DSPARK_DRAFT = 'sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark'
DRAFT_PROJECT = 'sglang.srt.models.dspark.project_through_lm_head'
INDEXER = 'sglang.srt.layers.attention.dsv4.v41_indexer.make_candidate_indexer'
FP8_CONFIGS = 'sglang.kernels.ops.gemm.fp8_kernel.get_w8a8_block_fp8_configs'
REPLAY_TAIL = 'sglang.srt.models.deepseek_v4_replay_tail.DecoderReplayTail.maybe_create'
MXFP8_LINEAR = 'sglang.srt.layers.quantization.fp8_utils.flashinfer_mxfp8_blockscaled_linear'
FP8_CONFIG_DIR = Path(__file__).with_name('fp8_configs')
ENGRAM_LAYERS = 2   # DeepSeek V4.1 Flash has two Engram layers (1 and 14); the cache budget is split over them
ROW_BYTES, SCALE_BYTES, BLOCK = 256, 8, 32   # an FP8 E4M3 row of 256 with one E8M0 scale per 32
FP8_MAX = 448.0
_prefetching = {}   # Engram layer's hash index -> (its EngramEmbedding on the row table, side stream); one target model


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

    def fill(self, indices, out):
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        if indices.numel():
            rows, scales, sequential = self.lookup.fetch(indices)
            engram_gather(rows.data_ptr(), scales.data_ptr(), sequential, out.view(-1, ROW_BYTES), ROW_BYTES, BLOCK)
        return out

    def __call__(self, indices):
        return self.fill(indices, torch.empty(*indices.shape, ROW_BYTES, dtype=torch.bfloat16, device=indices.device))


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


def _view_key(t):
    return t.data_ptr(), tuple(t.shape), tuple(t.stride())


def _engram_layer_init(result, self, *args, **kwargs):
    """After ``Engram.__init__``: a layer on the row table joins the prefetch, with a side stream of its own."""
    if getattr(self.embed, '_xtier_lookup', None) is not None:
        _prefetching[self.layer_hash_index] = (self.embed, torch.cuda.Stream())
    return result


def _prefetch(hash_ids, self, input_ids, *args, **kwargs):
    """After ``EngramHasher.forward`` (hash ids [T, Engram layers, hash columns]): every Engram layer's lookup starts
    on its side stream. The ids are made contiguous and the output allocated on the current stream, and both stay
    referenced until the layer has waited for the lookup."""
    if not _prefetching or hash_ids.dim() != 3 or hash_ids.shape[0] == 0 or not hash_ids.is_cuda:
        return hash_ids
    main = torch.cuda.current_stream()
    for index, (embedding, side) in _prefetching.items():
        if index >= hash_ids.shape[1]:
            continue
        view = hash_ids[:, index]
        ids = view.contiguous()
        out = torch.empty(*view.shape, ROW_BYTES, dtype=torch.bfloat16, device=view.device)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            embedding._xtier_lookup.fill(ids, out)
            done = torch.cuda.Event()
            done.record(side)
        embedding._xtier_prefetched = (_view_key(view), out, done, ids)
    return hash_ids


def _owned_rows(self, indices):
    prefetched = getattr(self, '_xtier_prefetched', None)
    if prefetched is not None:
        self._xtier_prefetched = None
        key, out, done, _ = prefetched
        torch.cuda.current_stream().wait_event(done)   # also orders an inline lookup after it (shared staging)
        if key == _view_key(indices):
            return out
    return self._xtier_lookup(indices)


def _quantize_lm_head(lm_head):
    """Row-scaled FP8 E4M3 copy of a BF16 LM head for the draft: (weight [vocab, hidden], scales [1, vocab])."""
    weight = lm_head.weight.data
    n, k = weight.shape
    qweight = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty((n, 1), dtype=torch.float32, device=weight.device)
    for start in range(0, n, 8192):   # bounded FP32 transients
        rows = weight[start:start + 8192].float()
        sc = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
        qweight[start:start + 8192] = (rows / sc).to(torch.float8_e4m3fn)
        scale[start:start + 8192] = sc
    lm_head._xtier_draft_fp8 = (qweight, scale.t().contiguous())
    logger.info('xtier: DSpark draft LM head as a row-scaled FP8 copy %s (%.2f GiB)', tuple(qweight.shape),
                qweight.numel() / 2**30)


def _attach_shared(result, self, *args, **kwargs):
    """After the DSpark draft attached the target's LM head (before its graphs are captured)."""
    head = getattr(self, 'lm_head', None)
    if (head is not None and getattr(head, 'weight', None) is not None and head.weight.dtype == torch.bfloat16
            and head.weight.dim() == 2 and getattr(head, '_xtier_draft_fp8', None) is None):
        _quantize_lm_head(head)
    return result


def _draft_project(original, hidden, lm_head):
    packed = getattr(lm_head, '_xtier_draft_fp8', None)
    if packed is None:
        return original(hidden, lm_head)
    qweight, scale = packed
    x = hidden.reshape(-1, hidden.shape[-1]).float()
    sx = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    out = torch._scaled_mm((x / sx).to(torch.float8_e4m3fn), qweight.t(), scale_a=sx, scale_b=scale,
                           out_dtype=torch.bfloat16)
    return out.view(*hidden.shape[:-1], qweight.shape[0])


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


def _replay_tail(original, cls, model, *args, **kwargs):
    from xtier import config
    if config.enabled():
        logger.info('xtier: DeepSeek V4.1 decoder replay tail runs eagerly (no row-bucket graphs: their MoE calls of up '
                    'to %s rows cannot be captured through XTier)',
                    os.environ.get('SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS', '2048'))
        return None
    return original(cls, model, *args, **kwargs)


@functools.lru_cache
def _b12x_rows():
    if torch.cuda.get_device_capability()[0] != 12:
        return frozenset()
    return frozenset(int(r) for r in os.environ.get('XTIER_MXFP8_B12X_ROWS', '1,4,6').split(',') if r.strip())


def _mxfp8_linear(original, input, *args, **kwargs):
    if kwargs.get('backend', 'cutlass') == 'cutlass' and input.numel() // input.shape[-1] in _b12x_rows():
        kwargs['backend'] = 'b12x'
    return original(input, *args, **kwargs)


def install(registry, hook_type):
    from xtier.packs.deepseek_v4 import paged_indexer, wo_a

    registry.register(INDEXER, _make_candidate_indexer, hook_type.AROUND)
    registry.register(REPLAY_TAIL, _replay_tail, hook_type.AROUND)
    wo_a.install(registry, hook_type)
    paged_indexer.install(registry, hook_type)
    if os.environ.get('XTIER_FP8_CONFIGS', '1') != '0':   # speed only: 0 keeps SGLang's own configs
        registry.register(FP8_CONFIGS, _fp8_configs, hook_type.AROUND)
    registry.register(MXFP8_LINEAR, _mxfp8_linear, hook_type.AROUND)
    if os.environ.get('XTIER_ENGRAM_ROWS', '') == 'nvme':
        registry.register(f'{ENGRAM}.__init__', _init, hook_type.AROUND)
        registry.register(f'{ENGRAM}._owned_rows', _owned_rows, hook_type.REPLACE)
        if os.environ.get('XTIER_ENGRAM_PREFETCH', '1') != '0':
            registry.register(f'{ENGRAM_LAYER}.__init__', _engram_layer_init, hook_type.AFTER)
            registry.register(f'{HASHER}.forward', _prefetch, hook_type.AFTER)
    if os.environ.get('XTIER_DRAFT_LMHEAD_FP8', '1') != '0':
        registry.register(f'{DSPARK_DRAFT}.attach_shared_modules', _attach_shared, hook_type.AFTER)
        registry.register(DRAFT_PROJECT, _draft_project, hook_type.AROUND)


def install_serving(registry, hook_type):
    """Hooks of the process that serves the API: prompts tokenized as deepseek-recipe does (recipe_tokens)."""
    from xtier.packs.deepseek_v4 import recipe_tokens

    return recipe_tokens.install_serving(registry, hook_type)
