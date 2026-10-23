"""PLE n-gram table on NVMe (``XTIER_PLE_ROWS=nvme``; docs/design-tables-nvme.md).

SGLang's offloaded PLE embedding (``Qwen4ExpPinnedHostEmbedding``) holds the n-gram table in pinned host memory, or in
a file-backed mapping the GPU reads through host page tables. Here it keeps no table: the rows stay in the checkpoint's
shard tensors (``...ngram_embedding.shard_N.weight``, global rows in shard order) and come through a row table
(xtier.tables) with a row cache of ``XTIER_PLE_CACHE_GIB`` (default 4); ``gather`` becomes the graph-capturable lookup
plus the conversion to BF16 that SGLang's pinned-table kernel does (rows outside the TP shard are zero; the per-tensor
``weight_scale`` stays the caller's). The shard tensors are dropped from the weight iterator, so they are never read.
Staging for ``XTIER_PLE_GRAPH_IDS`` ids (default 16384, 16 per token on Qwen3.8) exists before any graph capture.
"""

import inspect
import json
import logging
import os
import re
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
SHARD = re.compile(r'^(.*)\.ngram_embedding\.shard_(\d+)\.weight$')
DTYPES = {torch.float8_e4m3fn: 'F8_E4M3', torch.bfloat16: 'BF16'}
_building = []   # (module prefix, checkpoint shard count) of the Qwen4ExpNGramEmbedding under construction


def model_dir():
    path = os.environ.get('XTIER_PLE_MODEL')
    if not path:
        from sglang.srt.runtime_context import get_model
        path = get_model().model_path
    return path


def shard_groups(model):
    """{checkpoint module prefix: shard tensor names in shard order}"""
    index = json.loads((Path(model) / 'model.safetensors.index.json').read_text())['weight_map']
    groups = {}
    for name in index:
        m = SHARD.match(name)
        if m:
            groups.setdefault(m.group(1), []).append((int(m.group(2)), name))
    return {k: [n for _, n in sorted(v)] for k, v in groups.items()}


def find_shards(groups, prefix):
    """Shards of the SGLang module `prefix` (model.layers.1.ple.ple_embedding <-> the checkpoint's
    model.language_model.layers.1.ple.ple_embedding)."""
    if 'layers.' in prefix:
        tail = 'layers.' + prefix.split('layers.', 1)[1]
        hits = [k for k in groups if k.endswith(tail)]
        if len(hits) == 1:
            return groups[hits[0]]
    if len(groups) == 1:
        return next(iter(groups.values()))
    raise ValueError(f'xtier: no unique PLE shard group for {prefix!r} among {sorted(groups)}')


def open_table(model, prefix, dtype, dim, vocab, parts, cache_bytes, row_range):
    """Row table of one PLE n-gram embedding: the checkpoint shards in order, `vocab` rows split into equal shards of
    ceil(vocab / parts) rows, as SGLang's loader places them."""
    from xtier.tables import RowStore, Segment, tensor_locations

    names = find_shards(shard_groups(model), prefix)
    found = tensor_locations(model, names)
    segments = []
    for name in names:
        path, shape, st_dtype, offset = found[name]
        if len(shape) != 2 or shape[1] != dim or st_dtype != DTYPES[dtype]:
            raise ValueError(f'xtier: PLE shard {name}: {shape} {st_dtype}, expected [*, {dim}] {DTYPES[dtype]}')
        segments.append(Segment(path, shape[0], offset))
    parts = parts or len(names)
    shard = -(-vocab // parts)
    if len(names) != parts or any(s.rows != shard for s in segments[:-1]) or segments[-1].rows > shard:
        raise ValueError(f'xtier: PLE shards of {[s.rows for s in segments[:3]]}... rows, expected {parts} of {shard}')
    return RowStore(segments, dim * torch.empty(0, dtype=dtype).element_size(), 0, cache_bytes, row_range), names


def _no_load(param, loaded_weight, *args, **kwargs):
    raise RuntimeError('xtier: the PLE n-gram table stays on NVMe; its checkpoint shards must not be loaded')


def ngram_init(original, self, *args, **kwargs):
    bound = inspect.signature(original).bind(self, *args, **kwargs)
    config = bound.arguments.get('config')
    _building.append((bound.arguments.get('prefix', ''), int(getattr(config, 'split_ngram_parts', 0) or 0)))
    try:
        return original(self, *args, **kwargs)
    finally:
        _building.pop()


def host_embedding_init(original, self, embedding, *args, **kwargs):
    from sglang.srt.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding

    from xtier.tables.lookup import GraphLookup

    torch.nn.Module.__init__(self)
    source = embedding.weight
    if not isinstance(embedding.quant_method, UnquantizedEmbeddingMethod) or source.dtype not in DTYPES \
            or embedding.num_added_embeddings:
        raise NotImplementedError(f'xtier: PLE rows on NVMe need an unquantized BF16 / FP8 table without added rows, '
                                  f'got {type(embedding.quant_method).__name__} {source.dtype}')
    for name in Qwen4ExpPinnedHostEmbedding._COPIED_ATTRIBUTES:
        setattr(self, name, getattr(embedding, name))
    self.quant_method = None
    prefix, parts = _building[-1] if _building else ('', 0)
    lo, hi = self.shard_indices.org_vocab_start_index, self.shard_indices.org_vocab_end_index
    cache = int(float(os.environ.get('XTIER_PLE_CACHE_GIB', '4')) * 2**30)
    store, names = open_table(model_dir(), prefix, source.dtype, self.embedding_dim, self.org_vocab_size, parts,
                              cache, (lo, hi))
    self._xtier_rows, self._xtier_lookup = store, GraphLookup(store)
    self._xtier_lookup.warm(int(os.environ.get('XTIER_PLE_GRAPH_IDS', '16384')),
                            torch.device('cuda', torch.cuda.current_device()))
    self._file_prefetcher = self._file_rss_trimmer = None
    weight = torch.nn.Parameter(torch.empty((0, self.embedding_dim), dtype=source.dtype, device='cpu'),
                                requires_grad=False)
    for name, value in vars(source).items():
        setattr(weight, name, value)
    weight.weight_loader = _no_load
    self.register_parameter('weight', weight)
    self.register_buffer('weight_scale', embedding.weight_scale, persistent=True)
    del embedding.weight
    logger.info('xtier: PLE n-gram table %s: %d rows x %d %s [%d, %d) on NVMe (%d shards, %s), row cache %.1f GiB',
                prefix or '?', store.rows, self.embedding_dim, source.dtype, lo, hi, len(names), store.describe(),
                cache / 2**30)


def gather(self, input_ids, out=None):
    expected = (*input_ids.shape, self.embedding_dim)
    if out is None:
        output = self.allocate_output(expected, input_ids.device)
    elif tuple(out.shape) != expected or out.dtype != torch.bfloat16 or out.device != input_ids.device:
        raise ValueError(f'invalid PLE output {tuple(out.shape)} {out.dtype} {out.device}, expected {expected} '
                         f'bfloat16 on {input_ids.device}')
    else:
        output = out
    ids = input_ids.reshape(-1)
    if ids.numel():
        rows, _, _ = self._xtier_lookup.fetch(ids.long())
        output.view(-1, self.embedding_dim).copy_(rows.view(self.weight.dtype))
    return output


def load_weights(original, self, weights):
    owned = [name for name, m in self.named_modules() if getattr(m, '_xtier_lookup', None) is not None]
    if not owned:
        return original(self, weights)
    dropped = []

    def kept():
        for name, tensor in weights:
            if SHARD.match(name):
                dropped.append(name)
                continue
            yield name, tensor

    loaded = original(self, kept())
    logger.info('xtier: %d PLE n-gram shards stay on NVMe (not loaded) for %s', len(dropped), owned)
    if isinstance(loaded, set):
        loaded.update(f'{name}.weight' for name in owned)
    return loaded
