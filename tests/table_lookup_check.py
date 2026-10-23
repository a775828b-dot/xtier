"""Graph-capturable row-table lookups (xtier/tables/lookup.py) as the packs use them, against SGLang's own kernels on
rows read directly from the checkpoint files: DeepSeek V4.1 Engram (EngramLookup vs engram_gather) and the Qwen3.8 PLE
n-gram table (ple_rows.gather vs SGLang's pinned-table gather kernel), eager calls and a CUDA graph replayed with new
ids, the whole table and a TP shard range, bitwise.

  python tests/table_lookup_check.py --ds /data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017 \\
      --qwen /data/fast/nvme2/models/huggingface/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4
"""
import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from xtier.packs.deepseek_v4 import BLOCK, ROW_BYTES, EngramLookup, open_table as engram_table  # noqa: E402
from xtier.packs.qwen4_exp import ple_rows  # noqa: E402
from xtier.tables import tensor_locations  # noqa: E402
from xtier.tables.lookup import GraphLookup  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--ds')
P.add_argument('--ds-layer', type=int, default=14)
P.add_argument('--qwen')
A = P.parse_args()
g = torch.Generator().manual_seed(20261023)
_maps = {}


def host_rows(store, ids):
    """FP8 rows and scales of `ids` straight from the files (zero outside the owned range), as uint8 arrays."""
    W, S = store.row_bytes, store.scale_bytes
    starts = np.cumsum([0] + [s.rows for s in store.segments])
    w = np.zeros((len(ids), W), np.uint8)
    s = np.zeros((len(ids), max(S, 1)), np.uint8)
    for i, r in enumerate(ids.tolist()):
        if not store.lo <= r < store.hi:
            continue
        k = int(np.searchsorted(starts, r, side='right')) - 1
        seg, local = store.segments[k], r - int(starts[k])
        if str(seg.path) not in _maps:
            _maps[str(seg.path)] = np.memmap(seg.path, dtype=np.uint8, mode='r')
        data = _maps[str(seg.path)]
        w[i] = data[seg.weight_offset + local * W: seg.weight_offset + (local + 1) * W]
        if S:
            s[i] = data[seg.scale_offset + local * S: seg.scale_offset + (local + 1) * S]
    return w, s[:, :S]


def engram_reference(store, ids):
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather
    w, s = host_rows(store, ids.cpu().numpy().reshape(-1))
    dw, ds = torch.from_numpy(w).cuda(), torch.from_numpy(s).cuda()
    out = torch.empty(len(w), ROW_BYTES, dtype=torch.bfloat16, device='cuda')
    engram_gather(dw.data_ptr(), ds.data_ptr(), torch.arange(len(w), device='cuda'), out, ROW_BYTES, BLOCK)
    return out.view(*ids.shape, ROW_BYTES)


def ple_reference(store, ids, dim):
    """SGLang's pinned-table kernel on a pinned table of just these rows (ids remapped to 0..n-1)."""
    from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel
    w, _ = host_rows(store, ids.cpu().numpy().reshape(-1))
    table = torch.from_numpy(w).view(torch.float8_e4m3fn).pin_memory()
    n = len(w)
    out = torch.empty(n, dim, dtype=torch.bfloat16, device='cuda')
    _gather_ple_embedding_from_pinned_kernel[(n,)](table.data_ptr(), torch.arange(n, device='cuda'), out,
                                                   embedding_dim=dim, tp_vocab_start=0, tp_vocab_end=n, is_fp8=True,
                                                   BLOCK_D=1 << (dim - 1).bit_length())
    torch.cuda.synchronize()
    return out.view(*ids.shape, dim)


def run(name, store, lookup, reference, shapes, static_shape):
    results = []
    for shape in shapes:
        ids = torch.randint(0, store.rows, shape, generator=g).cuda()
        results.append(torch.equal(lookup(ids), reference(ids)))
    static = torch.randint(0, store.rows, static_shape, generator=g).cuda()
    out = lookup(static)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = lookup(static)
    for _ in range(3):
        static.copy_(torch.randint(0, store.rows, static.shape, generator=g))
        graph.replay()
        torch.cuda.synchronize()
        results.append(torch.equal(out, reference(static)))
    del graph
    ok = all(results)
    print(f'{name}: eager {results[:len(shapes)]}, graph replays {results[len(shapes):]} -> '
          f'{"bitwise" if ok else "DIFFERENT"}, {store.stats()}', flush=True)
    return not ok


bad = 0
if A.ds:
    rows = engram_table(A.ds, A.ds_layer, 0).rows
    for label, row_range in (('whole table', (0, rows)), ('TP rank 0 of 2', (0, rows // 2))):
        store = engram_table(A.ds, A.ds_layer, 64 << 20, row_range)
        store.preload_scales()
        lookup = EngramLookup(store)
        lookup.warm(256, 'cuda')
        # one token x 24 columns / a verify batch / a prefill
        bad += run(f'Engram layer {A.ds_layer}, {label}', store, lookup, lambda ids: engram_reference(store, ids),
                   ((1, 24), (12, 24), (600, 24)), (4, 24))
        store.close()
if A.qwen:
    names = next(iter(ple_rows.shard_groups(A.qwen).values()))
    found = tensor_locations(A.qwen, names)
    dim, rows = found[names[0]][1][1], sum(found[n][1][0] for n in names)
    for label, row_range in (('whole table', (0, rows)), ('TP rank 1 of 2', (rows // 2, rows))):
        store = ple_rows.open_table(A.qwen, '', torch.float8_e4m3fn, dim, rows, len(names), 64 << 20, row_range)[0]
        module = types.SimpleNamespace(
            _xtier_lookup=GraphLookup(store), embedding_dim=dim, weight=torch.empty(0, dim, dtype=torch.float8_e4m3fn),
            allocate_output=lambda shape, device: torch.empty(shape, dtype=torch.bfloat16, device=device))
        module._xtier_lookup.warm(256, 'cuda')
        bad += run(f'Qwen PLE n-grams, {label}', store, lambda ids: ple_rows.gather(module, ids),
                   lambda ids: ple_reference(store, ids, dim), ((1, 16), (8, 16), (600, 16)), (4, 16))
        store.close()
print('TABLE-LOOKUP-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
