"""Row tables (xtier/tables, csrc/tables/rows.cpp) against the checkpoint files read directly: random ids with repeats
plus the table ends and every segment boundary, two lookups each (the second mostly from the row cache), the whole
table and a TP shard range (rows outside it must be zero), scales read or preloaded, byte for byte. CPU only.

  python tests/tables_check.py --ds /data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017 \\
      --qwen /data/fast/nvme2/models/huggingface/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from xtier.packs.deepseek_v4 import open_table as engram_table  # noqa: E402
from xtier.packs.qwen4_exp.ple_rows import open_table as ple_table, shard_groups  # noqa: E402
from xtier.tables import tensor_locations  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--ds', help='DeepSeek V4.1 checkpoint (Engram tables)')
P.add_argument('--ds-layers', default='1,14')
P.add_argument('--qwen', help='Qwen3.8 checkpoint (PLE n-gram table)')
P.add_argument('--count', type=int, default=20000)
P.add_argument('--cache-mib', type=int, default=64)
A = P.parse_args()
g = torch.Generator().manual_seed(20261023)
_maps = {}


def mapped(path):
    if path not in _maps:
        _maps[path] = np.memmap(path, dtype=np.uint8, mode='r')
    return _maps[path]


def reference(store, ids):
    """Rows and scales of `ids` read from the files with numpy, zero outside the owned range."""
    W, S = store.row_bytes, store.scale_bytes
    starts = np.cumsum([0] + [s.rows for s in store.segments])
    w = np.zeros((len(ids), W), np.uint8)
    s = np.zeros((len(ids), max(S, 1)), np.uint8)
    for i, r in enumerate(ids.tolist()):
        if not store.lo <= r < store.hi:
            continue
        k = int(np.searchsorted(starts, r, side='right')) - 1
        seg, local = store.segments[k], r - int(starts[k])
        data = mapped(str(seg.path))
        w[i] = data[seg.weight_offset + local * W: seg.weight_offset + (local + 1) * W]
        if S:
            s[i] = data[seg.scale_offset + local * S: seg.scale_offset + (local + 1) * S]
    return w, s[:, :S]


def lookup_ids(store):
    rows = store.rows
    bounds = np.cumsum([s.rows for s in store.segments])[:-1].tolist()
    edges = sorted({0, rows - 1, *bounds, *[b - 1 for b in bounds]})
    base = torch.randint(0, rows, (A.count,), generator=g)
    ids = torch.cat([base, base[: A.count // 4], torch.tensor(edges, dtype=torch.int64)])
    return ids[torch.randperm(len(ids), generator=g)]


def check(name, make_store, rows, scales):
    bad = 0
    for label, row_range in (('whole table', None), ('TP rank 1 of 2', (rows // 2, rows))):
        for preload in ((False, True) if scales else (False,)):
            store = make_store(row_range)
            if preload:
                store.preload_scales()
            ids = lookup_ids(store)
            want = reference(store, ids.numpy())
            results = []
            for _ in range(2):
                w, s = store.lookup(ids)
                results.append(np.array_equal(w.numpy(), want[0]) and np.array_equal(s.numpy(), want[1]))
            ok = all(results)
            bad += not ok
            mode = (', scales preloaded' if preload else ', scales read') if scales else ''
            print(f'{name}, {label}{mode}: {results} -> {"byte-exact" if ok else "DIFFERENT"}, {store.stats()}, '
                  f'{store.io_stats()}', flush=True)
            store.close()
    return bad


bad = 0
if A.ds:
    for layer in map(int, A.ds_layers.split(',')):
        rows = engram_table(A.ds, layer, 0).rows
        bad += check(f'Engram layer {layer} ({rows} rows)',
                     lambda rr, layer=layer: engram_table(A.ds, layer, A.cache_mib << 20, rr), rows, True)
if A.qwen:
    names = next(iter(shard_groups(A.qwen).values()))
    first = tensor_locations(A.qwen, names[:1])[names[0]]
    dim, rows = first[1][1], sum(tensor_locations(A.qwen, names)[n][1][0] for n in names)
    bad += check(f'Qwen PLE n-grams ({len(names)} shards, {rows} rows x {dim})',
                 lambda rr: ple_table(A.qwen, '', torch.float8_e4m3fn, dim, rows, len(names), A.cache_mib << 20,
                                      rr)[0], rows, False)
print('TABLES-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
