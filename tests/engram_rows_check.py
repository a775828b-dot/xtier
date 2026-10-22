"""Engram row store (xtier/packs/deepseek_v4/rows.py, csrc/engram/rows.cpp) against the checkpoint file read directly:
random ids with repeats (cache hits and misses), the whole table and a TP shard range (rows outside it must be zero),
byte for byte. CPU only.

  python tests/engram_rows_check.py --model /data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017 --layers 1,14
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from xtier.packs.deepseek_v4.rows import ROW_BYTES, SCALE_BYTES, RowStore, table_location  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--model', required=True)
P.add_argument('--layers', default='1,14')
P.add_argument('--count', type=int, default=20000)
P.add_argument('--cache-mib', type=int, default=64)
A = P.parse_args()
g = torch.Generator().manual_seed(20261023)
bad = 0
for layer in map(int, A.layers.split(',')):
    path, rows, woff, soff = table_location(A.model, layer)
    data = np.memmap(path, dtype=np.uint8, mode='r')
    for name, row_range in (('whole table', None), ('TP rank 1 of 2', (rows // 2, rows))):
        store = RowStore(path, rows, woff, soff, A.cache_mib << 20, row_range)
        store.preload_scales()
        hot = torch.randint(0, rows, (A.count // 10,), generator=g)
        ids = torch.cat([torch.randint(0, rows, (A.count,), generator=g), hot, hot])   # repeats: cache hits
        for _ in range(2):   # the second pass reads through the warm cache
            w, s = store.lookup(ids)
        idx = ids.numpy()
        want_w = np.stack([data[woff + i * ROW_BYTES: woff + (i + 1) * ROW_BYTES] for i in idx])
        want_s = np.stack([data[soff + i * SCALE_BYTES: soff + (i + 1) * SCALE_BYTES] for i in idx])
        if row_range is not None:
            outside = (idx < row_range[0]) | (idx >= row_range[1])
            want_w[outside] = 0
            want_s[outside] = 0
        same = np.array_equal(w.numpy(), want_w) and np.array_equal(s.numpy(), want_s)
        bad += not same
        print(f'layer {layer} ({rows} rows), {name}: {len(idx)} lookups x 2 {"byte for byte" if same else "DIFFERENT"}, '
              f'{store.stats()}', flush=True)
        store.close()
print('ENGRAM-ROWS-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
