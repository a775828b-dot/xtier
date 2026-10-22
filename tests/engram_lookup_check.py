"""Graph-capturable Engram lookup from NVMe rows (xtier/packs/deepseek_v4/engram.py) against SGLang's engram_gather on
rows read directly from the checkpoint file: eager calls and a CUDA graph replayed with new ids, whole table and a TP
shard range, bitwise.

  python tests/engram_lookup_check.py --model /data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017 --layer 14
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sglang.kernels.ops.embeddings.engram_gather import engram_gather  # noqa: E402

from xtier.packs.deepseek_v4.engram import NvmeLookup  # noqa: E402
from xtier.packs.deepseek_v4.rows import ROW_BYTES, SCALE_BYTES, RowStore, table_location  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--model', required=True)
P.add_argument('--layer', type=int, default=14)
A = P.parse_args()
path, rows, woff, soff = table_location(A.model, A.layer)
data = np.memmap(path, dtype=np.uint8, mode='r')
g = torch.Generator().manual_seed(20261023)


def reference(ids, row_range):
    idx = ids.cpu().numpy().reshape(-1)
    w = np.stack([data[woff + i * ROW_BYTES: woff + (i + 1) * ROW_BYTES] for i in idx])
    s = np.stack([data[soff + i * SCALE_BYTES: soff + (i + 1) * SCALE_BYTES] for i in idx])
    lo, hi = row_range
    outside = (idx < lo) | (idx >= hi)
    w[outside] = 0
    s[outside] = 0
    dw, ds = torch.from_numpy(w).cuda(), torch.from_numpy(s).cuda()
    out = torch.empty(len(idx), ROW_BYTES, dtype=torch.bfloat16, device='cuda')
    engram_gather(dw.data_ptr(), ds.data_ptr(), torch.arange(len(idx), device='cuda'), out, ROW_BYTES, 32)
    return out.view(*ids.shape, ROW_BYTES)


bad = 0
for name, row_range in (('whole table', (0, rows)), ('TP rank 0 of 2', (0, rows // 2))):
    store = RowStore(path, rows, woff, soff, 64 << 20, row_range)
    store.preload_scales()
    lookup = NvmeLookup(store)
    results = []
    for shape in ((7,), (3, 25), (1, 600)):   # one token / a verify batch / a prefill of 600 x 1 head
        ids = torch.randint(0, rows, shape, generator=g).cuda()
        results.append(torch.equal(lookup(ids), reference(ids, row_range)))
    static = torch.randint(0, rows, (4, 6), generator=g).cuda()
    lookup.warm(static.numel(), 'cuda')
    out = lookup(static)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = lookup(static)
    for _ in range(3):
        static.copy_(torch.randint(0, rows, static.shape, generator=g))
        graph.replay()
        torch.cuda.synchronize()
        results.append(torch.equal(out, reference(static, row_range)))
    del graph
    ok = all(results)
    bad += not ok
    print(f'layer {A.layer}, {name}: eager {results[:3]}, graph replays {results[3:]} -> {"bitwise" if ok else "DIFFERENT"}, '
          f'{store.stats()}', flush=True)
    store.close()
print('ENGRAM-LOOKUP-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
