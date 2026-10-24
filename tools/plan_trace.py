"""Routing profile with counts from a route trace (XTIER_ROUTE_TRACE=<file> on a server run), for the planner.

  python tools/plan_trace.py TRACE --log SERVER_LOG --experts E [--model NAME] [--skip SECONDS] --out PROFILE.json

The trace has one record per CPU decode call of every layer that uses the mailbox (partly cached or uncached layers;
fully cached layers never reach the CPU): a 32-byte header (csrc/mailbox/mailbox.cpp TraceHead: realtime ns, wall /
gate / down ns, layer = mailbox creation order, m, k, ...) and the call's m x k routes (with the expert cache: the
original routes, cached ones included). One call of a layer is one decode window. The mailbox creation order is the
order of the server log's "xtier: layer N: ..." lines; a layer id seen twice (a speculative draft's layer numbered
like a model layer, e.g. Qwen's MTP layer 0) keeps its first (model) occurrence, the draft's records are dropped.
Output: the LKQstrata profile format (experts most used first per layer) plus "counts" (windows per expert),
"routes" (token-expert pairs per expert) and "windows", as tools/plan_auto.py and XTIER_CACHE_PROFILE read it.
"""
import argparse
import json
import re
import struct
import sys
from collections import Counter

P = argparse.ArgumentParser()
P.add_argument('trace')
P.add_argument('--log', required=True, help='server log of the traced run (layer creation order)')
P.add_argument('--experts', type=int, required=True)
P.add_argument('--model', default='')
P.add_argument('--skip', type=float, default=0.0, help='drop the first SECONDS of records (warm-up)')
P.add_argument('--out', required=True)
A = P.parse_args()

order = []
for line in open(A.log, errors='ignore'):
    m = re.search(r'xtier: layer (\d+): \d+ experts', line)
    if m:
        order.append(int(m.group(1)))
model_layers = {}
for i, l in enumerate(order):
    if l not in model_layers.values():
        model_layers[i] = l
print(f'{len(order)} XTier layers in the log, {len(model_layers)} model layers', file=sys.stderr)

HEAD = struct.Struct('<QIIIHBBhBBI')
counts, route_counts, windows, t_first = {}, {}, Counter(), None
calls = dropped = 0
with open(A.trace, 'rb') as f:
    data = f.read()
pos = 0
while pos + HEAD.size <= len(data):
    t_ns, wall, gate, down, layer, m, k, rc, active, _, seq = HEAD.unpack_from(data, pos)
    pos += HEAD.size
    n = m * k
    ids = struct.unpack_from(f'<{n}i', data, pos)
    pos += 4 * n
    calls += 1
    if t_first is None:
        t_first = t_ns
    if (t_ns - t_first) / 1e9 < A.skip or layer not in model_layers:
        dropped += 1
        continue
    lid = model_layers[layer]
    c = counts.setdefault(lid, Counter())
    rc = route_counts.setdefault(lid, Counter())
    for e in ids:
        if 0 <= e < A.experts:
            rc[e] += 1
    for e in set(ids):
        if 0 <= e < A.experts:
            c[e] += 1
    windows[lid] += 1
if pos != len(data):
    print(f'warning: {len(data) - pos} trailing bytes (trace still being written?)', file=sys.stderr)
layers = sorted(counts)
W = max(windows.values())
rows = [[c[e] for e in range(A.experts)] for c in (counts[l] for l in layers)]
prof = dict(model=A.model, experts=A.experts, windows=W, sets='trace', layer_ids=layers,
            layers=[sorted(range(A.experts), key=lambda e: -r[e]) for r in rows], counts=rows,
            routes=[[route_counts[l][e] for e in range(A.experts)] for l in layers],
            note=f'from XTIER_ROUTE_TRACE: {calls} calls, {dropped} dropped (warm-up / draft), windows per layer '
                 f'{min(windows.values())}..{W}')
json.dump(prof, open(A.out, 'w'))
dist = [sum(r) / windows[l] for l, r in zip(layers, rows)]
print(f'{len(layers)} layers, {W} windows, distinct experts per window {min(dist):.1f}..{max(dist):.1f} '
      f'(mean {sum(dist) / len(dist):.1f}); written {A.out}')
