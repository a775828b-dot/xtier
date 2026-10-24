"""Engine time per CPU call (XTIER_ROUTE_TRACE wall_ns) in a serving model against the probe, at equal CPU experts.

  python planner_trace_cmp.py SERVING_TRACE SERVING_LOG PROFILE FULL_LAYERS SLOTS PROBE_TRACE PROBE_JSON

Serving: hand layout with the adaptive residency off, so a partly cached layer holds its profile's top SLOTS experts;
a call's CPU experts are its distinct routes outside them. Probe: the plain-path calls (every route on the CPU): the
first len(cpu_only) x 1024 records (32 calls x (1 eager pass + 1 + 3 x 10 replays) per measured point).
"""
import json
import re
import struct
import sys
from collections import defaultdict

sys.path.insert(0, '/data/fast/workspaces/xtier')
from xtier.config import cpulist  # noqa: E402

HEAD = struct.Struct('<QIIIHBBhBBI')


def records(path):
    data = open(path, 'rb').read()
    pos = 0
    while pos + HEAD.size <= len(data):
        t, wall, gate, down, layer, m, k, rc, act, _, seq = HEAD.unpack_from(data, pos)
        pos += HEAD.size
        ids = struct.unpack_from(f'<{m * k}i', data, pos)
        pos += 4 * m * k
        yield layer, m, k, wall / 1e3, ids


strace, slog, prof, full_spec, slots, ptrace, pjson = sys.argv[1:8]
n_plain = len(json.load(open(pjson))['cpu_only']) * 1024
slots = int(slots)
order = [int(x) for x in re.findall(r'xtier: layer (\d+): \d+ experts', open(slog, errors='ignore').read())]
seen, bridge = set(), {}
for i, l in enumerate(order):
    if l not in seen:
        bridge[i] = l
        seen.add(l)
d = json.load(open(prof))
cached = {l: set(r[:slots]) for l, r in zip(d['layer_ids'], d['layers'])}
full = set(cpulist(full_spec))


def table(rows, title):
    by = defaultdict(list)
    for n, r, us in rows:
        by[n].append((r, us))
    print(title)
    for n in sorted(by):
        if len(by[n]) < 20:
            continue
        rs = [x[0] for x in by[n]]
        us = sorted(x[1] for x in by[n])
        print(f'  CPU experts {n:3d}: {len(us):6d} calls, routes {sum(rs) / len(rs):5.1f}, engine us median {us[len(us) // 2]:6.1f}'
              f' mean {sum(us) / len(us):6.1f}')
    return by


serve = []
for layer, m, k, wall, ids in records(strace):
    lid = bridge.get(layer)
    if lid is None or lid in full or lid not in cached:
        continue
    cpu = [e for e in ids if e >= 0 and e not in cached[lid]]
    if not cpu:
        continue
    serve.append((len(set(cpu)), len(cpu), wall))
probe = []
for i, (layer, m, k, wall, ids) in enumerate(records(ptrace)):
    if i >= n_plain:
        break
    routes = [e for e in ids if e >= 0]
    if routes:
        probe.append((len(set(routes)), len(routes), wall))
a = table(serve, f'serving ({len(serve)} CPU calls of partly cached layers)')
b = table(probe, f'probe ({len(probe)} plain-path calls)')
