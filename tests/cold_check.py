"""Cold experts on NVMe, one real layer (DeepSeek V4.1 Flash 1017 layer 20, MXFP4) on the CPU engine: a plan with
RAM slots for (1 - X) of the 384 experts next to the full plan, the others in the store on the Optane drives, the
layer's decode routes replayed from a production route trace (XTIER_ROUTE_TRACE of agent_bench). Per call: the CPU's
experts (the 50 most used of the layer are the GPU's, as the hot-expert pool serves them) that are not in RAM are read
into the slots of the experts with the lowest decayed use counts (half-life 50 calls), then both plans run the call
and must agree bit for bit. Reported: the reads per call and their time, the drives' bandwidth, and the bytes
/proc/diskstats counted on the drives against the bytes requested (O_DIRECT: the page cache must not grow).

  python tests/cold_check.py --lib-dir build/engine --trace TRACE --x 0.1 0.2 0.3 --store DIR1 DIR2
"""
import argparse
import ctypes as C
import json
import os
import struct
import time
from pathlib import Path

import numpy as np

P = argparse.ArgumentParser()
P.add_argument('--lib-dir', required=True)
P.add_argument('--trace', required=True)
P.add_argument('--x', type=float, nargs='+', default=[0.1, 0.2, 0.3])
P.add_argument('--store', nargs='+', required=True, help='one directory per drive')
P.add_argument('--threads', type=int, default=112)
P.add_argument('--io-threads', type=int, default=8)
P.add_argument('--gpu-slots', type=int, default=50)
P.add_argument('--half-life', type=float, default=50)
P.add_argument('--calls', type=int, default=0, help='replay at most this many calls (0: all)')
P.add_argument('--mode', choices=('engine', 'python'), default='engine',
               help='engine: the forward reads the missing experts itself (xtier_cold_attach); python: this script does')
P.add_argument('--out')
A = P.parse_args()
MODEL, KEY, NAMES = '/data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017', 'layers.{layer}.ffn.experts.{e}.', ('w1', 'w3', 'w2')
H, I, E, K, LIMIT, LAYER, G = 5120, 2304, 384, 6, 10.0, 20, 32
for name, value in {'DYNAMIC': '2', 'DOWN_BF16': '1', 'DOWN_GRAIN': '24', 'SPIN_COUNT': '1048576', 'THP': '0'}.items():
    os.environ.setdefault('XTIER_' + name, value)
import torch
from safetensors import safe_open

torch.set_num_threads(8)
lib = C.CDLL(str(Path(A.lib_dir) / 'libxtier_engine.so'))
u8, f32, i32 = C.POINTER(C.c_uint8), C.POINTER(C.c_float), C.POINTER(C.c_int32)
lib.xtier_create3.restype = lib.xtier_create4.restype = lib.xtier_cold_open.restype = C.c_void_p
lib.xtier_create3.argtypes = [C.c_int] * 4 + [C.c_float, C.c_int] + [C.c_void_p] * 6
lib.xtier_create4.argtypes = [C.c_int] * 5 + [C.c_float, C.c_int] + [C.c_void_p] * 7
lib.xtier_forward.argtypes = [C.c_void_p, C.c_int, C.c_int] + [C.c_void_p] * 4
lib.xtier_place.argtypes = [C.c_void_p, C.c_int, C.c_int]
lib.xtier_slot_pieces.argtypes = [C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)]
lib.xtier_nodes.argtypes = [C.c_void_p]
lib.xtier_destroy.argtypes = [C.c_void_p]
lib.xtier_last_error.restype = lib.xtier_cold_error.restype = C.c_char_p
lib.xtier_cold_open.argtypes = [C.POINTER(C.c_char_p), C.c_int, C.c_int, C.c_int]
lib.xtier_cold_write.argtypes = [C.c_void_p, C.c_int, C.c_longlong, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_int]
lib.xtier_cold_read.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_longlong), C.POINTER(C.c_void_p),
                                C.POINTER(C.c_size_t), C.c_int]
lib.xtier_cold_stats.argtypes = [C.c_void_p, C.POINTER(C.c_double)]
lib.xtier_cold_close.argtypes = [C.c_void_p]
lib.xtier_cold_attach.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_float]
lib.xtier_cold_plan_stats.argtypes = [C.c_void_p, C.POINTER(C.c_double)]


def load():
    root = Path(MODEL)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    w13 = torch.empty((E, 2 * I, H // 2), dtype=torch.uint8); w2 = torch.empty((E, H, I // 2), dtype=torch.uint8)
    s13 = torch.empty((E, 2 * I, H // G), dtype=torch.uint8); s2 = torch.empty((E, H, I // G), dtype=torch.uint8)
    grouped = {}
    for e in range(E):
        p = KEY.format(layer=LAYER, e=e)
        for role, n in zip(('gate', 'up', 'down'), NAMES):
            grouped.setdefault(index[p + n + '.weight'], []).append((e, p, n, role))
    for file, items in grouped.items():
        with safe_open(root / file, framework='pt', device='cpu') as f:
            for e, p, n, role in items:
                w, s, rows = {'gate': (w13[e], s13[e], slice(0, I)), 'up': (w13[e], s13[e], slice(I, 2 * I)),
                              'down': (w2[e], s2[e], slice(None))}[role]
                w[rows].copy_(f.get_tensor(p + n + '.weight').view(torch.uint8))
                s[rows].copy_(f.get_tensor(p + n + '.scale').view(torch.uint8))
    return w13, w2, s13, s2, torch.ones((E, 2), dtype=torch.float32), torch.ones((E,), dtype=torch.float32)


def trace_calls(path):
    """The layer's decode calls: (tokens, k) expert ids per call, from the route trace."""
    head = struct.Struct('<QIIIHBBhBBI')
    data = open(path, 'rb').read()
    pos, out = 0, []
    while pos + head.size <= len(data):
        t, wall, gate, down, layer, m, k, rc, active, _, seq = head.unpack_from(data, pos)
        pos += head.size
        ids = np.frombuffer(data, np.int32, m * k, pos).reshape(m, k)
        pos += 4 * m * k
        if layer == LAYER:
            out.append(ids.copy())
    return out


def diskstats(names):
    sectors = {}
    for line in open('/proc/diskstats'):
        f = line.split()
        if f[2] in names:
            sectors[f[2]] = int(f[5])
    return sum(sectors.values()) * 512


def cached_bytes():
    for line in open('/proc/meminfo'):
        if line.startswith('Cached:'):
            return int(line.split()[1]) * 1024


w13, w2, s13, s2, g13, g2 = load()
calls = trace_calls(A.trace)[:A.calls or None]
counts = np.zeros(E)
for ids in calls:
    counts[np.unique(ids[ids >= 0])] += 1
order = np.argsort(-counts)                      # most used first
gpu = set(order[:A.gpu_slots].tolist())          # the hot-expert pool's experts: computed on the GPU
ptr = lambda t: C.c_void_p(t.data_ptr())
full = lib.xtier_create3(H, I, E, A.threads, LIMIT, 1, *(ptr(t) for t in (w13, w2, s13, s2, g13, g2)))
assert full, lib.xtier_last_error()
nodes = lib.xtier_nodes(full)
IN, HN = I // nodes, H // nodes
piece = [IN * H // 2, IN * H // 2, IN * H // G, IN * H // G, HN * I // 2, HN * I // G]
node_bytes = sum(piece)
record = nodes * node_bytes
assert node_bytes % 512 == 0
# the partitions holding the store (/proc/diskstats has a line for the disk and one for each partition: count one)
devices = {os.path.basename(os.path.realpath(f'/sys/dev/block/{os.major(os.stat(d).st_dev)}:{os.minor(os.stat(d).st_dev)}'))
           for d in A.store}
print(f'layer {LAYER}: {len(calls)} decode calls, {nodes} nodes, record {record / 1e6:.1f} MB per expert, store on {sorted(devices)}',
      flush=True)
rng = np.random.default_rng(1026)
results = []
for x in A.x:
    slots = E - int(round(x * E))
    resident = np.ascontiguousarray(order[:slots], dtype=np.int32)   # the GPU's experts first: pinned in RAM
    cold = [int(e) for e in order[slots:]]
    stored = [int(e) for e in order[A.gpu_slots:]]   # every expert that can leave RAM has its record on NVMe
    paths = [str(Path(d) / f'ds41-layer{LAYER}-x{int(round(x * 100))}.cold') for d in A.store]
    store = lib.xtier_cold_open((C.c_char_p * len(paths))(*[p.encode() for p in paths]), len(paths), A.io_threads, 1)
    assert store, lib.xtier_cold_error()
    where = {}                                   # expert -> [(file, offset) of node part n]
    D = len(paths)
    share = -(-nodes // D) * node_bytes          # a record's bytes on each drive (cold.py's striped layout)
    t0 = time.time()
    for rank, e in enumerate(stored):
        place = [(n % D, rank * share + (n // D) * node_bytes) for n in range(nodes)]
        for n in range(nodes):
            parts = [w13[e, n * IN:(n + 1) * IN], w13[e, I + n * IN:I + (n + 1) * IN], s13[e, n * IN:(n + 1) * IN],
                     s13[e, I + n * IN:I + (n + 1) * IN], w2[e, n * HN:(n + 1) * HN], s2[e, n * HN:(n + 1) * HN]]
            assert [p.numel() for p in parts] == piece and all(p.is_contiguous() for p in parts)
            rc = lib.xtier_cold_write(store, place[n][0], place[n][1], (C.c_void_p * 6)(*[p.data_ptr() for p in parts]),
                                      (C.c_size_t * 6)(*piece), 6)
            assert rc == 0, lib.xtier_cold_error()
        where[e] = place
    written = time.time() - t0
    plan = lib.xtier_create4(H, I, E, slots, A.threads, LIMIT, 1, *(ptr(t) for t in (w13, w2, s13, s2, g13, g2)),
                             resident.ctypes.data_as(C.c_void_p))
    assert plan, lib.xtier_last_error()
    if A.mode == 'engine':   # the forward reads what it misses; the GPU's experts have no record and stay
        files = np.full(E * nodes, -1, dtype=np.int32)   # per expert and node part
        offsets = np.zeros(E * nodes, dtype=np.int64)
        for e, place in where.items():
            for n, (f, off) in enumerate(place):
                files[e * nodes + n], offsets[e * nodes + n] = f, off
        initial = (counts / len(calls) * A.half_life).astype(np.float32)   # about the decayed count at equilibrium
        assert lib.xtier_cold_attach(plan, store, files.ctypes.data_as(C.c_void_p), offsets.ctypes.data_as(C.c_void_p),
                                     initial.ctypes.data_as(C.c_void_p), A.half_life) == 0, lib.xtier_last_error()
    slot_of = {int(e): j for j, e in enumerate(resident)}
    expert_of = {j: int(e) for j, e in enumerate(resident)}
    pieces = {}                                  # slot -> 8 x 6 destination pointers

    def slot_pieces(j):
        if j not in pieces:
            out = []
            for n in range(nodes):
                p, b = (C.c_void_p * 6)(), (C.c_size_t * 6)()
                assert lib.xtier_slot_pieces(plan, j, n, p, b) == 0
                assert list(b) == piece
                out += list(p)
            pieces[j] = out
        return pieces[j]

    score, stamp = np.zeros(E), np.zeros(E)
    lam = 0.5 ** (1 / A.half_life)
    stats0 = (C.c_double * 4)()
    lib.xtier_cold_stats(store, stats0)
    disk0, cache0 = diskstats(devices), cached_bytes()
    loads, load_s, fwd_s, misses_per_call, mismatch, worst = 0, 0.0, 0.0, [], 0, 0.0
    for i, ids in enumerate(calls):
        cpu = np.where(np.isin(ids, list(gpu)), -1, ids).astype(np.int32)
        used = [int(e) for e in np.unique(cpu[cpu >= 0])]
        score[used] = score[used] * lam ** (i - stamp[used]) + 1
        stamp[used] = i
        miss = [e for e in used if e not in slot_of] if A.mode == 'python' else []
        if miss:
            busy = set(used) | gpu
            jobs_f, jobs_o, dst = [], [], []
            for e in miss:
                cand = [j for j, v in expert_of.items() if v not in busy]
                now = np.array([score[expert_of[j]] * lam ** (i - stamp[expert_of[j]]) for j in cand])
                j = cand[int(np.argmin(now))]
                del slot_of[expert_of[j]]
                expert_of[j], slot_of[e] = e, j
                for f, off in where[e]:
                    jobs_f.append(f); jobs_o.append(off)
                dst += slot_pieces(j)
            t = time.perf_counter()
            rc = lib.xtier_cold_read(store, len(jobs_f), (C.c_int * len(jobs_f))(*jobs_f), (C.c_longlong * len(jobs_o))(*jobs_o),
                                     (C.c_void_p * len(dst))(*dst), (C.c_size_t * len(dst))(*(piece * len(jobs_f))), 6)
            load_s += time.perf_counter() - t
            assert rc == 0, lib.xtier_cold_error()
            for e in miss:
                assert lib.xtier_place(plan, e, slot_of[e]) >= -1, lib.xtier_last_error()
            loads += len(miss)
        misses_per_call.append(len(miss))
        m = cpu.shape[0]
        xb = torch.from_numpy(rng.standard_normal((m, H), dtype=np.float32)).bfloat16()
        rw = torch.from_numpy(rng.random((m, K), dtype=np.float32))
        ids_t = torch.from_numpy(cpu.copy())
        y0, y1 = torch.zeros((m, H)), torch.zeros((m, H))
        args = lambda y: (C.c_int(m), C.c_int(K), ptr(xb.view(torch.int16)), ptr(ids_t), ptr(rw), ptr(y))
        if used:
            t = time.perf_counter()
            assert lib.xtier_forward(plan, *args(y1)) >= 0, lib.xtier_last_error()
            fwd_s += time.perf_counter() - t
            assert lib.xtier_forward(full, *args(y0)) >= 0, lib.xtier_last_error()
            mismatch += int(not torch.equal(y0, y1))
            worst = max(worst, float((y1 - y0).norm() / y0.norm().clamp_min(1e-30)))
    stats1 = (C.c_double * 4)()
    lib.xtier_cold_stats(store, stats1)
    asked = stats1[0] - stats0[0]
    if A.mode == 'engine':   # the plan's own count of the experts it read and the time it waited
        ps = (C.c_double * 6)()
        lib.xtier_cold_plan_stats(plan, ps)
        loads, load_s = int(ps[1]), ps[2]
        misses_per_call = [loads / len(calls)]
    disk = diskstats(devices) - disk0
    grew = cached_bytes() - cache0
    # the drives' bandwidth: every cold expert read once, all at once
    probe = min(len(cold), slots)
    jobs_f, jobs_o, dst = [], [], []
    for r, e in enumerate(cold[:probe]):
        for f, off in where[e]:
            jobs_f.append(f); jobs_o.append(off)
        dst += slot_pieces(r)
    t = time.perf_counter()
    assert lib.xtier_cold_read(store, len(jobs_f), (C.c_int * len(jobs_f))(*jobs_f), (C.c_longlong * len(jobs_o))(*jobs_o),
                               (C.c_void_p * len(dst))(*dst), (C.c_size_t * len(dst))(*(piece * len(jobs_f))), 6) == 0
    bw = probe * record / (time.perf_counter() - t)
    lib.xtier_destroy(plan)
    lib.xtier_cold_close(store)
    for p in paths:
        os.remove(p)
    mpc = np.array(misses_per_call)
    res = dict(x=x, slots=slots, cold=len(cold), calls=len(calls), mismatched_calls=mismatch, worst_rel_l2=worst,
               overlap=os.environ.get('XTIER_COLD_OVERLAP', '1'), loads=loads,
               misses_per_call=round(float(mpc.mean()), 3), p99_misses=int(np.percentile(mpc, 99)),
               load_ms_per_call=round(load_s * 1e3 / len(calls), 3), forward_ms_per_call=round(fwd_s * 1e3 / len(calls), 3),
               compute_ms_per_call=round((fwd_s - (load_s if A.mode == 'engine' else 0)) * 1e3 / len(calls), 3), mode=A.mode,
               load_GBps=round(loads * record / load_s / 1e9, 2) if load_s else None, burst_GBps=round(bw / 1e9, 2),
               requested_GB=round(asked / 1e9, 3), drives_read_GB=round(disk / 1e9, 3), page_cache_grew_MB=round(grew / 1e6, 1),
               write_GBps=round(len(stored) * record / written / 1e9, 2))
    results.append(res)
    print(json.dumps(res), flush=True)
lib.xtier_destroy(full)
if A.out:
    Path(A.out).write_text(json.dumps(results, indent=1))
