"""Cold experts on NVMe (XTIER_COLD=1): host RAM keeps only part of every partly cached MoE layer's routed experts.
The store on NVMe (one file per drive, every record split across the drives by node part; csrc/cold/store.cpp,
O_DIRECT both ways: no page cache) holds a record of every expert, so any expert can leave RAM and one store serves
every XTIER_COLD_FRACTION. A forward reads the experts it routes to that are not in RAM into the slots of the experts
with the lowest decayed use counts (xtier_cold_attach). The GPU hot-expert pool uploads only experts that are in RAM
(pinned for the copy); a GPU prefill reads the layer's cold experts it routes to into a pinned staging area while it
uploads the RAM slots, and uploads them after (PrefillCold).

The drives' bandwidth sizes it: decode waits for the experts it misses (DeepSeek V4.1 Flash, the store on four PCIe
4.0 drives: the agent's decode step +8 / +24 / +44 % at 10 / 20 / 30 %), a prefill call reads every cold expert of
the layer it routes to, the next layer's ahead (X x the experts' bytes / the drives' bandwidth).

  XTIER_COLD=1                   on
  XTIER_COLD_PATHS=dir[:dir...]  one directory per drive; node part n of every record goes to drive n % drives
  XTIER_COLD_PARTS=3:3:2         node parts of a record per drive instead (a faster drive takes more; the sum is the
                                 engine's node count)
  XTIER_COLD_FRACTION=0.3        share of each layer's experts not in RAM
  XTIER_COLD_HALF_LIFE=50        forwards over which a use count halves (decode steps)
  XTIER_COLD_THREADS=8           readers per drive
  XTIER_COLD_KEY                 store name (default: the model directory's name and the layer shape)
The ranking is XTIER_CACHE_PROFILE's (the hot-expert pool's), then expert order. The store is reused across
restarts: the manifest (first directory, KEY.json) names every layer's records, which are checked by two 4 KiB
samples each; a layer whose records are missing or differ is written again (a few seconds per layer).
"""
import json
import logging
import os
import threading
import time
import zlib
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_store = None
_store_lock = threading.Lock()
_staging = [None, None]   # the prefills' pinned stagings: layers alternate (a layer's reads run during the last one)
_locked = [None, None]    # the page-locked (cudaHostRegister) start of each staging
_open = set()             # PrefillColds holding a ticket or a read-ahead (finished by the next prepare)


def enabled():
    return os.environ.get('XTIER_COLD', '0') == '1'


def _float(name, default):
    return float(os.environ.get(name, default))


def _model_name():
    try:
        from sglang.srt.runtime_context import get_model
        return Path(get_model().model_path.rstrip('/')).name
    except Exception:   # noqa: BLE001 - outside a server (tests)
        return 'model'


class Store:
    """The process's store: one record file per drive and the manifest of every layer's records."""

    def __init__(self, key):
        from xtier.engine import ColdStore
        dirs = [Path(p) for p in os.environ.get('XTIER_COLD_PATHS', '').split(':') if p]
        if not dirs:
            raise RuntimeError('xtier cold: XTIER_COLD=1 needs XTIER_COLD_PATHS (one directory per drive)')
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.paths = [d / f'{key}.cold' for d in dirs]
        self.manifest_path = dirs[0] / f'{key}.json'
        self.manifest = {'version': 2, 'layers': {}}
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text())
            except ValueError:
                manifest = None
                logger.warning('xtier cold: unreadable manifest %s, writing every layer again', self.manifest_path)
            if manifest is not None and manifest.get('version') == 2:
                self.manifest = manifest
            else:   # version 1 (a whole record on one drive) or unreadable: the files start again
                for p in self.paths:
                    p.unlink(missing_ok=True)
                logger.info('xtier cold: store %s has another layout, writing every layer again', key)
        self.native = ColdStore(self.paths, int(_float('XTIER_COLD_THREADS', 8)))
        self.ends = [0] * len(self.paths)
        for layer in self.manifest['layers'].values():
            for d, end in enumerate(layer['ends'][:len(self.paths)]):
                self.ends[d] = max(self.ends[d], end)
        self.lock = threading.Lock()
        self.written = self.reused = 0

    def save(self):
        tmp = self.manifest_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.manifest))
        os.replace(tmp, self.manifest_path)


def store(key):
    global _store
    with _store_lock:
        if _store is None:
            _store = Store(key)
        return _store


class LayerPlan:
    """One layer's split: resident (the RAM slots' first experts, ranking order) and stored (a record each: every
    expert) experts."""

    def __init__(self, layer_id, experts, gpu_slots):
        from xtier.cache.pool import Profile
        rank = Profile.ranking(layer_id, experts)
        seen = set(rank)
        rank = rank + [e for e in range(experts) if e not in seen]
        self.layer_id, self.experts, self.rank = layer_id, experts, rank
        cold = int(round(_float('XTIER_COLD_FRACTION', 0.3) * experts))
        self.slots = max(1, experts - cold)
        self.resident = rank[:self.slots]
        self.stored = rank
        self.half_life = _float('XTIER_COLD_HALF_LIFE', 50)
        self.gpu_slots = max(0, int(gpu_slots))   # the GPU pool starts with the top of the ranking (pool.fill)


def _drive_parts(nodes, drives):
    """Per node part, its drive: XTIER_COLD_PARTS counts (or an even split) handed out in turn, a drive leaving the
    turn once it has its count."""
    spec = os.environ.get('XTIER_COLD_PARTS')
    counts = [int(c) for c in spec.split(':')] if spec else [nodes // drives + (d < nodes % drives) for d in range(drives)]
    if len(counts) != drives or sum(counts) != nodes or min(counts) < 1:
        raise RuntimeError(f'xtier cold: XTIER_COLD_PARTS={spec} needs {drives} counts of at least 1 summing to {nodes}')
    left, drive_of, d = list(counts), [], 0
    while len(drive_of) < nodes:
        if left[d]:
            drive_of.append(d)
            left[d] -= 1
        d = (d + 1) % drives
    return counts, drive_of


def _pieces(inputs, e, node, nodes, H, I, group):
    """Expert e's six pieces on worker node `node` as the engine shards them (xtier_slot_pieces order)."""
    w13, w2, s13, s2 = inputs[:4]
    IN, HN = I // nodes, H // nodes
    parts = [w13[e, node * IN:(node + 1) * IN], w13[e, I + node * IN:I + (node + 1) * IN], s13[e, node * IN:(node + 1) * IN],
             s13[e, I + node * IN:I + (node + 1) * IN], w2[e, node * HN:(node + 1) * HN], s2[e, node * HN:(node + 1) * HN]]
    assert all(p.is_contiguous() for p in parts)
    return [(p.data_ptr(), p.numel()) for p in parts]


def _sample(inputs, e, nodes, H, I, group):
    """CRC of a record's first and last 4 KiB."""
    first = _pieces(inputs, e, 0, nodes, H, I, group)[0]
    last = _pieces(inputs, e, nodes - 1, nodes, H, I, group)[5]
    import ctypes as C
    a = C.string_at(first[0], 4096)
    b = C.string_at(last[0] + last[1] - 4096, 4096)
    return zlib.crc32(b, zlib.crc32(a))


def attach(plan, lp, inputs, scale_format):
    """After the plan is created from lp.resident: the records of the layer's stored experts (written, or checked and
    reused), then the plan's forwards handle the cold experts (xtier_cold_attach). Returns the layer's PrefillCold."""
    group = 32 if scale_format == 'mxfp4' else 16
    nodes, H, I, E = plan.nodes(), plan.hidden, plan.inter, plan.experts
    IN, HN = I // nodes, H // nodes
    sizes = [IN * H // 2, IN * H // 2, IN * H // group, IN * H // group, HN * I // 2, HN * I // group]
    part = sum(sizes)
    record = nodes * part
    if part % 512 or min(sizes) < 4096:
        raise RuntimeError(f'xtier cold: record pieces {sizes} are not whole 512-byte blocks')
    st = store(os.environ.get('XTIER_COLD_KEY') or f'{_model_name()}-E{E}-H{H}-I{I}-{scale_format}-n{nodes}')
    D = len(st.paths)
    counts, drive_of = _drive_parts(nodes, D)
    share = [c * part for c in counts]   # a record's bytes on each drive
    rank_in = [drive_of[:n].count(drive_of[n]) for n in range(nodes)]   # a part's place among its drive's parts

    def where(starts, i, n):   # (drive, offset) of node part n of the layer's i-th stored record
        d = drive_of[n]
        return d, starts[d] + i * share[d] + rank_in[n] * part

    key = str(lp.layer_id)
    t0 = time.time()
    with st.lock:
        entry = st.manifest['layers'].get(key)
        layout_ok = (entry is not None and entry.get('record') == record and entry.get('stored') == lp.stored
                     and len(entry.get('starts', [])) == D and entry.get('parts', counts) == counts)
        reuse = False
        if layout_ok:   # the records are there: check two samples of each against the weights
            buf = torch.empty(3 * 4096, dtype=torch.uint8)
            skip = -buf.data_ptr() % 4096                       # O_DIRECT: a block-aligned destination
            base = buf.data_ptr() + skip
            reuse = True
            for i, e in enumerate(lp.stored):
                (f0, a), (f1, b) = where(entry['starts'], i, 0), where(entry['starts'], i, nodes - 1)
                st.native.read([(f0, a, [(base, 4096)]), (f1, b + part - 4096, [(base + 4096, 4096)])])
                got = bytes(buf[skip:skip + 8192].numpy())
                if zlib.crc32(got[4096:], zlib.crc32(got[:4096])) != _sample(inputs, e, nodes, H, I, group):
                    reuse = False
                    break
        if reuse:
            starts = entry['starts']
            st.reused += 1
        else:
            size = [len(lp.stored) * b for b in share]   # on each drive
            if layout_ok or (entry is not None and entry.get('record') == record and len(entry.get('starts', [])) == D
                             and entry.get('parts', counts) == counts and len(entry.get('stored', [])) >= len(lp.stored)):
                starts = entry['starts']   # rewrite in place
            else:
                starts = list(st.ends)
                st.ends = [s + z for s, z in zip(starts, size)]
            st.manifest['layers'].pop(key, None)
            st.save()

            def write(d):   # drive d: its node parts of every record, in part order, back to back
                mine = [n for n in range(nodes) if drive_of[n] == d]
                for i, e in enumerate(lp.stored):
                    pieces = [p for n in mine for p in _pieces(inputs, e, n, nodes, H, I, group)]
                    st.native.write(d, starts[d] + i * share[d], pieces)

            threads = [threading.Thread(target=write, args=(d,)) for d in range(D)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            st.manifest['layers'][key] = dict(record=record, stored=lp.stored, starts=starts, parts=counts,
                                              ends=[s + z for s, z in zip(starts, size)])
            st.save()
            st.written += 1
    files, offsets = [-1] * (E * nodes), [0] * (E * nodes)   # per expert and node part
    for i, e in enumerate(lp.stored):
        for n in range(nodes):
            files[e * nodes + n], offsets[e * nodes + n] = where(starts, i, n)
    position = {e: r for r, e in enumerate(lp.rank)}
    # hotter experts start ahead, but the GPU pool's first experts at 0: the GPU computes them, their RAM slots go first
    scores = [0.0 if position[e] < lp.gpu_slots else lp.half_life * 1.44 * (1 - position[e] / E) for e in range(E)]
    plan.attach_cold(st.native, files, offsets, scores, lp.half_life)
    plan.dynamic = plan.slots   # slots experts can leave: bounds a CPU call's pass (CpuPlan.forward)
    pc = PrefillCold(plan, st, files, offsets, sizes, part, nodes, len(_layers))
    if _layers:
        _layers[-1][2].next = pc   # read ahead by the previous layer's calls (layers attach in model order)
    _layers.append((lp.layer_id, plan, pc, record))
    _start_reporter()
    logger.info('xtier cold: layer %d: %d of %d experts in RAM, %d records (%.1f GB) on %d drives %s in %.1f s',
                lp.layer_id, plan.slots, E, len(lp.stored), len(lp.stored) * record / 1e9, D,
                'reused' if reuse else 'written', time.time() - t0)
    return pc


_layers = []         # (layer id, plan, PrefillCold, record bytes) of every layer with cold experts
_reporter = None


def _start_reporter():
    """XTIER_COLD_LOG=60: every this many seconds with new reads, a log line of what decode (forwards of at most 8
    tokens: misses per step), the CPU parts of prefills and the GPU prefills read from NVMe since the last line."""
    global _reporter
    every = _float('XTIER_COLD_LOG', 60)
    if _reporter is not None or every <= 0:
        return

    def run():
        last = [0.0] * 7
        while True:
            time.sleep(every)
            stats = [p.cold_stats() for _, p, _, _ in _layers]
            now = [sum(s[i] for s in stats) for i in (1, 2, 3, 4, 5)]   # loads, s, small forwards, small loads, s
            now += [sum(pc.reads for _, _, pc, _ in _layers), sum(pc.read_s for _, _, pc, _ in _layers)]
            d = [a - b for a, b in zip(now, last)]
            record = _layers[0][3] if _layers else 0
            if d[0] or d[5]:
                steps = d[2] / max(1, len(_layers))   # one CPU call per layer and decode step
                logger.info('xtier cold: last %.0f s: decode %.0f steps read %d experts (%.1f per step, %.1f GB, '
                            '%.1f s), CPU prefill read %d (%.1f GB, %.1f s), GPU prefill %d (%.1f GB, uploads waited '
                            '%.1f s)', every, steps, d[3], d[3] / steps if steps else 0, d[3] * record / 1e9, d[4],
                            d[0] - d[3], (d[0] - d[3]) * record / 1e9, d[1] - d[4], d[5], d[5] * record / 1e9, d[6])
            last = now

    _reporter = threading.Thread(target=run, name='xtier-cold-log', daemon=True)
    _reporter.start()


def _stage(b, need):
    """Staging b for need bytes: pageable memory with its block-aligned range page-locked (cudaHostRegister). torch's
    pinned allocator rounds a block up to a power of two: two 5.06 GB stagings took 16.3 GiB at 70 % cold."""
    from xtier.backend.prefill import _check, _runtime
    cudart = _runtime()
    if _locked[b] is not None:   # a larger one replaces it: no upload or read may still use it
        torch.cuda.synchronize()
        cudart.cudaHostUnregister(_locked[b])
        _staging[b] = _locked[b] = None
    buf = torch.empty(need + 8192, dtype=torch.uint8)
    base = (buf.data_ptr() + 4095) // 4096 * 4096
    _check(cudart.cudaHostRegister(base, (need + 4095) // 4096 * 4096, 0), 'cudaHostRegister')
    _staging[b], _locked[b] = buf, base


class PrefillCold:
    """A GPU prefill's view of a layer with cold experts. Per call (prepare, before its first upload): the virtual
    expert order (the RAM slots, then the experts on NVMe), the global scales in that order, and the cold experts read
    into a pinned staging, in that order and laid out like the engine's shards; an upload of staging rows waits for
    their reads (wait), the RAM slots upload first. With XTIER_COLD_READ_AHEAD=1 (default) a call that routes to at
    least XTIER_COLD_READ_AHEAD_COVER (0.9) of its cold experts also starts reading the next layer's cold experts (all
    of them: its routes are not known yet) into the other staging, so the drives keep reading while this layer computes
    and the next attention runs; a call without a read-ahead (the first layer, a call that routes to fewer, or the
    residency changed since) reads the cold experts it routes to."""

    def __init__(self, plan, st, files, offsets, sizes, part, nodes, index):
        self.plan, self.store, self.files, self.offsets = plan, st, files, offsets
        self.sizes, self.part, self.nodes = sizes, part, nodes
        self.E, self.C = plan.experts, plan.slots
        self.K = self.E - self.C   # experts not in RAM between forwards
        self.index = index         # the layer's position: its staging is index % 2
        self.next = None           # the next layer's PrefillCold
        self.ahead = None          # (cold experts, staging, ticket) read ahead by the previous layer's call
        self.pool = None           # the layer's GPU hot-expert pool (method.py): the experts it holds are not read
        self.skipped = set()       # staging rows not read because the pool held their expert (read if uploaded)
        self.vmap = self.vexpert = self.shards = None
        self.ticket = None
        self.reads = 0        # experts read
        self.read_s = 0.0     # time the uploads waited for their reads

    def _buffers(self):
        b = self.index % 2
        need = self.K * self.part * self.nodes
        if _staging[b] is None or _staging[b].numel() < need + 8192:
            _stage(b, need)
        base = (_staging[b].data_ptr() + 4095) // 4096 * 4096   # O_DIRECT: block-aligned destinations
        shards, at = [], base
        for _ in range(self.nodes):
            row = []
            for size in self.sizes:
                row.append((at, self.K * size))
                at += self.K * size
            shards.append(row)
        return shards

    def _cold(self, table=None):
        table = self.plan.slot_table() if table is None else table
        cold = [e for e in range(self.E) if table[e] < 0]
        if len(cold) != self.K:
            raise RuntimeError(f'xtier cold: {len(cold)} experts outside RAM, expected {self.K}')
        return table, cold

    def _held(self):
        """Per expert, whether the GPU pool holds it (its FP4 prefill copies it from there)."""
        pool = self.pool
        if pool is None or pool.full:
            return None
        return (pool.table >= 0).tolist()

    def _submit(self, cold, routed=None):
        """Reads of the cold experts (only the routed ones, if given; not those the GPU pool holds) into this layer's
        staging: (staging, ticket, the staging rows skipped for the pool)."""
        shards = self._buffers()
        held = self._held()
        jobs, groups, skipped = [], [], set()
        for i, e in enumerate(cold):
            if routed is not None and not routed[e]:   # never computed: its staging rows may hold anything
                continue
            if held is not None and held[e]:
                skipped.add(i)
                continue
            for n in range(self.nodes):
                jobs.append((self.files[e * self.nodes + n], self.offsets[e * self.nodes + n],
                             [(shards[n][j][0] + i * size, size) for j, size in enumerate(self.sizes)]))
                groups.append(i)
        self.reads += len(jobs) // self.nodes
        _open.add(self)
        return shards, self.store.native.submit(jobs, groups, self.K), skipped

    def _jobs(self, rows):
        """Reads of the given staging rows (the cold experts there) into this call's staging."""
        jobs = []
        for i in rows:
            e = self.vexpert[self.C + i]
            for n in range(self.nodes):
                jobs.append((self.files[e * self.nodes + n], self.offsets[e * self.nodes + n],
                             [(self.shards[n][j][0] + i * size, size) for j, size in enumerate(self.sizes)]))
        return jobs

    def prepare(self, globals_, ids, table=None):
        """ids: the call's routes (int32 [m, k] on the device). Snapshot the slot table (no decode forward may move
        slots: the device is synchronized first; `table`: taken by a split prefill before its CPU part started), build
        the virtual order, start the reads (or use the read-ahead), read the next layer ahead. Returns the global
        scales in virtual order."""
        torch.cuda.synchronize()   # the uploads of earlier calls are done: their stagings may be reused
        ahead, self.ahead = self.ahead, None
        for pc in list(_open):     # finished reads, and read-aheads no call will use (an earlier pass stopped)
            pc.finish()
            if pc.ahead is not None:
                pc.store.native.finish(pc.ahead[2])
                pc.ahead = None
            _open.discard(pc)
        table, cold = self._cold(table)
        routed = torch.bincount(ids[ids >= 0].long(), minlength=self.E).tolist()
        if ahead is not None and ahead[0] == cold:
            self.shards, self.ticket, self.skipped = ahead[1], ahead[2], ahead[3]
            _open.add(self)
        else:
            if ahead is not None:   # the residency changed since the read-ahead
                self.store.native.finish(ahead[2])
            self.shards, self.ticket, self.skipped = self._submit(cold, routed)
        cover = sum(1 for e in cold if routed[e]) / max(1, len(cold))
        nxt = self.next
        if (nxt is not None and os.environ.get('XTIER_COLD_READ_AHEAD', '1') == '1'
                and cover >= _float('XTIER_COLD_READ_AHEAD_COVER', 0.9)):
            ncold = nxt._cold()[1]
            nxt.ahead = (ncold,) + nxt._submit(ncold)
        vexpert = [0] * self.E
        for e in range(self.E):
            if table[e] >= 0:
                vexpert[table[e]] = e
        for i, e in enumerate(cold):
            vexpert[self.C + i] = e
        vmap = [0] * self.E
        for v, e in enumerate(vexpert):
            vmap[e] = v
        self.vexpert = vexpert
        dev = globals_[0].device
        self.vmap = torch.tensor(vmap, dtype=torch.int64, device=dev)
        order = torch.tensor(vexpert, dtype=torch.int64, device=dev)
        return tuple(g[order] for g in globals_)

    def wait(self, s0, s1):
        """Before an upload copies staging rows [s0, s1): their reads are done (rows skipped for the GPU pool are read
        now: the pool no longer holds their expert)."""
        if self.ticket is not None:
            t0 = time.perf_counter()
            self.store.native.wait(self.ticket, s0, s1)
            late = [i for i in range(s0, s1) if i in self.skipped]
            if late:
                self.store.native.read(self._jobs(late))
                self.skipped.difference_update(late)
                self.reads += len(late)
            self.read_s += time.perf_counter() - t0

    def finish(self):
        if self.ticket is not None:
            ticket, self.ticket = self.ticket, None
            self.store.native.finish(ticket)
