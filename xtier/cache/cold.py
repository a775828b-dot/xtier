"""Cold experts on NVMe (XTIER_COLD=1): host RAM keeps only part of every partly cached MoE layer's routed experts.
The store on NVMe (one file per drive, every record split across the drives by node part; csrc/cold/store.cpp,
O_DIRECT both ways: no page cache) holds a record of every expert, so any expert can leave RAM and one store serves
every XTIER_COLD_FRACTION. A forward reads the experts it routes to that are not in RAM into the slots of the experts
with the lowest decayed use counts (xtier_cold_attach). The GPU hot-expert pool uploads only experts that are in RAM
(pinned for the copy); a GPU prefill reads the layer's cold experts it routes to into a pinned staging area while it
uploads the RAM slots, and uploads them after (PrefillCold).

The drives' bandwidth sizes it: decode waits for the experts it misses (DeepSeek V4.1 Flash, the store on four PCIe
5.0 drives: the agent's decode step +5 / +15 / +26 / +48 / +78 % at 10 / 20 / 30 / 40 / 50 %, about 0.5 ms a miss), a
prefill call reads every cold expert of the layer it routes to, the next layer's ahead (X x the experts' bytes / the
drives' bandwidth).

  XTIER_COLD=1                   on
  XTIER_COLD_PATHS=dir[:dir...]  one directory per drive; node part n of every record goes to drive n % drives
  XTIER_COLD_PARTS=3:3:2         node parts of a record per drive instead (a faster drive takes more; the sum is the
                                 engine's node count), or units with XTIER_COLD_UNITS=3 (the sum: 3 x the node count)
  XTIER_COLD_UNITS=3             every node part in three units on the drives (its gate piece, its up piece, the rest):
                                 finer shares for drives of unequal speed (four PCIe 5.0 and two PCIe 4.0 drives:
                                 XTIER_COLD_PARTS=5:5:5:5:2:2); the default 1 keeps node parts whole
  XTIER_COLD_FRACTION=0.3        share of each layer's experts not in RAM
  XTIER_COLD_HALF_LIFE=50        forwards over which a use count halves (decode steps)
  XTIER_COLD_THREADS=8           readers per drive
  XTIER_COLD_KEY                 store name (default: the model directory's name and the layer shape)
  XTIER_FAST_LOAD=1              fast load (backend/direct.py): a layer whose records were written from (or checked
                                 against) this checkpoint skips its expert tensors, its RAM slots are read from the
                                 store (loadable, attach_loaded)
The ranking is XTIER_CACHE_PROFILE's (the hot-expert pool's), then expert order. The store is reused across
restarts: the manifest (first directory, KEY.json) names every layer's records, which are checked by two 4 KiB
samples each; a layer whose records are missing or differ is written again (a few seconds per layer). Each layer
also notes the checkpoint its records match (source: the model files' names, sizes and modification times), saved
only after its records are flushed to the drives.

Only one copy of the experts (a model larger than the drive it was downloaded to can spare): the store holds every
expert, so tools/cold_punch.py can take their bytes out of the checkpoint files once it has compared them with the
records (holes: the files keep their sizes, modification times and everything else). KEY.ranges, next to the manifest,
says where each expert tensor lies in the files (noted while loading, save_ranges). The marker XTIER-COLD-ONLY in the
model directory then lists the layers that can only load from the store; any other way of loading them stops the start
(require_checkpoint) instead of reading zeros, and such a layer's records are never rewritten.
"""
import hashlib
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
MARKER = 'XTIER-COLD-ONLY'   # in the model directory (tools/cold_punch.py); not a .json file: source() stays the same


def enabled():
    return os.environ.get('XTIER_COLD', '0') == '1'


def _float(name, default):
    return float(os.environ.get(name, default))


def _model_dir():
    try:
        from sglang.srt.runtime_context import get_model
        return Path(get_model().model_path.rstrip('/'))
    except Exception:   # noqa: BLE001 - outside a server (tests: XTIER_COLD_SOURCE)
        return Path(os.environ.get('XTIER_COLD_SOURCE', 'model'))


def _model_name():
    return _model_dir().name


def load_enabled():
    return os.environ.get('XTIER_FAST_LOAD', '0') == '1'


_source = []
_marker = []


def checkpoint_id(d):
    """Directory d's .safetensors and .json files' names, sizes and modification times."""
    h = hashlib.sha1()
    for p in sorted(list(d.glob('*.safetensors')) + list(d.glob('*.json'))):
        s = p.stat()
        h.update(f'{p.name} {s.st_size} {s.st_mtime_ns}\n'.encode())
    return h.hexdigest()[:16]


def source():
    """The checkpoint a layer's records match."""
    if not _source:
        _source.append(checkpoint_id(_model_dir()))
    return _source[0]


def cold_only():
    """The marker of a checkpoint whose experts' bytes tools/cold_punch.py took out, or None. Its files must be the ones
    it punched (a new download over the old one leaves a stale marker)."""
    if not _marker:
        d = _model_dir()
        marker = json.loads((d / MARKER).read_text()) if (d / MARKER).exists() else None
        if marker is not None:
            for name, (size, mtime) in marker['files'].items():
                s = (d / name).stat()
                if (s.st_size, s.st_mtime_ns) != (size, mtime):
                    raise RuntimeError(f'xtier cold: {d / name} changed since tools/cold_punch.py punched it (or it did '
                                       f'not finish): tools/cold_punch.py {d} --restore, or remove {MARKER} if this is '
                                       'a new download')
        _marker.append(marker)
    return _marker[0]


def only_in_store(layer_id):
    """Whether the checkpoint no longer holds the bytes of layer_id's experts."""
    marker = cold_only()
    return marker is not None and str(layer_id) in marker['layers']


def require_checkpoint(layer_id):
    """A layer about to load from anything but the store: stops the start if its experts are only in the store."""
    if only_in_store(layer_id):
        marker = cold_only()
        raise RuntimeError(
            f'xtier cold: layer {layer_id}: the checkpoint no longer holds its experts ({MARKER}; they would read as '
            f'zeros) and it cannot load from the store. It needs XTIER_COLD=1, XTIER_FAST_LOAD=1, a partly cached layer '
            f"and the records: XTIER_COLD_PATHS={':'.join(marker['paths'])}, store {marker['key']}. "
            f'tools/cold_punch.py {_model_dir()} --restore writes the experts back.')


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
                logger.warning('xtier cold: unreadable manifest %s', self.manifest_path)
            if manifest is not None and manifest.get('version') == 2:
                self.manifest = manifest
            else:   # version 1 (a whole record on one drive) or unreadable: the files start again
                if cold_only() is not None:
                    raise RuntimeError(f'xtier cold: {self.manifest_path} is not usable and these records are the only '
                                       'copy of the experts: nothing is rewritten')
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
        """The manifest replaced on the drive before the records it drops are overwritten (fsync, then the rename)."""
        tmp = self.manifest_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            f.write(json.dumps(self.manifest))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.manifest_path)
        _fsync(self.manifest_path.parent, os.fsync)

    def sync(self):
        """The records written so far on the drives themselves (their write caches flushed): only then may a manifest
        entry vouch for them, XTIER_FAST_LOAD reads them without checking against the checkpoint."""
        for p in self.paths:
            _fsync(p, os.fdatasync)


def _fsync(path, how):
    fd = os.open(path, os.O_RDONLY)
    try:
        how(fd)
    finally:
        os.close(fd)


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
        self.reads = []   # where the loader's expert tensors lie in the checkpoint files (direct.locate; save_ranges)


def _units():
    """XTIER_COLD_UNITS: the units a record's node part is split into across the drives, 1 (the part whole) or 3."""
    u = (os.environ.get('XTIER_COLD_UNITS') or '1').strip()
    if u not in ('1', '3'):
        raise RuntimeError(f'xtier cold: XTIER_COLD_UNITS={u}: 1 or 3')
    return int(u)


def _unit_pieces(sizes, units):
    """Per unit of a node part, its pieces as (piece, offset in the piece, bytes), in the order they lie on the drive.
    One unit: the six pieces. Three: the gate piece and the up piece, each as four 512-byte aligned quarters, then the
    other four pieces: four pieces a read, as the engine's reads take them (csrc/engine/engine.cpp ensure_resident)."""
    if units == 1:
        return [[(p, 0, s) for p, s in enumerate(sizes)]]
    quarter = sizes[0] // 4 // 512 * 512   # the gate and the up piece are the same size
    return ([[(p, r * quarter, quarter if r < 3 else sizes[p] - 3 * quarter) for r in range(4)] for p in (0, 1)]
            + [[(p, 0, sizes[p]) for p in range(2, 6)]])


def _drive_parts(nodes, drives, unit_bytes=None):
    """Per unit of a record (node part n's unit t is unit n * units + t; one unit a part without unit_bytes), its drive:
    XTIER_COLD_PARTS counts of units per drive (or an even split). Whole node parts are handed out in turn (the layout
    of the stores written before units); units go largest first to the drive whose count they fill least (its bytes
    over its count), so that a record's bytes on a drive follow its count."""
    units = 1 if unit_bytes is None else len(unit_bytes)
    total = nodes * units
    spec = os.environ.get('XTIER_COLD_PARTS')
    counts = [int(c) for c in spec.split(':')] if spec else [total // drives + (d < total % drives) for d in range(drives)]
    if len(counts) != drives or sum(counts) != total or min(counts) < 1:
        raise RuntimeError(f'xtier cold: XTIER_COLD_PARTS={spec} needs {drives} counts of at least 1 summing to {total}')
    if units == 1:
        return counts, hand_out(counts)
    size = [unit_bytes[u % units] for u in range(total)]
    load, left, drive_of = [0] * drives, list(counts), [0] * total
    for u in sorted(range(total), key=lambda u: (-size[u], u)):
        d = min((d for d in range(drives) if left[d]), key=lambda d: ((load[d] + size[u]) / counts[d], d))
        drive_of[u], load[d], left[d] = d, load[d] + size[u], left[d] - 1
    return counts, drive_of


def _unit_layout(sizes, nodes, drives):
    """(the units' pieces, the units' bytes, the units per drive, each unit's drive) of XTIER_COLD_UNITS and
    XTIER_COLD_PARTS for node parts of these piece sizes."""
    units = _unit_pieces(sizes, _units())
    unit_bytes = [sum(b for _, _, b in u) for u in units]
    counts, drive_of = _drive_parts(nodes, drives, unit_bytes)
    return units, unit_bytes, counts, drive_of


def _layout_matches(entry, record, drives, units, counts, drive_of):
    """Whether a manifest entry's records lie as this layout puts them (entries from before units have no units and no
    map: whole node parts, handed out in turn)."""
    return (entry is not None and entry.get('record') == record and len(entry.get('starts', [])) == drives
            and entry.get('parts', counts) == counts and entry.get('units', 1) == units
            and entry.get('map', drive_of) == drive_of)


def hand_out(counts):
    """Per node part, its drive, from the node parts per drive."""
    left, drive_of, d = list(counts), [], 0
    while len(drive_of) < sum(counts):
        if left[d]:
            drive_of.append(d)
            left[d] -= 1
        d = (d + 1) % len(counts)
    return drive_of


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


def _layout(H, I, nodes, scale_format):
    """The six pieces of one node part of a record (xtier_slot_pieces order), the part's and the record's bytes."""
    group = 32 if scale_format == 'mxfp4' else 16
    IN, HN = I // nodes, H // nodes
    sizes = [IN * H // 2, IN * H // 2, IN * H // group, IN * H // group, HN * I // 2, HN * I // group]
    part = sum(sizes)
    if part % 512 or min(sizes) < 4096:
        raise RuntimeError(f'xtier cold: record pieces {sizes} are not whole 512-byte blocks')
    return sizes, part, nodes * part


def _key(E, H, I, scale_format, nodes):
    units = _units()
    return os.environ.get('XTIER_COLD_KEY') or (f'{_model_name()}-E{E}-H{H}-I{I}-{scale_format}-n{nodes}'
                                                + (f'-u{units}' if units > 1 else ''))


def where_records(drive_of, unit_bytes):
    """where(starts, i, u): (drive, offset) of unit u (node part u // units, its unit u % units; units: len(unit_bytes))
    of the layer's i-th stored record. A drive holds its units of every record back to back, in unit order;
    where.share: a record's bytes on each drive."""
    units = len(unit_bytes)
    share = [0] * (max(drive_of) + 1)
    before = []   # a unit's offset among its drive's units of a record
    for u, d in enumerate(drive_of):
        before.append(share[d])
        share[d] += unit_bytes[u % units]

    def where(starts, i, u):
        d = drive_of[u]
        return d, starts[d] + i * share[d] + before[u]
    where.share = share
    return where


def _nodes(threads):
    """The NUMA nodes of the engine's workers for `threads`: the node count of a plan of this process."""
    from xtier.engine import worker_cpus
    found = set()
    for c in worker_cpus(threads):
        found.update(n for n in os.listdir(f'/sys/devices/system/cpu/cpu{c}') if n.startswith('node') and n[4:].isdigit())
    return len(found)


def loadable(lp, hidden, inter, scale_format, threads):
    """Whether the store holds layer lp's records as attach wrote or checked them from this checkpoint: the layer can
    leave its expert tensors unread and fill its RAM slots from the store (attach_loaded). A changed ranking moves the
    RAM slots, not the records: lp takes their order from the manifest. A layer only in the store has no checkpoint
    bytes to match (its files are checked by cold_only, the model's .json files may change)."""
    nodes = _nodes(threads)
    sizes, _, record = _layout(hidden, inter, nodes, scale_format)
    st = store(_key(lp.experts, hidden, inter, scale_format, nodes))
    units, _, counts, drive_of = _unit_layout(sizes, nodes, len(st.paths))
    with st.lock:
        entry = st.manifest['layers'].get(str(lp.layer_id))
        ok = (_layout_matches(entry, record, len(st.paths), len(units), counts, drive_of)
              and (entry.get('source') == source() or only_in_store(lp.layer_id))
              and sorted(entry.get('stored', [])) == sorted(lp.stored))
        if ok:
            lp.stored = entry['stored']
        return ok


class _Ticket:
    """A store read as a future: direct.finish waits for it at the end of loading."""

    def __init__(self, native, ticket):
        self.native, self.ticket = native, ticket

    def result(self):
        self.native.finish(self.ticket)


def attach_loaded(plan, lp, scale_format):
    """After the plan is created from lp.resident without sources (loadable was true, the layer's expert tensors were
    not read): its RAM slots read from the store while the loader goes on (done and their scales checked at the end of
    loading: direct.finish), the cold experts attached as attach does. Returns the layer's PrefillCold."""
    from xtier.backend import direct
    nodes, H, I, E = plan.nodes(), plan.hidden, plan.inter, plan.experts
    sizes, _, record = _layout(H, I, nodes, scale_format)
    st = store(_key(E, H, I, scale_format, nodes))
    units, unit_bytes, _, drive_of = _unit_layout(sizes, nodes, len(st.paths))
    where = where_records(drive_of, unit_bytes)
    t0 = time.time()
    with st.lock:
        starts = st.manifest['layers'][str(lp.layer_id)]['starts']
    index = {e: i for i, e in enumerate(lp.stored)}
    U, jobs = len(units), []
    for j, e in enumerate(lp.resident):
        for n in range(nodes):
            slot = plan.slot_pieces(j, n)
            jobs += [(*where(starts, index[e], n * U + t), [(slot[p][0] + o, b) for p, o, b in unit])
                     for t, unit in enumerate(units)]
    direct.defer(plan, _Ticket(st.native, st.native.submit(jobs, [0] * len(jobs), 1)), scale_format, lp.layer_id)
    return _attach(plan, lp, st, starts, where, sizes, units, nodes, record,
                   f'matching this checkpoint, {plan.slots * record / 1e9:.1f} GB being read into the RAM slots', t0)


def attach(plan, lp, inputs, scale_format):
    """After the plan is created from lp.resident: the records of the layer's stored experts (written, or checked and
    reused), then the plan's forwards handle the cold experts (xtier_cold_attach). Returns the layer's PrefillCold."""
    group = 32 if scale_format == 'mxfp4' else 16
    nodes, H, I, E = plan.nodes(), plan.hidden, plan.inter, plan.experts
    sizes, _, record = _layout(H, I, nodes, scale_format)
    st = store(_key(E, H, I, scale_format, nodes))
    D = len(st.paths)
    units, unit_bytes, counts, drive_of = _unit_layout(sizes, nodes, D)
    U = len(units)
    where = where_records(drive_of, unit_bytes)
    share = where.share   # a record's bytes on each drive
    key = str(lp.layer_id)
    t0 = time.time()
    with st.lock:
        entry = st.manifest['layers'].get(key)
        layout_ok = _layout_matches(entry, record, D, U, counts, drive_of) and entry.get('stored') == lp.stored
        reuse = False
        if layout_ok:   # the records are there: check two samples of each against the weights
            buf = torch.empty(3 * 4096, dtype=torch.uint8)
            skip = -buf.data_ptr() % 4096                       # O_DIRECT: a block-aligned destination
            base = buf.data_ptr() + skip
            reuse = True
            for i, e in enumerate(lp.stored):
                (f0, a), (f1, b) = where(entry['starts'], i, 0), where(entry['starts'], i, nodes * U - 1)
                st.native.read([(f0, a, [(base, 4096)]), (f1, b + unit_bytes[-1] - 4096, [(base + 4096, 4096)])])
                got = bytes(buf[skip:skip + 8192].numpy())
                if zlib.crc32(got[4096:], zlib.crc32(got[:4096])) != _sample(inputs, e, nodes, H, I, group):
                    reuse = False
                    break
        if reuse:
            starts = entry['starts']
            st.reused += 1
            if entry.get('source') != source():   # checked against this checkpoint: the next start may read them
                st.sync()
                entry['source'] = source()
                st.save()
        else:
            size = [len(lp.stored) * b for b in share]   # on each drive
            if layout_ok or (_layout_matches(entry, record, D, U, counts, drive_of)
                             and len(entry.get('stored', [])) >= len(lp.stored)):
                starts = entry['starts']   # rewrite in place
            else:
                starts = list(st.ends)
                st.ends = [s + z for s, z in zip(starts, size)]
            st.manifest['layers'].pop(key, None)
            st.save()

            errors = []

            def write(d):   # drive d: its units of every record, in unit order, back to back
                try:
                    mine = [u for u in range(nodes * U) if drive_of[u] == d]
                    for i, e in enumerate(lp.stored):
                        pieces = []
                        for u in mine:
                            src = _pieces(inputs, e, u // U, nodes, H, I, group)
                            pieces += [(src[p][0] + o, b) for p, o, b in units[u % U]]
                        st.native.write(d, starts[d] + i * share[d], pieces)
                except BaseException as e:   # re-raised after the join: the manifest must not vouch for these records
                    errors.append(e)

            threads = [threading.Thread(target=write, args=(d,)) for d in range(D)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if errors:
                raise errors[0]
            st.sync()
            st.manifest['layers'][key] = dict(record=record, stored=lp.stored, starts=starts, parts=counts,
                                              ends=[s + z for s, z in zip(starts, size)], source=source(),
                                              **({'units': U, 'map': drive_of} if U > 1 else {}))
            st.save()
            st.written += 1
    return _attach(plan, lp, st, starts, where, sizes, units, nodes, record, 'reused' if reuse else 'written', t0)


def _attach(plan, lp, st, starts, where, sizes, units, nodes, record, how, t0):
    """The cold experts handled by the plan's forwards, the layer's PrefillCold linked to the previous layer's."""
    E, U = plan.experts, len(units)
    # per expert, node part and unit ([(e * nodes + n) * U + t]; the engine reads XTIER_COLD_UNITS as _units does)
    files, offsets = [-1] * (E * nodes * U), [0] * (E * nodes * U)
    for i, e in enumerate(lp.stored):
        for u in range(nodes * U):
            files[e * nodes * U + u], offsets[e * nodes * U + u] = where(starts, i, u)
    position = {e: r for r, e in enumerate(lp.rank)}
    # hotter experts start ahead, but the GPU pool's first experts at 0: the GPU computes them, their RAM slots go first
    scores = [0.0 if position[e] < lp.gpu_slots else lp.half_life * 1.44 * (1 - position[e] / E) for e in range(E)]
    plan.attach_cold(st.native, files, offsets, scores, lp.half_life)
    plan.dynamic = plan.slots   # slots experts can leave: bounds a CPU call's pass (CpuPlan.forward)
    pc = PrefillCold(plan, st, files, offsets, sizes, units, nodes, len(_layers))
    if _layers:
        _layers[-1][2].next = pc   # read ahead by the previous layer's calls (layers attach in model order)
    _layers.append((lp.layer_id, plan, pc, record))
    _ranges[lp.layer_id] = (sizes, E, lp.reads)
    _start_reporter()
    logger.info('xtier cold: layer %d: %d of %d experts in RAM, %d records (%.1f GB) on %d drives %s in %.1f s',
                lp.layer_id, plan.slots, E, len(lp.stored), len(lp.stored) * record / 1e9, len(st.paths), how,
                time.time() - t0)
    return pc


_layers = []         # (layer id, plan, PrefillCold, record bytes) of every layer with cold experts
_ranges = {}         # layer id -> (piece bytes per node part, experts, LayerPlan.reads), until save_ranges
_reporter = None


def save_ranges():
    """After loading: where this start's cold layers' expert tensors lie in the checkpoint files, next to the manifest
    (KEY.ranges, for tools/cold_punch.py), unless the checkpoint is already punched. A layer not all of whose tensors
    are bytes of the records in a file (with tensor parallelism, say) is left out: it stays in the checkpoint."""
    if _store is None or not _ranges:
        return
    try:
        if cold_only() is None:
            names = {str(p.resolve()): p.name for p in _model_dir().glob('*.safetensors')}
            files, layers = {}, {}
            for layer_id, (sizes, experts, reads) in _ranges.items():
                if len(reads) == 6 * experts and all(r and r[5] is None and r[2] in names for r in reads):
                    layers[str(layer_id)] = dict(sizes=sizes, reads=[
                        (r[0], r[1], files.setdefault(names[r[2]], len(files)), r[3], r[4]) for r in reads])
            path = _store.manifest_path.with_suffix('.ranges')
            with open(f'{path}.tmp', 'w') as f:
                f.write(json.dumps(dict(source=source(), files=list(files), layers=layers)))
            os.replace(f'{path}.tmp', path)
            logger.info('xtier cold: noted the expert tensors of %d of %d layers in %s', len(layers), len(_ranges), path)
    except OSError as e:   # only tools/cold_punch.py needs them
        logger.warning('xtier cold: where the expert tensors lie was not saved: %s', e)
    for _, _, reads in _ranges.values():
        reads.clear()
    _ranges.clear()


def _start_reporter():
    """XTIER_COLD_LOG=60: every this many seconds with new reads, a log line of what decode (forwards of at most 8
    tokens: misses per step), the CPU parts of prefills and the GPU prefills read from NVMe since the last line."""
    global _reporter
    every = _float('XTIER_COLD_LOG', 60)
    if _reporter is not None or every <= 0:
        return

    def run():
        last = [0.0] * 8
        while True:
            time.sleep(every)
            stats = [p.cold_stats() for _, p, _, _ in _layers]
            now = [sum(s[i] for s in stats) for i in (1, 2, 3, 4, 5)]   # loads, s, small forwards, small loads, s
            now += [sum(pc.reads for _, _, pc, _ in _layers), sum(pc.read_s for _, _, pc, _ in _layers)]
            now += [sum(s[6] for s in stats)]   # experts not in RAM XTIER_TAIL_SKIP left out (reads saved)
            d = [a - b for a, b in zip(now, last)]
            record = _layers[0][3] if _layers else 0
            if d[0] or d[5] or d[7]:
                steps = d[2] / max(1, len(_layers))   # one CPU call per layer and decode step
                logger.info('xtier cold: last %.0f s: decode %.0f steps read %d experts (%.1f per step, %.1f GB, '
                            '%.1f s), CPU prefill read %d (%.1f GB, %.1f s), GPU prefill %d (%.1f GB, uploads waited '
                            '%.1f s)%s', every, steps, d[3], d[3] / steps if steps else 0, d[3] * record / 1e9, d[4],
                            d[0] - d[3], (d[0] - d[3]) * record / 1e9, d[1] - d[4], d[5], d[5] * record / 1e9, d[6],
                            f', tail skip saved {d[7]:.0f} reads ({d[7] / steps:.1f} per step)' if d[7] and steps else '')
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

    def __init__(self, plan, st, files, offsets, sizes, units, nodes, index):
        self.plan, self.store, self.files, self.offsets = plan, st, files, offsets
        self.sizes, self.part, self.nodes = sizes, sum(sizes), nodes
        self.units = units   # a node part's units: their pieces (_unit_pieces); files / offsets per expert, part, unit
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
            mine = self._expert_jobs(shards, i, e)
            jobs += mine
            groups += [i] * len(mine)
        self.reads += len(jobs) // (self.nodes * len(self.units))
        _open.add(self)
        return shards, self.store.native.submit(jobs, groups, self.K), skipped

    def _expert_jobs(self, shards, i, e):
        """Reads of expert e's records (every unit of every node part) into staging row i."""
        U = len(self.units)
        return [(self.files[(e * self.nodes + n) * U + t], self.offsets[(e * self.nodes + n) * U + t],
                 [(shards[n][p][0] + i * self.sizes[p] + o, b) for p, o, b in unit])
                for n in range(self.nodes) for t, unit in enumerate(self.units)]

    def _jobs(self, rows):
        """Reads of the given staging rows (the cold experts there) into this call's staging."""
        return [job for i in rows for job in self._expert_jobs(self.shards, i, self.vexpert[self.C + i])]

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
