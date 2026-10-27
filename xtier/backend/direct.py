"""Fast load (XTIER_FAST_LOAD=1, off by default): expert tensors read straight from the checkpoint files into the CPU
engine's shards, all of them read and checked before the server comes up.

SGLang's loader hands every expert tensor of a layer to FusedMoE's weight loader, which copies it into the layer's
host parameters on the loading thread (page faults on the checkpoint's mapping); the hand-off then copies those into
the engine's NUMA shards, for some GPU methods ([up; gate] row order) after one more copy into [gate; up] order
(DeepSeek V4.1 Flash: about 7.4 s a layer, the checkpoint read at about 0.9 GB/s). Here the loader only notes where
each expert tensor lies in its file (SGLang's tensors are views of the file's mapping: /proc/self/maps); once the
layer is complete the engine allocates its shards without filling them and readers fill them from the files: an
O_DIRECT read (no page cache) of each tensor's block-aligned range into a staging buffer, then its node slices copied
to the expert's slot. The shards hold the same bytes as the loader's copies made them; their block scales are checked
as the engine checks its sources.

Taken where the tensors are the shards' bytes as they are: tensor parallelism 1, a w1 / w3 / w2 shard of one expert,
rows x bytes as the engine keeps them, a one-byte dtype; any other layer is loaded as before. With cold experts the
cold store fills a layer's RAM slots instead when its records match the checkpoint (cache/cold.py).

  XTIER_FAST_LOAD=1              on (here and for the cold store's layers)
  XTIER_FAST_LOAD_THREADS=16     readers
  XTIER_FAST_LOAD_CHUNK          bytes of a read at most (32 MiB): tensors next to each other in a file read at once
  XTIER_LOAD_DIGEST=1            log a digest of every layer's shards once they are filled (to compare two loads)
"""
import ctypes as C
import hashlib
import logging
import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)
ALIGN = 4096
SCALE_LIMIT = {'mxfp4': 254, 'nvfp4': 126}   # the engine's check of its sources' block scales (engine.cpp)
# (bulk parameter index, loader shard) -> slot piece (xtier_shard order: gate, up, gate scales, up scales, down, down scales)
PIECES = {(0, 'w1'): 0, (0, 'w3'): 1, (1, 'w2'): 4, (2, 'w1'): 2, (2, 'w3'): 3, (3, 'w2'): 5}


def enabled():
    return os.environ.get('XTIER_FAST_LOAD', '0') == '1'


def digest_enabled():
    return os.environ.get('XTIER_LOAD_DIGEST', '0') == '1'


_maps = []        # (start, end, file offset, path, range) of this process's checkpoint mappings, as last read
_last = [None]    # the mapping of the last tensor found (a file's tensors come one after another)
_maps_lock = threading.Lock()


def _read_maps():
    out = []
    with open('/proc/self/maps') as f:
        for line in f:
            parts = line.split(None, 5)
            if len(parts) == 6 and parts[5].rstrip('\n').endswith('.safetensors'):
                lo, hi = parts[0].split('-')
                out.append((int(lo, 16), int(hi, 16), int(parts[2], 16), parts[5].rstrip('\n'), parts[0]))
    return out


def _current(m):
    """Whether mapping m still maps its file over its range: the loader unmaps a file once its tensors are loaded
    and the next file may be mapped at the same addresses."""
    try:
        return os.readlink(f'/proc/self/map_files/{m[4]}') == m[3]
    except OSError:
        return False


def file_range(address, nbytes):
    """(path, offset) of [address, address + nbytes) in a mapped checkpoint file, or None."""
    with _maps_lock:
        for fresh in (False, True):
            if fresh:
                _maps[:], _last[0] = _read_maps(), None
            for m in ([_last[0]] if _last[0] else []) + _maps:
                if m[0] <= address and address + nbytes <= m[1]:
                    if not _current(m):
                        break   # read the maps again
                    _last[0] = m
                    return m[3], address - m[0] + m[2]
    return None


def locate(bulk, shard, expert, w, hidden, inter, experts, scale_format):
    """(expert, piece, path, offset, bytes) of checkpoint tensor `w` (shard `shard` of expert `expert`, for the
    format's bulk parameter number `bulk`) when it is a slot piece's bytes as they are, else None."""
    piece = PIECES.get((bulk, shard))
    try:
        expert = int(expert)
    except (TypeError, ValueError):
        return None
    if piece is None or not 0 <= expert < experts:
        return None
    group = 32 if scale_format == 'mxfp4' else 16
    H, I = hidden, inter
    rows, cols = ((I, H // 2), (I, H // 2), (I, H // group), (I, H // group), (H, I // 2), (H, I // group))[piece]
    if w.dim() != 2 or tuple(w.shape) != (rows, cols) or w.element_size() != 1 or not w.is_contiguous():
        return None
    where = file_range(w.data_ptr(), rows * cols)
    return None if where is None else (expert, piece, where[0], where[1], rows * cols)


class Reader:
    """Threads that read file ranges with O_DIRECT into their staging buffers and copy parts of them out."""

    def __init__(self, threads):
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix='xtier-direct')
        self.local = threading.local()
        self.lock = threading.Lock()
        self.fds = {}
        self.buffers = []

    def _fd(self, path):
        with self.lock:
            fd = self.fds.get(path)
            if fd is None:
                fd = self.fds[path] = os.open(path, os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC)
            return fd

    def _staging(self, size):
        buf = getattr(self.local, 'buf', None)
        if buf is None or len(buf) < size:
            buf = mmap.mmap(-1, max(size, 8 << 20))   # page-aligned, as O_DIRECT needs
            with self.lock:
                self.buffers.append(buf)
            self.local.buf, self.local.addr = buf, C.addressof(C.c_char.from_buffer(buf))
        return buf, self.local.addr

    def _read(self, path, offset, nbytes, targets):
        fd = self._fd(path)
        a0, a1 = offset // ALIGN * ALIGN, -(-(offset + nbytes) // ALIGN) * ALIGN
        buf, addr = self._staging(a1 - a0)
        view, need, got = memoryview(buf), offset + nbytes - a0, 0
        try:
            while got < need:
                n = os.preadv(fd, [view[got:a1 - a0]], a0 + got)
                if n <= 0:
                    raise OSError(f'xtier: {path}: the read ended at {a0 + got}, the tensor at {offset + nbytes}')
                got += n
        finally:
            view.release()
        base = addr + offset - a0
        for src, dst, size in targets:
            C.memmove(dst, base + src, size)

    def submit(self, path, offset, nbytes, targets):
        """Bytes [offset, offset + nbytes) of `path`; (start in them, destination, bytes) of each part to copy out."""
        return self.pool.submit(self._read, path, offset, nbytes, targets)

    def close(self):
        self.pool.shutdown(wait=True)
        for fd in self.fds.values():
            os.close(fd)
        for buf in self.buffers:
            buf.close()
        self.fds, self.buffers = {}, []


_reader = None


def reader():
    global _reader
    if _reader is None:
        _reader = Reader(int(os.environ.get('XTIER_FAST_LOAD_THREADS', '16')))
    return _reader


def close():
    """After loading: the started layers finished, then the readers, their files and staging buffers go."""
    global _reader
    finish()
    if _reader is not None:
        _reader.close()
        _reader = None


_pending = []   # (plan, futures, scale format, layer id) of layers started and not finished


def start(plan, jobs, scale_format, layer_id):
    """The noted reads of a layer (locate) into the shards of `plan` (created without sources; slot e = expert e),
    started on the readers; the loader goes on with the next layer while they run. finish() waits for them (before
    anything reads the shards: the end of loading). Returns the bytes being read."""
    futures, total = _submit(plan, jobs)
    _pending.append((plan, futures, scale_format, layer_id))
    return total


def defer(plan, future, scale_format, layer_id):
    """A layer whose shards another reader fills (the cold store): finish() waits for `future` and checks them."""
    _pending.append((plan, [future], scale_format, layer_id))


def finish():
    """Every started layer's reads done, its block scales checked (and with XTIER_LOAD_DIGEST its digest logged)."""
    while _pending:
        plan, futures, scale_format, layer_id = _pending.pop(0)
        for f in futures:
            f.result()
        check_scales(plan, scale_format)
        if digest_enabled():
            logger.info('xtier: layer %d: shards digest %s', layer_id, digest(plan))


def fill(plan, jobs, scale_format):
    """start() and wait: the layer's shards filled and checked; returns the bytes read."""
    futures, total = _submit(plan, jobs)
    for f in futures:
        f.result()
    check_scales(plan, scale_format)
    return total


def _submit(plan, jobs):
    """Tensors next to each other in a file are read together, in file order, up to XTIER_FAST_LOAD_CHUNK bytes
    (32 MiB) a read: one stream through each file (a drive without DRAM collapses under many scattered streams)."""
    shards, slots = plan.shards(), plan.slots
    per = [[nbytes // slots for _, nbytes in node] for node in shards]   # each piece's bytes per slot
    nodes, total, chunk = len(shards), 0, int(os.environ.get('XTIER_FAST_LOAD_CHUNK', 32 << 20))
    reads = []   # [path, start, end, targets]
    for expert, piece, path, offset, nbytes in sorted(jobs, key=lambda j: (j[2], j[3])):
        step = nbytes // nodes   # the tensor's rows split evenly over the nodes, in node order
        if step * nodes != nbytes or any(per[n][piece] != step for n in range(nodes)):
            raise RuntimeError(f'xtier: expert {expert} piece {piece}: {nbytes} bytes do not split into the shards')
        r = reads[-1] if reads else None
        if r is None or r[0] != path or not 0 <= offset - r[2] <= ALIGN or offset + nbytes - r[1] > chunk:
            r = [path, offset, offset, []]
            reads.append(r)
        r[3] += [(offset - r[1] + n * step, shards[n][piece][0] + expert * step, step) for n in range(nodes)]
        r[2] = offset + nbytes
        total += nbytes
    return [reader().submit(path, first, end - first, targets) for path, first, end, targets in reads], total


def check_scales(plan, scale_format):
    """Every block scale of the shards within what the engine accepts (as it checks its sources)."""
    limit = SCALE_LIMIT[scale_format]
    for n, node in enumerate(plan.shards()):
        for piece in (2, 3, 5):
            addr, nbytes = node[piece]
            if nbytes and int(np.ctypeslib.as_array((C.c_uint8 * nbytes).from_address(addr)).max()) > limit:
                raise ValueError(f'xtier: unsupported block scale in node {n} piece {piece} (above {limit})')


def digest(plan):
    """SHA-1 over the shards in node and piece order (hashed on threads: hashlib releases the GIL)."""
    parts = [(addr, nbytes) for node in plan.shards() for addr, nbytes in node]

    def one(part):
        h = hashlib.sha1()
        h.update(memoryview((C.c_uint8 * part[1]).from_address(part[0])))
        return h.digest()

    with ThreadPoolExecutor(max_workers=16) as pool:
        return hashlib.sha1(b''.join(pool.map(one, parts))).hexdigest()[:16]
