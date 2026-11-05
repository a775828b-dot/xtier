"""Only one copy of a big model's experts: the cold store (xtier/cache/cold.py) holds every expert, so their bytes can
leave the checkpoint files. A start with XTIER_COLD=1 wrote the records and noted where each expert tensor lies in the
files (KEY.ranges, next to the manifest). With the server stopped:

  python tools/cold_punch.py MODEL_DIR --paths DIR[:DIR...] --dry-run   compare every expert byte with its records
  python tools/cold_punch.py MODEL_DIR --paths DIR[:DIR...]             compare, then punch holes in the files
  python tools/cold_punch.py MODEL_DIR --restore                        write the expert bytes back from the records

Nothing is punched unless every byte matched. The holes free the expert tensors' blocks and nothing else: the files
keep their sizes, modification times, headers and other tensors (SGLang's loader goes through them as before, XTier does
not read the punched tensors), and the marker XTIER-COLD-ONLY in the model directory makes any start that would read
them stop (cold.require_checkpoint). The records are then the only copy: losing a drive of the store means downloading
the model again, and SGLang without XTier cannot load this directory.
"""
import argparse
import ctypes as C
import json
import mmap
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xtier.cache import cold  # noqa: E402

ALIGN = 4096
PUNCH_HOLE = 0x03   # FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE
libc = C.CDLL(None, use_errno=True)
libc.memcmp.argtypes = (C.c_void_p, C.c_void_p, C.c_size_t)
libc.fallocate.argtypes = (C.c_int, C.c_int, C.c_int64, C.c_int64)
_local = threading.local()


def stop(message):
    sys.exit(f'cold_punch: {message}')


def read(fd, offset, nbytes, slot):
    """Bytes [offset, offset + nbytes) of an O_DIRECT descriptor: their address in the thread's buffer `slot`."""
    a0, a1 = offset // ALIGN * ALIGN, -(-(offset + nbytes) // ALIGN) * ALIGN
    buffers = _local.__dict__.setdefault('buffers', {})
    if slot not in buffers or len(buffers[slot][0]) < a1 - a0:
        buf = mmap.mmap(-1, a1 - a0)   # page-aligned, as O_DIRECT needs
        buffers[slot] = buf, C.addressof(C.c_char.from_buffer(buf))
    buf, addr = buffers[slot]
    view, got = memoryview(buf), 0
    try:
        while got < offset + nbytes - a0:
            n = os.preadv(fd, [view[got:a1 - a0]], a0 + got)
            if n <= 0:
                raise OSError(f'the read ended at {a0 + got}, the bytes at {offset + nbytes}')
            got += n
    finally:
        view.release()
    return addr + offset - a0


def scratch(slot, nbytes):
    """The address of the thread's buffer `slot` of at least nbytes (not read into)."""
    buffers = _local.__dict__.setdefault('buffers', {})
    if slot not in buffers or len(buffers[slot][0]) < nbytes:
        buf = mmap.mmap(-1, -(-nbytes // ALIGN) * ALIGN)
        buffers[slot] = buf, C.addressof(C.c_char.from_buffer(buf))
    return buffers[slot][1]


class Layer:
    """One layer's records (manifest) and the checkpoint tensors they hold (KEY.ranges)."""

    def __init__(self, layer_id, noted, entry):
        self.id, self.sizes = layer_id, noted['sizes']
        self.part = sum(self.sizes)
        self.at = [sum(self.sizes[:k]) for k in range(len(self.sizes))]   # a piece's place in a node part
        self.nodes = entry['record'] // self.part
        self.units = entry.get('units', 1)   # XTIER_COLD_UNITS of the store that wrote them
        if self.nodes * self.part != entry['record'] or self.nodes * self.units != sum(entry['parts']):
            stop(f'layer {layer_id}: {self.part}-byte node parts do not make its {entry["record"]}-byte records')
        self.unit_bytes = [sum(b for _, _, b in u) for u in cold._unit_pieces(self.sizes, self.units)]
        self.where = cold.where_records(entry.get('map') or cold.hand_out(entry['parts']), self.unit_bytes)
        self.starts, self.stored = entry['starts'], entry['stored']
        self.tensors = {}   # expert -> [(piece, file number, offset, bytes)]
        for e, piece, f, offset, nbytes in noted['reads']:
            if nbytes != self.nodes * self.sizes[piece]:
                stop(f'layer {layer_id} expert {e} piece {piece}: {nbytes} bytes in the checkpoint, not {self.nodes} x '
                     f'{self.sizes[piece]} as in the records')
            self.tensors.setdefault(e, []).append((piece, f, offset, nbytes))
        if sorted(self.tensors) != sorted(self.stored) or any(sorted(t[0] for t in v) != list(range(6))
                                                              for v in self.tensors.values()):
            stop(f'layer {layer_id}: the noted tensors are not the six pieces of each stored expert')

    def parts(self, i, fds):
        """The addresses of the i-th stored record's node parts, read (a part in units: its units one after another,
        which is the part's own layout)."""
        if self.units == 1:
            return [read(fds[d], offset, self.part, n)
                    for n, (d, offset) in enumerate(self.where(self.starts, i, n) for n in range(self.nodes))]
        out = []
        for n in range(self.nodes):
            whole, at = scratch(('part', n), self.part), 0
            for t, nbytes in enumerate(self.unit_bytes):
                d, offset = self.where(self.starts, i, n * self.units + t)
                C.memmove(whole + at, read(fds[d], offset, nbytes, (n, t)), nbytes)
                at += nbytes
            out.append(whole)
        return out


class Store:
    """A store's manifest and ranges (first directory), its record files, the checkpoint files of its noted tensors."""

    def __init__(self, model, paths, key, only=None):
        self.model, self.paths = model, paths
        first = paths[0]
        if key is None:
            found = sorted(first.glob(f'{model.name}-*.ranges'))
            if len(found) != 1:
                stop(f'{len(found)} stores of {model.name} noted in {first}: --key picks one')
            key = found[0].name[:-len('.ranges')]
        self.key = key
        try:
            manifest = json.loads((first / f'{key}.json').read_text())
            ranges = json.loads((first / f'{key}.ranges').read_text())
        except (OSError, ValueError) as e:
            stop(f'store {key}: {e} (a start with XTIER_COLD=1 writes it)')
        self.source, self.names = ranges['source'], ranges['files']
        self.layers, self.entries = [], []
        for layer_id, noted in sorted(ranges['layers'].items(), key=lambda kv: int(kv[0])):
            if only is not None and layer_id not in only:
                continue
            entry = manifest['layers'].get(layer_id)
            if entry is None:
                stop(f'layer {layer_id}: noted but not in the manifest')
            self.layers.append(Layer(layer_id, noted, entry))
            self.entries.append(entry)
        if not self.layers:
            stop(f'store {key}: no layer to work on')
        self.files = [model / n for n in self.names]
        for p in self.record_files():
            if not p.exists():
                stop(f'{p} is missing')

    def record_files(self):
        return [d / f'{self.key}.cold' for d in self.paths]

    def open_records(self):
        return [os.open(p, os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC) for p in self.record_files()]

    def open_checkpoint(self, flags):
        return [os.open(p, flags | os.O_CLOEXEC) for p in self.files]

    def every_expert(self, what, fn, threads):
        t0, done, total = time.time(), 0, sum(len(L.stored) for L in self.layers)
        with ThreadPoolExecutor(threads) as pool:
            for L in self.layers:
                for future in [pool.submit(fn, L, i, e) for i, e in enumerate(L.stored)]:
                    future.result()
                done += len(L.stored)
                print(f'{what}: layer {L.id}, {done} of {total} experts, {time.time() - t0:.0f} s', flush=True)

    def compare(self, threads):
        records, tensors = self.open_records(), self.open_checkpoint(os.O_RDONLY | os.O_DIRECT)

        def one(L, i, e):
            parts = L.parts(i, records)
            for piece, f, offset, nbytes in L.tensors[e]:
                t, size = read(tensors[f], offset, nbytes, 'tensor'), L.sizes[piece]
                for n, at in enumerate(parts):
                    if libc.memcmp(at + L.at[piece], t + n * size, size):
                        raise ValueError(f'layer {L.id} expert {e} piece {piece} node part {n}: the record differs '
                                         f'from {self.names[f]}')

        try:
            self.every_expert('compared', one, threads)
        except (ValueError, OSError) as e:
            stop(f'{e}: nothing changed')
        finally:
            for fd in records + tensors:
                os.close(fd)

    def write_back(self, threads):
        records, tensors = self.open_records(), self.open_checkpoint(os.O_WRONLY)

        def one(L, i, e):
            parts = L.parts(i, records)
            for piece, f, offset, _ in L.tensors[e]:
                size = L.sizes[piece]
                for n, at in enumerate(parts):
                    if os.pwrite(tensors[f], (C.c_char * size).from_address(at + L.at[piece]), offset + n * size) != size:
                        raise OSError(f'a short write to {self.names[f]}')

        try:
            self.every_expert('written back', one, threads)
            for fd in tensors:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            for fd in records + tensors:
                os.close(fd)

    def spans(self):
        """Per checkpoint file, the merged byte ranges of the noted tensors."""
        ranges = [[] for _ in self.files]
        for L in self.layers:
            for tensors in L.tensors.values():
                for _, f, offset, nbytes in tensors:
                    ranges[f].append((offset, offset + nbytes))
        merged = []
        for spans in ranges:
            out = []
            for lo, hi in sorted(spans):
                if out and lo <= out[-1][1]:
                    out[-1][1] = max(out[-1][1], hi)
                else:
                    out.append([lo, hi])
            merged.append(out)
        return merged


def write_marker(model, marker):
    tmp = model / (cold.MARKER + '.tmp')
    with open(tmp, 'w') as f:
        f.write(json.dumps(marker))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, model / cold.MARKER)
    fd = os.open(model, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def punch(model, paths, key, dry_run, threads):
    if (model / cold.MARKER).exists():
        stop(f'{model} is already punched ({cold.MARKER}); --restore writes the experts back')
    st = Store(model, paths, key)
    if st.source != cold.checkpoint_id(model):
        stop('the files changed since the records were noted: a start with XTIER_COLD=1 checks them again')
    for L, entry in zip(st.layers, st.entries):
        if entry.get('source') != st.source:
            stop(f'layer {L.id}: its records come from another state of the checkpoint')
    for p in st.files:
        if p.stat().st_nlink != 1:   # the other name would lose the bytes too
            stop(f'{p} has another hard link')
        if not os.access(p, os.W_OK):
            stop(f'{p} is not writable')
    for p in st.record_files():
        fd = os.open(p, os.O_RDONLY)
        os.fdatasync(fd)
        os.close(fd)
    st.compare(threads)
    spans = st.spans()
    nbytes = sum(hi - lo for file in spans for lo, hi in file)
    print(f'{len(st.layers)} layers, {nbytes / 2 ** 30:.1f} GiB of expert tensors in {len(st.files)} files match the '
          f'records', flush=True)
    if dry_run:
        return
    stats = [p.stat() for p in st.files]
    write_marker(model, dict(key=st.key, paths=[str(d) for d in paths], layers=[L.id for L in st.layers],
                             files={n: [s.st_size, s.st_mtime_ns] for n, s in zip(st.names, stats)},
                             date=time.strftime('%Y-%m-%d %H:%M:%S')))   # first: a crash below leaves the guard on
    freed = 0
    for p, s, file in zip(st.files, stats, spans):
        fd = os.open(p, os.O_RDWR | os.O_CLOEXEC)
        try:
            for lo, hi in file:
                a0, a1 = -(-lo // ALIGN) * ALIGN, hi // ALIGN * ALIGN   # whole blocks inside the tensors
                if a1 > a0 and libc.fallocate(fd, PUNCH_HOLE, a0, a1 - a0):
                    stop(f'{p}: {os.strerror(C.get_errno())}: holes cannot be punched here; --restore undoes the rest')
            os.fsync(fd)
        finally:
            os.close(fd)
        os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns))
        freed += (s.st_blocks - p.stat().st_blocks) * 512
    if cold.checkpoint_id(model) != st.source:
        stop('the checkpoint id changed')
    print(f'punched: {freed / 2 ** 30:.1f} GiB freed, {len(st.layers)} layers now load from {st.key} only', flush=True)


def restore(model, paths, key, threads):
    if not (model / cold.MARKER).exists():
        stop(f'{model} is not punched')
    marker = json.loads((model / cold.MARKER).read_text())
    st = Store(model, paths or [Path(d) for d in marker['paths']], key or marker['key'], only=set(marker['layers']))
    stats = [p.stat() for p in st.files]
    for p, s, name in zip(st.files, stats, st.names):
        if s.st_size != marker['files'][name][0]:
            stop(f'{p} changed size since it was punched')
    need = sum(hi - lo for file in st.spans() for lo, hi in file)
    if shutil.disk_usage(model).free < need:
        stop(f'{need / 2 ** 30:.1f} GiB needed on the drive of {model}')
    st.write_back(threads)
    st.compare(threads)
    for p, name in zip(st.files, st.names):
        os.utime(p, ns=(p.stat().st_atime_ns, marker['files'][name][1]))
    (model / cold.MARKER).unlink()
    print(f'restored: {need / 2 ** 30:.1f} GiB written back and compared', flush=True)


def main():
    ap = argparse.ArgumentParser(description='Take the experts out of a checkpoint the cold store holds, or back in.')
    ap.add_argument('model', type=Path)
    ap.add_argument('--paths', default=os.environ.get('XTIER_COLD_PATHS', ''), help="the store's directories, as XTIER_COLD_PATHS")
    ap.add_argument('--key', help='the store, if the first directory holds several of this model')
    ap.add_argument('--dry-run', action='store_true', help='compare only')
    ap.add_argument('--restore', action='store_true', help='write the experts back from the records')
    ap.add_argument('--threads', type=int, default=16)
    args = ap.parse_args()
    model = args.model.resolve()
    paths = [Path(p) for p in args.paths.split(':') if p]
    if args.restore:
        restore(model, paths, args.key, args.threads)
    elif not paths:
        stop('--paths: the directories of the store')
    else:
        punch(model, paths, args.key, args.dry_run, args.threads)


if __name__ == '__main__':
    main()
