"""A cold store (xtier/cache/cold.py) in another layout, without the checkpoint: new record files and a new manifest
under another key (other drives, XTIER_COLD_UNITS, XTIER_COLD_PARTS). Every record is read from the old store, written
to the new one, read back and compared byte for byte; the manifest is written only after every layer matched and the
files were flushed. The old store is opened read-only and stays as it is: remove it once a start from the new one has
been checked. A checkpoint whose experts were punched out (tools/cold_punch.py) has no other copy of them, so this is
the way to change its store's layout.

  python tools/cold_relayout.py --key OLD_KEY --paths DIR[:DIR...] --new-key NEW_KEY --new-paths DIR[:DIR...] \
      --units 3 --parts 5:5:5:5:2:2 [--dry-run]

then start with XTIER_COLD_KEY=NEW_KEY XTIER_COLD_PATHS=<the new paths> XTIER_COLD_UNITS=3 XTIER_COLD_PARTS=5:5:5:5:2:2.
The layer shape comes from the old key (-E..-H..-I..-mxfp4|nvfp4-n..) or --hidden / --inter / --nodes / --format.
"""
import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xtier.cache import cold  # noqa: E402
from xtier.engine import ColdStore  # noqa: E402

GIB = 2 ** 30


def stop(message):
    sys.exit(f'cold_relayout: {message}')


def aligned(nbytes):
    """A uint8 tensor of nbytes and the 4 KiB aligned address of its start (O_DIRECT)."""
    t = torch.empty(nbytes + 4096, dtype=torch.uint8)
    skip = -t.data_ptr() % 4096
    return t[skip:skip + nbytes], t.data_ptr() + skip


def fsync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--key', required=True)
    ap.add_argument('--paths', required=True)
    ap.add_argument('--new-key', required=True)
    ap.add_argument('--new-paths', required=True)
    ap.add_argument('--units', type=int, choices=(1, 3), default=3)
    ap.add_argument('--parts', default='', help="the new store's XTIER_COLD_PARTS (units per drive; default even)")
    ap.add_argument('--hidden', type=int)
    ap.add_argument('--inter', type=int)
    ap.add_argument('--nodes', type=int)
    ap.add_argument('--format', choices=('mxfp4', 'nvfp4'))
    ap.add_argument('--batch', type=int, default=64, help='records moved at a time')
    ap.add_argument('--threads', type=int, default=8, help='readers per drive')
    ap.add_argument('--dry-run', action='store_true', help='the plan and the space check only')
    a = ap.parse_args()

    old_dirs = [Path(p) for p in a.paths.split(':') if p]
    new_dirs = [Path(p) for p in a.new_paths.split(':') if p]
    old_manifest = old_dirs[0] / f'{a.key}.json'
    new_manifest = new_dirs[0] / f'{a.new_key}.json'
    if not old_manifest.exists():
        stop(f'no manifest {old_manifest}')
    if new_manifest.exists() or any((d / f'{a.new_key}.cold').exists() for d in new_dirs):
        stop(f'{a.new_key} already has a manifest or records in {a.new_paths}: another key, or remove them first')
    old = json.loads(old_manifest.read_text())
    if old.get('version') != 2 or not old.get('layers'):
        stop(f'{old_manifest}: not a version 2 manifest with layers')

    shape = re.search(r'-E(\d+)-H(\d+)-I(\d+)-(mxfp4|nvfp4)-n(\d+)', a.key)
    H = a.hidden or (shape and int(shape.group(2)))
    I = a.inter or (shape and int(shape.group(3)))
    fmt = a.format or (shape and shape.group(4))
    nodes = a.nodes or (shape and int(shape.group(5)))
    if not (H and I and fmt and nodes):
        stop(f'the key {a.key} does not say the layer shape: --hidden, --inter, --format, --nodes')
    sizes, part, record = cold._layout(H, I, nodes, fmt)

    os.environ['XTIER_COLD_UNITS'] = str(a.units)
    if a.parts:
        os.environ['XTIER_COLD_PARTS'] = a.parts
    else:
        os.environ.pop('XTIER_COLD_PARTS', None)
    units, unit_bytes, counts, drive_of = cold._unit_layout(sizes, nodes, len(new_dirs))
    U = len(units)
    where_new = cold.where_records(drive_of, unit_bytes)
    share = where_new.share
    start_new = [sum(unit_bytes[:t]) for t in range(U)]   # a unit's place in its node part

    layers = sorted(old['layers'], key=int)
    plan, ends = {}, [0] * len(new_dirs)
    for lid in layers:
        entry = old['layers'][lid]
        if entry.get('record') != record:
            stop(f'layer {lid}: {entry.get("record")}-byte records, not {record} as H {H}, I {I}, {fmt}, {nodes} nodes')
        units_old = entry.get('units', 1)
        bytes_old = [sum(b for _, _, b in u) for u in cold._unit_pieces(sizes, units_old)]
        where_old = cold.where_records(entry.get('map') or cold.hand_out(entry['parts']), bytes_old)
        if len(entry['starts']) != len(old_dirs):
            stop(f'layer {lid}: records on {len(entry["starts"])} drives, --paths names {len(old_dirs)}')
        starts = list(ends)
        ends = [s + len(entry['stored']) * z for s, z in zip(starts, share)]
        plan[lid] = (entry, units_old, bytes_old, where_old, starts, list(ends))
    print(f'{len(layers)} layers, {sum(len(old["layers"][l]["stored"]) for l in layers)} records of {record} bytes; '
          f'new layout: {U} unit(s) a node part, units per drive {counts}')
    short = []
    for d, (path, need) in enumerate(zip(new_dirs, ends)):
        free = shutil.disk_usage(path).free
        print(f'  {path}: {need / GIB:.1f} GiB of records ({share[d] / 2 ** 20:.2f} MiB a record), {free / GIB:.1f} GiB free')
        if need + GIB > free:
            short.append(str(path))
    if short:
        stop(f'not enough room on {", ".join(short)} (1 GiB kept free)')
    if a.dry_run:
        return

    src = ColdStore([d / f'{a.key}.cold' for d in old_dirs], a.threads, writable=False)
    dst = ColdStore([d / f'{a.new_key}.cold' for d in new_dirs], a.threads, writable=True)
    B = a.batch
    data, A = aligned(B * record)
    back, Cb = aligned(B * record)
    t0, moved = time.time(), 0
    for lid in layers:
        entry, units_old, bytes_old, where_old, starts, _ = plan[lid]
        start_old = [sum(bytes_old[:t]) for t in range(units_old)]
        stored = entry['stored']
        t1 = time.time()
        for b0 in range(0, len(stored), B):
            b1 = min(len(stored), b0 + B)
            # the old records, each as its node parts one after another (a part's units in order are the part)
            src.read([(*where_old(entry['starts'], i, u),
                       [(A + (i - b0) * record + (u // units_old) * part + start_old[u % units_old],
                         bytes_old[u % units_old])])
                      for i in range(b0, b1) for u in range(nodes * units_old)])
            errors = []

            def write(d):
                try:
                    mine = [u for u in range(nodes * U) if drive_of[u] == d]
                    for i in range(b0, b1):
                        base = A + (i - b0) * record
                        dst.write(d, starts[d] + i * share[d],
                                  [(base + (u // U) * part + start_new[u % U], unit_bytes[u % U]) for u in mine])
                except BaseException as e:   # noqa: BLE001 - re-raised after the join
                    errors.append(e)

            threads = [threading.Thread(target=write, args=(d,)) for d in range(len(new_dirs))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if errors:
                raise errors[0]
            dst.read([(*where_new(starts, i, u),
                       [(Cb + (i - b0) * record + (u // U) * part + start_new[u % U], unit_bytes[u % U])])
                      for i in range(b0, b1) for u in range(nodes * U)])
            n = (b1 - b0) * record
            if not torch.equal(data[:n], back[:n]):
                stop(f'layer {lid} records {b0}..{b1}: what was read back differs from what was written; nothing is '
                     f'noted, {a.new_key} can be removed')
            moved += n
        print(f'layer {lid}: {len(stored)} records moved and compared in {time.time() - t1:.1f} s '
              f'({moved / GIB:.0f} GiB so far, {moved / GIB / (time.time() - t0):.2f} GiB/s)', flush=True)

    for d in new_dirs:
        fsync(d / f'{a.new_key}.cold')
    layers_new = {}
    for lid in layers:
        entry, _, _, _, starts, ends = plan[lid]
        e = {k: v for k, v in entry.items() if k not in ('units', 'map')}
        e.update(starts=starts, ends=ends, parts=counts, **({'units': U, 'map': drive_of} if U > 1 else {}))
        layers_new[lid] = e
    tmp = new_manifest.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        f.write(json.dumps({'version': 2, 'layers': layers_new}))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, new_manifest)
    fsync(new_dirs[0])
    ranges = old_dirs[0] / f'{a.key}.ranges'
    if ranges.exists():
        shutil.copyfile(ranges, new_dirs[0] / f'{a.new_key}.ranges')
        fsync(new_dirs[0] / f'{a.new_key}.ranges')
    print(f'{a.new_key}: {moved / GIB:.1f} GiB in {time.time() - t0:.0f} s, every record compared; manifest '
          f'{new_manifest}. The old store {a.key} is unchanged.')


if __name__ == '__main__':
    main()
