"""tools/cold_punch.py and the guards around it on a small made-up checkpoint and store (two node parts on two drives):
a compare changes nothing and refuses a record that differs or a file with another hard link; a punch frees the expert
tensors' blocks and nothing else, keeps the checkpoint id; a punched layer can only load from the store, even after a
model .json changed or with another ranking, and an unusable manifest does not delete the records; a restore gives the
files back byte for byte.
    python tests/cold_punch_check.py DIR      (a directory on a drive with holes and O_DIRECT; removed at the end)
"""
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(sys.argv[1]) / 'cold-punch-check'
shutil.rmtree(ROOT, ignore_errors=True)
MODEL, DRIVES = ROOT / 'test-model', [ROOT / 'd0', ROOT / 'd1']
for d in (MODEL, *DRIVES):
    d.mkdir(parents=True)
os.environ.update(XTIER_COLD='1', XTIER_COLD_PATHS=':'.join(map(str, DRIVES)), XTIER_COLD_SOURCE=str(MODEL))
from xtier.cache import cold  # noqa: E402

H, I, E, NODES, FORMAT, THREADS = 2048, 1024, 4, 2, 'mxfp4', 2
KEY = f'test-model-E{E}-H{H}-I{I}-{FORMAT}-n{NODES}'
TOOL = [sys.executable, str(Path(__file__).resolve().parents[1] / 'tools' / 'cold_punch.py'), str(MODEL)]
PATHS = ['--paths', ':'.join(map(str, DRIVES))]
sizes, part, record = cold._layout(H, I, NODES, FORMAT)
rng = random.Random(7)
bad = 0


def check(name, ok, detail=''):
    global bad
    print(f'{"ok  " if ok else "FAIL"} {name} {detail}')
    bad += not ok


def tool(*args):
    r = subprocess.run(TOOL + list(args), capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def sha(p):
    return hashlib.sha1(p.read_bytes()).hexdigest()


def ids():
    return {p.name: sha(p) for p in sorted(MODEL.glob('*.safetensors'))}


# the checkpoint: expert tensors of layers 0 and 1 in two files with other tensors between them, not on block boundaries
tensors = {(lid, e, piece): rng.randbytes(NODES * sizes[piece]) for lid in (0, 1) for e in range(E) for piece in range(6)}
order = list(tensors)
rng.shuffle(order)
reads, others = {0: [], 1: []}, []
for number, chunk in enumerate((order[:len(order) // 2], order[len(order) // 2:])):
    path = MODEL / f'model-{number}.safetensors'
    with open(path, 'wb') as f:
        f.write(rng.randbytes(1234))
        for item in chunk:
            if rng.random() < 0.3:
                f.write(rng.randbytes(rng.randint(1, 9000)))
            lid, e, piece = item
            reads[lid].append((e, piece, str(path.resolve()), f.tell(), len(tensors[item]), None))
            f.write(tensors[item])
        f.write(rng.randbytes(777))
(MODEL / 'config.json').write_text('{}')

# the records, as attach writes them, and the manifest
counts, drive_of = cold._drive_parts(NODES, len(DRIVES))
where = cold.where_records(drive_of, counts, part)
fds = [os.open(d / f'{KEY}.cold', os.O_CREAT | os.O_RDWR) for d in DRIVES]
manifest = {'version': 2, 'layers': {}}
share = [c * part for c in counts]
for number, lid in enumerate((0, 1)):
    starts = [number * E * s for s in share]
    stored = [3, 1, 0, 2]
    for i, e in enumerate(stored):
        for n in range(NODES):
            d, offset = where(starts, i, n)
            body = b''.join(tensors[lid, e, k][n * sizes[k]:(n + 1) * sizes[k]] for k in range(6))
            os.pwrite(fds[d], body, offset)
    manifest['layers'][str(lid)] = dict(record=record, stored=stored, starts=starts, parts=counts,
                                        ends=[s + E * z for s, z in zip(starts, share)], source=cold.source())
for fd in fds:
    os.fsync(fd)
(DRIVES[0] / f'{KEY}.json').write_text(json.dumps(manifest))

# the notes of a start
st = cold.store(KEY)
cold._ranges[0] = (sizes, E, reads[0])
cold._ranges[1] = (sizes, E, reads[1])
cold._ranges[2] = (sizes, E, reads[1][:5])   # incomplete: stays in the checkpoint
cold.save_ranges()
noted = json.loads((DRIVES[0] / f'{KEY}.ranges').read_text())
check('ranges: complete layers only, file names', sorted(noted['layers']) == ['0', '1'] and sorted(noted['files'])
      == ['model-0.safetensors', 'model-1.safetensors'] and not cold._ranges)

before = ids()
times = {p: p.stat().st_mtime_ns for p in MODEL.glob('*.safetensors')}
blocks = sum(p.stat().st_blocks for p in MODEL.glob('*.safetensors'))
ident = cold.checkpoint_id(MODEL)

code, out = tool(*PATHS, '--dry-run')
check('dry run passes and changes nothing', code == 0 and ids() == before and not (MODEL / cold.MARKER).exists(), out[-90:])

byte = os.pread(fds[1], 1, 5000)
os.pwrite(fds[1], bytes([byte[0] ^ 1]), 5000)
os.fsync(fds[1])
code, out = tool(*PATHS)
check('a record that differs: refused, nothing changed', code != 0 and 'differs' in out and ids() == before
      and not (MODEL / cold.MARKER).exists(), out[-90:])
os.pwrite(fds[1], byte, 5000)
os.fsync(fds[1])

link = ROOT / 'link'
os.link(MODEL / 'model-0.safetensors', link)
code, out = tool(*PATHS)
check('a hard link: refused', code != 0 and 'hard link' in out and ids() == before, out[-70:])
link.unlink()

code, out = tool(*PATHS)
check('punch', code == 0 and (MODEL / cold.MARKER).exists(), out[-80:])
freed = (blocks - sum(p.stat().st_blocks for p in MODEL.glob('*.safetensors'))) * 512
total = sum(len(t) for t in tensors.values())
check('blocks freed', total - 2 * 4096 * len(tensors) <= freed <= total, f'{freed} of {total}')
check('sizes, times, id kept', all(p.stat().st_mtime_ns == t for p, t in times.items())
      and cold.checkpoint_id(MODEL) == ident)
# the tensors' whole blocks are gone; the bytes outside them are checked after the restore
punched = {name: (MODEL / name).read_bytes() for name in noted['files']}
wrong = 0
for layer in noted['layers'].values():
    for _, _, f, offset, nbytes in layer['reads']:
        name = noted['files'][f]
        a0, a1 = -(-offset // 4096) * 4096, (offset + nbytes) // 4096 * 4096
        wrong += any(punched[name][a0:a1])
check('the tensors read as zeros', wrong == 0, f'{wrong} tensors not')
code, out = tool(*PATHS)
check('a second punch: refused', code != 0 and 'already punched' in out, out[-70:])

# the guards, as a new process sees them
cold._marker.clear()
cold._source.clear()
cold._nodes = lambda threads: NODES
check('marked layers', cold.only_in_store(0) and cold.only_in_store(1) and not cold.only_in_store(2))
try:
    cold.require_checkpoint(0)
    check('require_checkpoint stops a punched layer', False)
except RuntimeError as e:
    check('require_checkpoint stops a punched layer', 'XTIER_COLD_PATHS' in str(e), str(e)[:70])
cold.require_checkpoint(2)
lp = SimpleNamespace(layer_id=0, experts=E, stored=[0, 1, 2, 3])
check('loadable, another ranking: the records keep their order', cold.loadable(lp, H, I, FORMAT, THREADS)
      and lp.stored == [3, 1, 0, 2], str(lp.stored))
(MODEL / 'config.json').write_text('{"edited": 1}')
cold._source.clear()
lp = SimpleNamespace(layer_id=1, experts=E, stored=[0, 1, 2, 3])
check('loadable after a model .json changed', cold.source() != manifest['layers']['1']['source']
      and cold.loadable(lp, H, I, FORMAT, THREADS))
lp = SimpleNamespace(layer_id=9, experts=E, stored=[0, 1, 2, 3])
check('a layer without records is not loadable', not cold.loadable(lp, H, I, FORMAT, THREADS))
(MODEL / 'config.json').write_text('{}')

text = (DRIVES[0] / f'{KEY}.json').read_text()
(DRIVES[0] / f'{KEY}.json').write_text('not json')
try:
    cold.Store(KEY)
    check('an unusable manifest stops the start', False)
except RuntimeError as e:
    check('an unusable manifest stops the start', 'only copy' in str(e), str(e)[:60])
check('the records are still there', all((d / f'{KEY}.cold').stat().st_size > 0 for d in DRIVES))
(DRIVES[0] / f'{KEY}.json').write_text(text)

stat = (MODEL / 'model-1.safetensors').stat()
os.utime(MODEL / 'model-1.safetensors', ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
cold._marker.clear()
try:
    cold.cold_only()
    check('a changed file: the stale marker stops the start', False)
except RuntimeError as e:
    check('a changed file: the stale marker stops the start', '--restore' in str(e), str(e)[-70:])
os.utime(MODEL / 'model-1.safetensors', ns=(stat.st_atime_ns, stat.st_mtime_ns))

code, out = tool('--restore')
check('restore', code == 0 and not (MODEL / cold.MARKER).exists(), out[-70:])
check('the files are back, byte for byte, with their times', ids() == before
      and all(p.stat().st_mtime_ns == t for p, t in times.items()))
# what the punch left in place was the original too: only bytes inside the noted tensors differ from it
wrong = 0
for name, data in punched.items():
    back = (MODEL / name).read_bytes()
    keep = bytearray(len(back))
    for layer in noted['layers'].values():
        for _, _, f, offset, nbytes in layer['reads']:
            if noted['files'][f] == name:
                keep[offset:offset + nbytes] = b'\1' * nbytes
    wrong += sum(1 for a, b, k in zip(data, back, keep) if a != b and not k)
check('the punch changed nothing outside the tensors', wrong == 0, f'{wrong} bytes')
code, out = tool('--restore')
check('a second restore: refused', code != 0 and 'not punched' in out, out[-50:])

for fd in fds:
    os.close(fd)
del st
shutil.rmtree(ROOT, ignore_errors=True)
print('cold_punch_check', 'PASS' if not bad else f'FAIL {bad}')
sys.exit(0 if not bad else 1)
