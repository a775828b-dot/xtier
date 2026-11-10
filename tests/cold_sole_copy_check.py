"""A cold store holding the only copy of a checkpoint's experts (tools/cold_punch.py took them out), started from another
directory with the same store (XTIER_COLD_KEY): a variant of the checkpoint, or a copy of the directory made without its
XTIER-COLD-ONLY marker. Its expert tensors read as zeros. Before 2026-11-10 (audit R2) such a start wrote the records
again, from those zeros. On a small made-up MXFP4 layer, each start in its own process as a server's is:

  A writes the records; the store marked as tools/cold_punch.py marks it   B (zeros, no marker): stops, records kept
  the same store without the mark (punched before the fix)                  B: stops on the zeros, records kept
  A again (its real tensors)                                                the records reused, nothing written

  python tests/cold_sole_copy_check.py DIR      (a directory on a drive with O_DIRECT; removed at the end)
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
H, I, E, FMT, THREADS, LAYER, KEY = 2048, 1024, 16, 'mxfp4', 16, 0, 'shared-store'


def experts(zero):
    import torch
    g = torch.Generator().manual_seed(20261110)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=g)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g)
    s13 = torch.randint(100, 140, (E, 2 * I, H // 32), dtype=torch.uint8, generator=g)
    s2 = torch.randint(100, 140, (E, H, I // 32), dtype=torch.uint8, generator=g)
    if zero:   # a punched checkpoint's expert tensors
        for t in (w13, w2, s13, s2):
            t.zero_()
    return w13, w2, s13, s2, torch.ones(E, 2), torch.ones(E)


def start(root, model, zero):
    """One start's layer, as xtier/backend/method.py takes it: the store if it can, else the checkpoint; then attach."""
    drives = [root / 'd0', root / 'd1']
    os.environ.update(XTIER_COLD='1', XTIER_COLD_PATHS=':'.join(map(str, drives)), XTIER_COLD_KEY=KEY,
                      XTIER_COLD_SOURCE=str(root / model), XTIER_COLD_FRACTION='0.5', XTIER_COLD_LOG='0',
                      XTIER_FAST_LOAD='1', XTIER_PROBE_DISPATCH_CPU='95')
    sys.path.insert(0, str(REPO))
    from xtier.cache import cold
    from xtier.engine import CpuPlan
    inputs = experts(zero)
    lp = cold.LayerPlan(LAYER, E, 4)
    from_store = cold.load_enabled() and cold.loadable(lp, H, I, FMT, THREADS)
    if not from_store:
        cold.require_checkpoint(LAYER)
    plan = CpuPlan(E, THREADS, [t.data_ptr() for t in inputs], hidden=H, inter=I, scale_format=FMT,
                   resident=lp.resident)
    try:
        if from_store:
            return 'read from the store'
        cold.attach(plan, lp, inputs, FMT)
        st = cold.store(cold._key(E, H, I, FMT, plan.nodes()))
        return f'written {st.written}, reused {st.reused}'
    except RuntimeError as e:
        return f'stopped: {e}'


def main():
    root = Path(sys.argv[1]) / 'cold-sole-copy-check'
    if len(sys.argv) > 2:   # one start: DIR MODEL zero|real
        print('RESULT', start(root, sys.argv[2], sys.argv[3] == 'zero'), flush=True)
        os._exit(0)
    shutil.rmtree(root, ignore_errors=True)
    for d in ('model-A', 'model-B', 'd0', 'd1'):
        (root / d).mkdir(parents=True)
    for d, tag in (('model-A', b'A'), ('model-B', b'B')):
        (root / d / 'model-0.safetensors').write_bytes(tag * 4096)
        (root / d / 'config.json').write_text('{}')
    bad = 0

    def check(name, ok, detail=''):
        nonlocal bad
        print(f'{name}: {"ok" if ok else "FAIL"} {detail}'.rstrip(), flush=True)
        bad += not ok

    def run(model, zero):
        r = subprocess.run([sys.executable, __file__, sys.argv[1], model, 'zero' if zero else 'real'],
                           capture_output=True, text=True)
        lines = [x for x in r.stdout.splitlines() if x.startswith('RESULT ')]
        return lines[-1][7:] if lines else f'no result (exit {r.returncode}): {r.stderr[-300:]}'

    def records():
        return [hashlib.sha1((root / d / f'{KEY}.cold').read_bytes()).hexdigest() for d in ('d0', 'd1')]

    got = run('model-A', False)
    check('A writes the records', got.startswith('written 1'), got)
    first = records()
    sys.path.insert(0, str(REPO))
    from xtier.cache import cold
    manifest_path = root / 'd0' / f'{KEY}.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['layers'][str(LAYER)]['only'] = dict(model=str(root / 'model-A'), date='test')   # as cold_punch marks it
    cold.write_json(manifest_path, manifest)
    cold.write_json(cold.only_path(root / 'd0', KEY), dict(layers=[str(LAYER)], model=str(root / 'model-A')))
    got = run('model-B', True)
    check('B (zeros, no marker), the store marked: stops', got.startswith('stopped') and 'only copy' in got, got[:140])
    check('the records are unchanged', records() == first)

    manifest['layers'][str(LAYER)].pop('only')   # a store punched before the mark existed
    cold.write_json(manifest_path, manifest)
    cold.only_path(root / 'd0', KEY).unlink()
    got = run('model-B', True)
    check('B, the store not marked: stops on the zeros', got.startswith('stopped') and 'zeros' in got, got[:140])
    check('the records are unchanged', records() == first)

    got = run('model-A', False)
    check('A again: the records reused', got in ('written 0, reused 1', 'read from the store'), got)
    check('the records are unchanged', records() == first)
    shutil.rmtree(root, ignore_errors=True)
    print('COLD-SOLE-COPY-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
