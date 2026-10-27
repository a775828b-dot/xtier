"""Shards filled without the engine's copy against shards the engine copied from the loaded tensors, one real layer,
every byte: the expert tensors read straight from the checkpoint files (xtier/backend/direct.py: locate on the
safetensors views, fill), and with a cold store the RAM slots read from it (cold.attach_loaded); then a bad block
scale must be refused.

  python tests/direct_load_check.py [PRESET ...]      (tests/layers.py presets; default ds41_1017)
  with XTIER_COLD_PATHS (the store's drives), XTIER_COLD_FRACTION, XTIER_COLD_SOURCE (the model directory: the store
  key) and the XTIER_CACHE_PROFILE the store was written with, the store's slots of that model's preset as well
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('XTIER_COLD_LOG', '0')
import ctypes as C
import json

import numpy as np
from safetensors import safe_open

import layers
from xtier.backend import direct
from xtier.cache import cold
from xtier.engine import CpuPlan


def same(a, b):
    """Every shard byte of plans a and b equal."""
    for na, nb in zip(a.shards(), b.shards()):
        for (pa, sa), (pb, sb) in zip(na, nb):
            if sa != sb or not np.array_equal(np.ctypeslib.as_array((C.c_uint8 * sa).from_address(pa)),
                                              np.ctypeslib.as_array((C.c_uint8 * sb).from_address(pb))):
                return False
    return True


def reads(preset, layer):
    """The direct reads of the layer's expert tensors, as the loader's hook notes them (files kept open while noted)."""
    MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = layers.PRESETS[preset]
    root = Path(MODEL)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    packed = KEY.format(layer=layer, e=0) + NAMES[0] + '.weight_packed' in index
    wsuf, ssuf = ('.weight_packed' if packed else '.weight'), ('.scale' if FMT == 'mxfp4' else '.weight_scale')
    out, grouped = [], {}
    for e in range(E):
        p = KEY.format(layer=layer, e=e)
        for shard, n in zip(('w1', 'w3', 'w2'), NAMES):
            grouped.setdefault(index[p + n + wsuf], []).append((e, p + n, shard))
    for file, items in grouped.items():
        with safe_open(root / file, framework='pt', device='cpu') as f:
            for e, name, shard in items:
                for bulk, suffix in ((0 if shard != 'w2' else 1, wsuf), (2 if shard != 'w2' else 3, ssuf)):
                    read = direct.locate(bulk, shard, e, f.get_tensor(name + suffix), H, I, E, FMT)
                    assert read is not None, (name + suffix, 'not readable as it is')
                    out.append(read)
    return out


for preset in sys.argv[1:] or ['ds41_1017']:
    MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = layers.PRESETS[preset]
    for name, value in SWITCHES.items():
        os.environ.setdefault('XTIER_' + name, value)
    t0 = time.time()
    inputs = layers.load(preset)
    pointers = [t.data_ptr() for t in inputs]
    empty = [None] * 4 + pointers[4:]
    alpha, beta = layers.SWIGLU.get(preset, (1.0, 0.0))
    kw = dict(swiglu_alpha=alpha, swiglu_beta=beta)
    copied = CpuPlan(E, 112, pointers, H, I, LIMIT, FMT, **kw)
    t1 = time.time()
    jobs = reads(preset, LAYER)
    filled = CpuPlan(E, 112, empty, H, I, LIMIT, FMT, **kw)
    nbytes = direct.fill(filled, jobs, FMT)
    t2 = time.time()
    ok = same(copied, filled)
    print(f'{preset} layer {LAYER}: {len(jobs)} reads, {nbytes / 1e9:.2f} GB in {t2 - t1:.2f} s '
          f'({nbytes / 1e9 / (t2 - t1):.1f} GB/s; loading the tensors and the engine copy took {t1 - t0:.1f} s); '
          f'shards identical {ok}; digests {direct.digest(copied)} {direct.digest(filled)}', flush=True)
    assert ok
    addr, size = filled.shards()[0][2]   # a bad block scale must be refused
    byte = C.c_uint8.from_address(addr)
    old, byte.value = byte.value, 255
    try:
        direct.check_scales(filled, FMT)
        raise AssertionError('a block scale of 255 was accepted')
    except ValueError as e:
        print(f'  refused: {e}')
    byte.value = old
    if os.environ.get('XTIER_COLD_PATHS') and Path(os.environ.get('XTIER_COLD_SOURCE', '')) == Path(MODEL):
        lp = cold.LayerPlan(LAYER, E, 50)
        resident = CpuPlan(E, 112, pointers, H, I, LIMIT, FMT, resident=lp.resident, **kw)
        slots = CpuPlan(E, 112, empty, H, I, LIMIT, FMT, resident=lp.resident, **kw)
        t3 = time.time()
        cold.attach_loaded(slots, lp, FMT)
        direct.finish()   # the slots' reads run in the background until the end of loading
        t4 = time.time()
        ok = same(resident, slots)
        print(f'{preset} layer {LAYER}, cold store: {slots.slots} of {E} slots read in {t4 - t3:.2f} s; slots identical '
              f'{ok}', flush=True)
        assert ok
    del copied, filled
direct.close()
print('DIRECT-LOAD-CHECK PASS')
