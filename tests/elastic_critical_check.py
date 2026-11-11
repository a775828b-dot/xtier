"""The elastic cache's low-memory check while decoding (xtier/cache/elastic.py Governor.after_decode): every
XTIER_CACHE_ELASTIC_CHECK decode forwards, free device memory under XTIER_CACHE_ELASTIC_CRITICAL_GIB gives every extra slot
back and stops growth for the cooldown. Memory a prefill left in the allocator's cache is not in use: the check reads the
free memory again after emptying that cache (2026-11-12, DeepSeek V4.1 with a 1M-token KV pool: 0.9 GiB free after a 128K
prompt, held by the cache alone, cost decode its extra slots for 600 s). No GPU: the device's free memory, the allocator's
cache and the release are stood in for.

  python tests/elastic_critical_check.py
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['XTIER_CACHE_ELASTIC'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from xtier.cache import elastic, pool  # noqa: E402

GIB = 2 ** 30
pool.started = lambda: False   # no growth in this test
state = {'free': 0.0, 'cached': 0.0, 'emptied': 0}


def empty_cache():
    state['free'] += state['cached']
    state['cached'] = 0.0
    state['emptied'] += 1


torch.cuda.empty_cache = empty_cache
torch.cuda.mem_get_info = lambda *a: (int(state['free'] * GIB), 72 * GIB)
bad = 0


def check(name, ok, detail=''):
    global bad
    print(f'{name}: {"ok" if ok else "FAIL"} {detail}'.rstrip(), flush=True)
    bad += not ok


def run(free, cached, extra=10):
    """A governor holding `extra` slots, `free` GiB free on the device and `cached` GiB in the allocator's cache, through one
    round of decode forwards: (releases asked for, whether growth is stopped, times the cache was emptied)."""
    pools = [SimpleNamespace(S=69, base=35, active=35 + extra, arena=[torch.zeros(69, 64, dtype=torch.uint8) for _ in range(4)])
             for _ in range(40)]
    g = elastic.Governor(pools, SimpleNamespace(), commit=lambda: None)
    g.extra = extra
    released = []
    g.release = lambda keep, why: released.append((keep, why))
    state.update(free=free, cached=cached, emptied=0)
    for _ in range(g.every):
        g.after_decode()
    return released, g.calm_until > 0, state['emptied']


released, calm, emptied = run(free=5.0, cached=3.0)
check('enough free memory: nothing released, the cache left alone', not released and not calm and emptied == 0)
released, calm, emptied = run(free=0.9, cached=11.5)
check('under the limit only until the allocator\'s cache is emptied: nothing released', not released and not calm and emptied == 1,
      f'({len(released)} releases, cache emptied {emptied} times)')
released, calm, emptied = run(free=0.4, cached=0.3)
check('under the limit with the cache emptied too: every extra slot back, no growth for the cooldown',
      len(released) == 1 and released[0][0] == 0 and calm and emptied == 1, f'({released})')
released, calm, emptied = run(free=0.4, cached=0.3, extra=0)
check('no extra slots held: nothing to release', not released and not calm)
print('ELASTIC-CRITICAL-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
sys.exit(1 if bad else 0)
