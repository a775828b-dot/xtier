"""The elastic cache's release (xtier/cache/elastic.py Governor.release) of more extra slots than the command ring takes
in one go (csrc/mailbox xtier_mailbox_pool_shrink: half the ring, 2048 evictions, else nothing at all): many layers
times many extra slots go back in steps (2026-11-10 audit R6: 92 layers x 23 slots refused, the slots kept their memory).
The pool manager is stood in for by a fake that refuses as the real one does; one GPU for torch.cuda.synchronize.

  python tests/elastic_release_check.py
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['XTIER_CACHE_ELASTIC'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from xtier.cache import elastic  # noqa: E402


class FakePools:
    """xtier_mailbox_pool_* as csrc/mailbox has them, every extra slot occupied; vmm calls succeed."""

    def __init__(self, layers, base, extra):
        self.layers, self.base, self.active, self.calls, self.error = layers, base, base + extra, [], b''

    def xtier_mailbox_pool_pause(self, on):
        return 0

    def xtier_mailbox_pool_quiesce(self, timeout_ms):
        return 0

    def xtier_mailbox_pool_shrink(self, extra):
        keep = min(self.active, self.base + max(0, extra))
        n = self.layers * (self.active - keep)
        self.calls.append(n)
        if n > elastic.RING_EVICTIONS:
            self.error = b'Pool shrink: too many evictions for the command ring'
            return -1
        self.active = keep
        return n

    def xtier_mailbox_pool_resume(self):
        return 0

    def xtier_mailbox_vmm_unmap(self, ptr, offset):
        return 0

    def xtier_mailbox_error(self):
        return self.error


bad = 0


def check(name, ok, detail=''):
    global bad
    print(f'{name}: {"ok" if ok else "FAIL"} {detail}'.rstrip(), flush=True)
    bad += not ok


def governor(layers, base, extra):
    pools = [SimpleNamespace(S=base + extra, base=base, active=base + extra,
                             arena=[torch.zeros(base + extra, 64, dtype=torch.uint8) for _ in range(4)])
             for _ in range(layers)]
    lib = FakePools(layers, base, extra)
    g = elastic.Governor(pools, lib, commit=lambda: None)
    g.extra = extra
    return g, lib


for layers, extra, target in ((40, 19, 0), (35, 60, 0), (92, 23, 0), (92, 23, 5), (61, 68, 0)):
    g, lib = governor(layers, 8, extra)
    try:
        n = g.release(target, 'the test')
        why = ''
    except RuntimeError as e:
        n, why = -1, str(e)
    check(f'{layers} layers x {extra} extra slots down to {target}', n == extra - target and lib.active == 8 + target
          and g.extra == target and max(lib.calls) <= elastic.RING_EVICTIONS,
          f'{len(lib.calls)} steps of {lib.calls} evictions{": " + why if why else ""}')
print('ELASTIC-RELEASE-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
sys.exit(1 if bad else 0)
