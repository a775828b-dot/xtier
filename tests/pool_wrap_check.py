"""The commit kernel of the hot expert pool (xtier/cache/pool.py _commit) when the swap sequence numbers wrap, at 2**31 (the
kernel's int32) and at 2**32 (the host's uint32), on the GPU without a model. Each case runs the windows of a few swaps,
written to the ring as csrc/mailbox/mailbox.cpp post() does:
- a first window: every posted swap leaves the table, nothing is published, `applied` follows `posted`;
- a window after the first copy landed: that expert enters the table, the others wait;
- a window after all copies landed: every expert is in the table, owner rows updated;
- a slot released by the elastic cache (no new expert, landed at once): out of the table in its window, nothing published.
A wrong index faults the GPU, so a failure can also be a CUDA error.

  python tests/pool_wrap_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from xtier.backend.kernels import Pointer  # noqa: E402
from xtier.cache.pool import RING, _commit  # noqa: E402

E = 32
bad = 0


def i32(x):
    return ((x + 2**31) % 2**32) - 2**31


def check(name, got, want):
    global bad
    if got != want:
        print(f'  {name}: {got}, expected {want}')
        bad += 1


def case(first, count):
    ring = torch.zeros(RING * 4, dtype=torch.int32, device='cuda')
    seqs = torch.zeros(64, dtype=torch.int32, device='cuda')
    state = torch.tensor([i32(first), i32(first)], dtype=torch.int32, device='cuda')
    # slot s holds expert s; swap j replaces expert j by expert E / 2 + j in slot j; one more slot is released
    table = torch.full((E,), -1, dtype=torch.int32, device='cuda')
    table[:count + 1] = torch.arange(count + 1, dtype=torch.int32, device='cuda')
    owner = torch.arange(E, dtype=torch.int32, device='cuda')
    for j in range(count):
        at = ((first + 1 + j) % 2**32 % RING) * 4
        ring[at:at + 4] = torch.tensor([0, E // 2 + j, j, j], dtype=torch.int32, device='cuda')

    def window(posted, landed):
        seqs[0], seqs[32] = i32(posted), i32(landed)
        _commit[(1,)](Pointer(ring.data_ptr(), torch.int32), Pointer(seqs.data_ptr(), torch.int32), state, table, owner,
                      E, RING)
        torch.cuda.synchronize()
        return table.tolist(), owner.tolist(), state.tolist(), seqs[16].item()

    tab, own, st, applied = window(first + count, first)
    check('first window, evicted', [tab[j] for j in range(count)], [-1] * count)
    check('first window, published', [tab[E // 2 + j] for j in range(count)], [-1] * count)
    check('first window, the others', tab[count], count)
    check('first window, state', st, [i32(first + count), i32(first)])
    check('first window, applied', applied, i32(first + count))
    tab, own, st, applied = window(first + count, first + 1)
    check('one landed, published', [tab[E // 2 + j] for j in range(count)], [0] + [-1] * (count - 1))
    check('one landed, state', st, [i32(first + count), i32(first + 1)])
    tab, own, st, applied = window(first + count, first + count)
    check('all landed, published', [tab[E // 2 + j] for j in range(count)], list(range(count)))
    check('all landed, owners', own[:count], [E // 2 + j for j in range(count)])
    check('all landed, state', st, [i32(first + count), i32(first + count)])
    # a slot released: its command lands at once, before the kernel evicted it
    seq = first + count + 1
    at = (seq % 2**32 % RING) * 4
    ring[at:at + 4] = torch.tensor([0, -1, count, count], dtype=torch.int32, device='cuda')
    tab, own, st, applied = window(seq, seq)
    check('released, evicted', tab[count], -1)
    check('released, state', st, [i32(seq), i32(seq - 1)])
    tab, own, st, applied = window(seq, seq)
    check('released, next window', (own[count], st), (-1, [i32(seq), i32(seq)]))
    check('released, the published ones stay', [tab[E // 2 + j] for j in range(count)], list(range(count)))


for first, count in ((0, 3), (1000, 3), (RING - 2, 4), (2**31 - 10, 3), (2**31 - 2, 4), (2**31 - 1, 3), (2**31, 3),
                     (2**31 + 100, 3), (2**32 - 2, 4), (2**32 - 1, 3)):
    before = bad
    case(first, count)
    print(f'first {first}, {count} swaps: {"ok" if bad == before else "FAIL"}', flush=True)
print('POOL-WRAP-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
sys.exit(1 if bad else 0)
