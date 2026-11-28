"""Checks of the elastic expert cache's learnt needs (xtier/cache/elastic.py Governor.learn / predict), CPU only: a
long session's large needs do not raise a short prompt's (2026-10-26: after a 490K-token session every new session's
first prefill released all extra slots); a need seen bounds smaller forwards and, scaled by the size and context
ratios, larger ones up to 2x; forwards nothing bounds, unknown contexts and the draft worker's apart predict None.

    python tests/elastic_need_check.py        (from the repository root)
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xtier.cache import elastic  # noqa: E402

GIB = 2 ** 30
g = object.__new__(elastic.Governor)   # the learnt needs only: no pools, no GPU
g.seen, g.recent, g.small, g.margin, g.extrapolate, g.debug = {}, [], 4096, 1.25, 2.0, False
grid = elastic.Governor._grid

# the console's warm-up (short prompts), then a session grown to about 490K tokens (GLM-5.3, 2026-10-26)
for tokens, ctx, need in [(340, 340, 0.3), (2500, 2500, 0.8), (6800, 6800, 1.5),
                          (1044, 488000, 5.0), (4400, 490000, 6.9), (4602, 450000, 7.0)]:
    g.learn(tokens, False, need * GIB, ctx)


def predict(tokens, ctx, draft=False):
    n = g.predict(tokens, draft, ctx)
    return None if n is None else n / GIB


assert grid(6800) <= 6800 < grid(6800) * 2 ** (1 / 8) and grid(4096) == 4096
# a new session's first prefill: the warm-up's 6.8K prompt scaled up, not the long session's 7 GiB
short = predict(6700, 6700)
assert abs(short - 1.5 * (6700 / grid(6800)) ** 2 * 1.25) < 1e-9, short
long = predict(4400, 495000)
assert long >= 6.9 * 1.25 and short < 0.3 * long, (short, long)
# a need bounds every forward no larger in size and context, unscaled
assert abs(predict(4400, 200000) - 7.0 * 1.25) < 1e-9 and abs(predict(300, 300) - 0.3 * 1.25) < 1e-9
# the tightest bound counts: the long session's needs bound a short prompt too, but the warm-up's are lower
assert abs(predict(2400, 2400) - 0.8 * 1.25) < 1e-9
# beyond 2x of every need seen (size times context), unknown contexts, the draft worker: nothing bounds it
assert predict(20000, 20000) is None and predict(340, None) is None and predict(340, 340, draft=True) is None
assert predict(6800 * 1.5, 6800 * 1.5) is None   # 2.6x the warm-up's prompt on its grid key, 2.3x the long turns' size
# the small forwards' needs for the growth buffer
assert g.recent == [0.3 * GIB, 0.8 * GIB, 5.0 * GIB]
# at most 1024 sizes and contexts are kept, the oldest go
rng = random.Random(0)
for _ in range(5000):
    g.learn(rng.randint(1, 16384), True, GIB, rng.randint(1, 524288))
assert len(g.seen[True]) == 1024
# warm-allocator chunk 2 must not undercut cold-allocator chunk 1
g2 = object.__new__(elastic.Governor)
g2.seen, g2.recent, g2.small, g2.margin, g2.extrapolate, g2.debug = {}, [], 4096, 1.25, 2.0, False
g2.learn(49152, False, 16.92 * GIB, 49152)
g2.learn(49152, False, 10.29 * GIB, 98304)
g2.learn(43444, False, 9.04 * GIB, 141748)   # warm tail chunk smaller in tokens must not undercut cold 48K
assert g2.predict(49152, False, 49152) >= 16.92 * GIB * 1.25 - 1
assert g2.predict(49152, False, 98304) >= 16.92 * GIB * 1.25 - 1

# torch_guard: compute cap = (total - non_torch_base - guard) / total, skip when disabled or too tight
import torch
_orig_info = torch.cuda.mem_get_info
_orig_res = torch.cuda.memory_reserved
_orig_set = torch.cuda.set_per_process_memory_fraction
try:
    calls = []
    torch.cuda.mem_get_info = lambda dev=0: (20 * GIB, 71 * GIB)
    torch.cuda.memory_reserved = lambda dev=0: 18 * GIB
    torch.cuda.set_per_process_memory_fraction = lambda f, d=0: calls.append((f, d))
    g2 = elastic.Governor.__new__(elastic.Governor)
    g2.device = 0
    g2.extra = 0
    g2.pools = []
    g2.row = []
    g2.torch_guard = 0.0
    g2._guard_frac = 1.0
    g2.debug = False
    g2.apply_torch_guard()
    assert calls == [], f'guard disabled should not set fraction: {calls}'
    g2.torch_guard = 1.6 * GIB
    g2.apply_torch_guard()
    # non_torch_base = (71 - 20) - 18 = 33 GiB; budget = 71 - 33 - 1.6 = 36.4 GiB -> frac = 36.4 / 71
    assert len(calls) == 1 and abs(calls[0][0] - 36.4 / 71.0) < 1e-6, calls
    # Calling again with same state is a no-op (<0.002 hysteresis)
    g2.apply_torch_guard()
    assert len(calls) == 1
finally:
    torch.cuda.mem_get_info = _orig_info
    torch.cuda.memory_reserved = _orig_res
    torch.cuda.set_per_process_memory_fraction = _orig_set

print('elastic needs ok: tightest bound from the needs seen, short prompts unaffected by a long session')
