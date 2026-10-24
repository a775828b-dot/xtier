"""Planner checks without a GPU: the cost model and the dynamic program on synthetic layers.

  python tests/planner_check.py

- emax (Clark) against a Monte Carlo estimate;
- solve: stays within the budget; an empty budget leaves every layer uncached, a budget of every expert caches all;
  its decode is no worse than the uniform split of the same budget or the hand layout heuristics;
- best: decode grows and upload shrinks along the prefill weight; the decode bound is kept;
- from_env (XTIER_CACHE_PLAN=auto) on temporary profile / probe files.
"""
import json
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xtier.planner import auto, cost  # noqa: E402

bad = []


def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what)
    if not ok:
        bad.append(what)


rng = random.Random(20261024)
# emax against sampling
for ma, va, mb, vb in ((10, 4, 12, 9), (50, 1, 20, 1), (5, 0, 5, 0), (30, 25, 31, 16)):
    n = 200000
    mc = sum(max(rng.gauss(ma, va ** 0.5), rng.gauss(mb, vb ** 0.5)) for _ in range(n)) / n
    check(abs(cost.emax(ma, va, mb, vb) - mc) < 0.05, f'emax({ma},{va},{mb},{vb}) = {cost.emax(ma, va, mb, vb):.3f} '
          f'vs sampled {mc:.3f}')

# synthetic model: 24 layers x 128 experts, Zipf-like usage with per-layer skew, 4 tokens x top-8 per window
E, L, W = 128, 24, 10000
probe = dict(model=dict(E=E, H=2048, I=1024, k=8, format='nvfp4', expert_bytes=3_538_944), threads=112, m=4,
             h2d_gbps=50.0, fit=dict(gpu=[14.0, 3.0], cpu=[50.0, 7.5], part=[6.0, -7.0]))
model = cost.Model.from_probe(probe)
usages, counts = {}, {}
for l in range(L):
    s = 0.6 + 0.8 * rng.random()
    w = [1.0 / (i + 1) ** s for i in range(E)]
    tot = sum(w)
    p = [min(1.0, 20.0 * x / tot) for x in w]   # about 20 distinct experts per window
    counts[l] = [round(x * W) for x in p]
    rng.shuffle(counts[l])
    usages[l] = cost.Usage(counts[l], W)
for budget in (0, 8 * L, 40 * L, 96 * L, E * L):
    plan = auto.solve(model, usages, budget)
    used = sum(plan.values())
    check(used <= budget, f'budget {budget}: plan uses {used} slots')
    uniform = {l: min(E, (budget // L) // 8 * 8) for l in usages}
    check(cost.plan_us(model, usages, plan) <= cost.plan_us(model, usages, uniform) + 1e-6,
          f'budget {budget}: decode {cost.plan_us(model, usages, plan):.1f} us <= uniform '
          f'{cost.plan_us(model, usages, uniform):.1f} us')
check(all(s == 0 for s in auto.solve(model, usages, 0).values()), 'empty budget: no slots')
check(all(s == E for s in auto.solve(model, usages, E * L).values()) or
      cost.plan_us(model, usages, auto.solve(model, usages, E * L)) <= cost.plan_us(model, usages, {l: E for l in usages}),
      'full budget: decode no worse than every layer cached')
prev = None
for lam in auto.LAMS:
    p = auto.solve(model, usages, 64 * L, lam)
    cur = (cost.plan_us(model, usages, p), auto.prefill_us(model, p))
    if prev is not None:
        check(cur[0] >= prev[0] - 1e-6 and cur[1] <= prev[1] + 1e-6, f'lam {lam:g}: decode {cur[0]:.1f} >= '
              f'{prev[0]:.1f}, upload {cur[1]:.0f} <= {prev[1]:.0f}')
    prev = cur
fast = auto.solve(model, usages, 64 * L)
bound = cost.plan_us(model, usages, fast) * 1.02
p, d, u, lam = auto.best(model, usages, 64 * L, bound=bound)
check(d <= bound and u <= auto.prefill_us(model, fast), f'best: decode {d:.1f} <= {bound:.1f}, upload {u:.0f} <= '
      f'{auto.prefill_us(model, fast):.0f} (lam {lam:g})')

# guard / even
ref = {l: 32 for l in usages}
l0 = next(iter(usages))
small = {**ref, l0: 40}
p, kept = auto.guard(model, usages, small, ref, 0.5)
check(kept and p == ref, 'guard: a small predicted gain keeps the reference')
p, kept = auto.guard(model, usages, {**ref, l0: E}, ref, 0.5)
check(not kept, 'guard: a plan with less prefill upload passes')
spread = auto.even(ref, sum(ref.values()) + 50, E)
check(sum(spread.values()) == sum(ref.values()) + 50 and max(spread.values()) - min(spread.values()) <= 1,
      f'even: 50 extra slots spread over {len(ref)} layers ({min(spread.values())}..{max(spread.values())})')

# from_env
with tempfile.TemporaryDirectory() as tmp:
    prof = Path(tmp) / 'profile.json'
    prof.write_text(json.dumps(dict(experts=E, windows=W, layer_ids=list(usages),
                                    layers=[sorted(range(E), key=lambda e: -counts[l][e]) for l in usages],
                                    counts=[counts[l] for l in usages])))
    pr = Path(tmp) / 'probe.json'
    pr.write_text(json.dumps(probe))
    os.environ.update(XTIER_CACHE_PROFILE=str(prof), XTIER_PLANNER_PROBE=str(pr), XTIER_PLANNER_KEEP='1')
    plan = auto.from_env({0, 1, 2}, 32)
    hand = {l: (E if l in (0, 1, 2) else 32) for l in usages}
    check(sum(plan.values()) <= sum(hand.values()), f'from_env: {sum(plan.values())} slots <= hand layout '
          f'{sum(hand.values())}')
    check(cost.plan_us(model, usages, plan) <= cost.plan_us(model, usages, hand) * 1.003 + 1e-6,
          f'from_env (keep): decode {cost.plan_us(model, usages, plan):.1f} <= hand '
          f'{cost.plan_us(model, usages, hand):.1f} us x 1.003')
print('PLANNER-CHECK', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
