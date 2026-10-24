"""Automatic expert-cache plan (xtier/planner/auto.py) from a routing profile with counts and a probe of this host.

  python tools/plan_auto.py --profile P --probe PROBE.json [--layout FULL_LAYERS:SLOTS] [--budget N | --budget-gib G]
                            [--tol 0.003] [--write PLAN.json] [--layers]

--layout is a hand-made layout as the launchers set it (XTIER_CACHE_FULL list and XTIER_CACHE_SLOTS, e.g.
"0,1,2,5:155"); the plans get its slots on the profiled layers as their budget unless --budget / --budget-gib say
otherwise. Prints the cost model's decode time per window (MoE part) and prefill upload per chunk for the layout, the
fastest-decode plan and the chosen one (decode within --tol of the fastest, least upload). --write saves the chosen
plan as XTIER_CACHE_PLAN reads it ({"slots": {layer: n}}; n = E: every expert cached); layers the profile does not
cover keep XTIER_CACHE_FULL / XTIER_CACHE_SLOTS.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xtier.config import cpulist  # noqa: E402
from xtier.planner import auto, cost  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--profile', required=True, help='LKQstrata profile with "counts" (decode windows per expert)')
P.add_argument('--probe', required=True, help='tools/plan_probe.py output for this model shape and host')
P.add_argument('--layout', help='FULL_LAYERS:SLOTS of the hand-made layout, e.g. "0,1,2:155" or ":50"')
P.add_argument('--budget', type=int, help='slots over the profiled layers')
P.add_argument('--budget-gib', type=float, help='GiB of slots over the profiled layers')
P.add_argument('--add-gib', type=float, default=0.0, help='GiB added to the budget (e.g. memory free at the prefill peak)')
P.add_argument('--tol', type=float, default=0.003, help='decode slack (fraction) traded for fully cached layers')
P.add_argument('--keep-decode', action='store_true',
               help='decode bound = the layout\'s decode (x 1 + tol) instead of the fastest plan\'s')
P.add_argument('--lam', type=float, help='solve for this prefill weight only (0: the fastest decode)')
P.add_argument('--min-gain', type=float, default=0.01,
               help='with --layout: below this predicted decode gain a plan that does not upload less keeps the layout')
P.add_argument('--adapt', type=float, default=1.0,
               help='adaptive residency gain: S slots hit like the profile\'s top adapt x S (cost.Model)')
P.add_argument('--q', type=int, help='slot granularity of the plan (default E / 64)')
P.add_argument('--write', help='save the chosen plan (XTIER_CACHE_PLAN)')
P.add_argument('--layers', action='store_true', help='print every layer of the layout and the plan')
A = P.parse_args()

d = json.load(open(A.profile))
probe = json.load(open(A.probe))
model = cost.Model.from_probe(probe, A.adapt)
E, windows = int(d['experts']), int(d['windows'])
if E != model.E:
    sys.exit(f'profile has {E} experts per layer, the probe {model.E}')
usages = cost.usages(d, model.tokens)
gib = model.slot_bytes() / 2**30
print(f'{d.get("model")}: {len(usages)} profiled layers x {E} experts, {windows} windows'
      f'{" (route counts)" if d.get("routes") else ""}; probe {probe.get("name")} ({probe["threads"]} threads, m '
      f'{probe["m"]}): gpu {model.g0:.1f} + {model.g1:.2f} n, cpu {model.c0:.1f} + {model.c1:.2f} n + {model.c2:.2f} r, '
      f'partial +{model.p0:g} / {model.p1:+g} us; upload {model.upload_us() / 1e3:.1f} ms per layer; adapt {A.adapt:g}')


def describe(name, plan):
    dec = cost.plan_us(model, usages, plan)
    full = sorted(l for l, s in plan.items() if s >= E)
    part = sorted(s for s in plan.values() if 0 < s < E)
    none = sum(1 for s in plan.values() if s <= 0)
    slots = sum(min(s, E) for s in plan.values())
    hit = sum(u.mean[min(plan.get(l, 0), E)] for l, u in usages.items()) / sum(u.mean[-1] for u in usages.values())
    print(f'{name:>12}: decode {dec / 1e3:6.3f} ms per window (MoE), prefill upload {auto.prefill_us(model, plan) / 1e3:6.1f} '
          f'ms per chunk; {slots} slots ({slots * gib:.1f} GiB), {len(full)} full, {len(part)} partly cached '
          f'(slots {part[0] if part else "-"}..{part[-1] if part else "-"}), {none} uncached; hit {hit:.3f}')
    return dec


fixed = None
if A.layout:
    full_spec, _, slot_spec = A.layout.partition(':')
    full = set(cpulist(full_spec))
    fixed = {l: (E if l in full else int(slot_spec or 0)) for l in usages}
budget = A.budget or (int(A.budget_gib / gib) if A.budget_gib else None) or (sum(fixed.values()) if fixed else None)
if budget is None:
    sys.exit('--budget, --budget-gib or --layout')
budget += int(A.add_gib / gib)
print(f'budget {budget} slots ({budget * gib:.1f} GiB) over the profiled layers')
ref = describe('layout', fixed) if fixed else None
fast = auto.solve(model, usages, budget, 0.0, A.q)
describe('fastest', fast)
if A.lam is not None:
    plan, lam = auto.solve(model, usages, budget, A.lam, A.q), A.lam
    dec, pre = cost.plan_us(model, usages, plan), auto.prefill_us(model, plan)
else:
    bound = ref * (1 + A.tol) if A.keep_decode and ref is not None else None
    plan, dec, pre, lam = auto.best(model, usages, budget, A.tol, A.q, bound)
    if bound is not None and pre > auto.prefill_us(model, auto.even(fixed, budget, E)):
        plan = auto.even(fixed, budget, E)   # as XTIER_PLANNER_KEEP: no plan that uploads more than the layout
    if fixed:   # a reshuffle whose predicted gain is within the model's error keeps the hand layout
        plan, kept = auto.guard(model, usages, plan, auto.even(fixed, budget, E), A.min_gain)
        if kept:
            dec, pre = cost.plan_us(model, usages, plan), auto.prefill_us(model, plan)
            print(f'guard: predicted gain below {A.min_gain:g} with no less upload: the hand layout is kept (extra '
                  f'budget spread evenly)')
describe(f'chosen', plan)
print(f'chosen: lam {lam:g}, full layers {sorted(l for l, s in plan.items() if s >= E)}')
if A.layers:
    print('layer  usage  | layout: slots  hit   gpu/cpu us  -> us | plan: slots  hit   gpu/cpu us  -> us')
    for l, u in sorted(usages.items()):
        row = f'{l:5d}  {u.mean[-1]:5.1f}  |'
        for p in ([fixed] if fixed else []) + [plan]:
            s = p[l]
            g, c = model.sides_us(u, s)
            row += (f' {s:5d} {u.hit(s):5.3f} {g:6.1f}/{c:6.1f} -> {model.layer_us(u, s):6.1f} |' if 0 < s < E else
                    f' {s:5d} {u.hit(s):5.3f} {"full" if s >= E else "cpu":>13} -> {model.layer_us(u, s):6.1f} |')
        print(row)
if A.write:
    with open(A.write, 'w') as f:
        json.dump(dict(model=d.get('model'), profile=A.profile, probe=A.probe, budget=budget, tol=A.tol, lam=lam,
                       adapt=A.adapt, decode_us=round(dec, 1), prefill_upload_us=round(pre, 1),
                       slots={str(l): s for l, s in sorted(plan.items())}), f, indent=1)
    print('written', A.write)
