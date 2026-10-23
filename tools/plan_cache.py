"""Expert-cache plan from a ranking profile with counts (xtier/planner/budget.py), against a fixed layout.

  python tools/plan_cache.py --profile P [--budget N | --layout FULL_LAYERS:SLOTS] [--full-bonus B ...]

--layout gives a layout as the launchers set it (XTIER_CACHE_FULL list and XTIER_CACHE_SLOTS, e.g.
"0,1,2,5:155"); its slot count is the budget of the plans unless --budget is given. Prints the CPU engine's expert
reads per decode window left by the layout and by the plans (one per --full-bonus value), with their full layers.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xtier.config import cpulist  # noqa: E402
from xtier.planner.budget import cpu_reads, full_layers, plan  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--profile', required=True)
P.add_argument('--layout', help='FULL_LAYERS:SLOTS, e.g. "0,1,2:155" or ":50"')
P.add_argument('--budget', type=int, help='slots in all (default: those of --layout)')
P.add_argument('--full-bonus', type=float, nargs='*', default=[0.0], help='windows a fully cached layer saves')
P.add_argument('--write', help='save the plan of the last --full-bonus as {"slots": {layer: n}} (XTIER_CACHE_PLAN)')
A = P.parse_args()
d = json.load(open(A.profile))
E, windows = int(d['experts']), int(d['windows'])
counts = {int(l): [float(x) for x in c] for l, c in zip(d['layer_ids'], d['counts'])}
print(f'{d.get("model")}: {len(counts)} profiled layers x {E} experts, {windows} windows')
fixed = None
if A.layout:
    full_spec, _, slot_spec = A.layout.partition(':')
    full = set(cpulist(full_spec))
    fixed = {l: (E if l in full else int(slot_spec or 0)) for l in counts}
    print(f'layout: {len(full & set(counts))} full + {sum(1 for l in counts if l not in full)} x {slot_spec} slots, '
          f'{sum(fixed.values())} slots, CPU reads per window {cpu_reads(counts, fixed, windows):.2f}')
budget = A.budget or (sum(fixed.values()) if fixed else 0)
none = {l: 0 for l in counts}
print(f'no cache: CPU reads per window {cpu_reads(counts, none, windows):.2f}; budget {budget} slots')
for bonus in A.full_bonus:
    p = plan(counts, budget, bonus)
    f = full_layers(counts, p)
    partial = sorted(v for l, v in p.items() if v < E)
    print(f'plan (full bonus {bonus:g}): CPU reads per window {cpu_reads(counts, p, windows):.2f}, {len(f)} full '
          f'layers {f}, partial slots min/median/max {partial[0] if partial else "-"}/'
          f'{partial[len(partial) // 2] if partial else "-"}/{partial[-1] if partial else "-"}')
if A.write:
    with open(A.write, 'w') as out:
        json.dump(dict(model=d.get('model'), profile=A.profile, budget=budget, full_bonus=A.full_bonus[-1],
                       slots={str(l): v for l, v in sorted(p.items())}), out, indent=1)
    print('written', A.write)
