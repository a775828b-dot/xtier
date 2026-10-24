"""Calibrate the planner's cost model against a server run: the adaptive residency gain and the CPU side.

  python tools/plan_calibrate.py --profile P --log SERVER_LOG --layout FULL_LAYERS:SLOTS [--probe PROBE.json]
                                 [--skip N]

The server log of a run with XTIER_CACHE_LOG=N has lines "XTier cache: N windows, hit H, D distinct experts and U us
CPU per layer call" (all partly cached layers together). Compared with the profile's static ranking under the run's
layout this gives the adaptive gain (cost.Model adapt: the S adaptive slots hit like the profile's top adapt x S),
and with a probe, the CPU time the model predicts for the measured CPU experts per call against the measured one
(the engine's time per call, without the mailbox round trip).
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xtier.config import cpulist  # noqa: E402
from xtier.planner import cost  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--profile', required=True)
P.add_argument('--log', required=True)
P.add_argument('--layout', required=True, help='FULL_LAYERS:SLOTS of the run, e.g. ":60"')
P.add_argument('--probe')
P.add_argument('--skip', type=int, default=2, help='first log lines to drop (start-up, adaptation warming up)')
A = P.parse_args()

pat = re.compile(r'XTier cache: (\d+) windows, hit ([\d.]+), ([\d.]+) distinct experts and (\d+) us CPU per layer call')
rows = [tuple(float(x) for x in m.groups()) for m in map(pat.search, open(A.log, errors='ignore')) if m][A.skip:]
if not rows:
    sys.exit('no "XTier cache:" lines (XTIER_CACHE_LOG) after --skip')
hit = statistics.mean(r[1] for r in rows)
distinct = statistics.mean(r[2] for r in rows)
cpu_us = statistics.mean(r[3] for r in rows)
d = json.load(open(A.profile))
E = int(d['experts'])
usages = cost.usages(d)
full_spec, _, slot_spec = A.layout.partition(':')
full = set(cpulist(full_spec))
slots = {l: int(slot_spec or 0) for l in usages if l not in full}
total = sum(usages[l].mean[-1] for l in slots)


def static_hit(a):
    return sum(usages[l].mean[min(E, int(round(a * s)))] for l, s in slots.items()) / total


lo, hi = 0.25, 8.0
if static_hit(hi) < hit:
    lo = hi
for _ in range(40):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if static_hit(mid) < hit else (lo, mid)
adapt = (lo + hi) / 2
print(f'run: {len(rows)} log lines, hit {hit:.3f}, {distinct:.1f} distinct experts and {cpu_us:.0f} us CPU per call; '
      f'profile: {total / len(slots):.1f} distinct experts per window, static hit {static_hit(1.0):.3f} with these slots')
print(f'adaptive gain {adapt:.2f}: the run\'s slots hit like the profile\'s top {adapt:.2f} x slots')
if A.probe:
    model = cost.Model.from_probe(json.load(open(A.probe)))
    nc = distinct * (1 - hit)
    rc = nc * model.tokens / distinct   # routes of the CPU part: the window's average routes per distinct expert
    print(f'CPU side: {nc:.1f} CPU experts per call measured, ~{rc:.0f} routes -> model c1 n + c2 r = '
          f'{model.c1 * nc + model.c2 * rc:.0f} us (+ c0 {model.c0:.0f} us with the mailbox round trip) against '
          f'{cpu_us:.0f} us measured in the engine')
