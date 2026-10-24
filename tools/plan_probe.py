"""Measure the planner's cost coefficients on this host (xtier/planner/probe.py) for one model's layer shape.

  python tools/plan_probe.py --preset qwen38 [--threads 112] [--m 4] [--slots 256] [--out FILE]
  python tools/plan_probe.py --shape E,H,I,K,FORMAT [...]

Synthetic weights of the shape (no checkpoint needed); the preset (tests/layers.py) also sets the engine switches of
the model's launcher. Takes the GPU and the CPU workers for about a minute: run it with no model server up.
Writes the raw points and the fit to --out (default ~/.cache/xtier/probe-<name>-<threads>t-m<m>.json).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))

P = argparse.ArgumentParser()
P.add_argument('--preset', help='a layer preset of tests/layers.py (shape and engine switches)')
P.add_argument('--shape', help='E,H,I,K,FORMAT instead of a preset, e.g. 512,2560,640,10,nvfp4')
P.add_argument('--threads', type=int, default=112)
P.add_argument('--m', type=int, help='tokens per decode call (default: the preset verify width, else 4)')
P.add_argument('--slots', type=int, help='slots of the partly cached probe layer (default E/2)')
P.add_argument('--calls', type=int, default=32, help='calls per graph, each with other routes')
P.add_argument('--out')
A = P.parse_args()

limit, alpha, beta, m = 0.0, 1.0, 0.0, A.m
if A.preset:
    from layers import PRESETS, SWIGLU
    _, _, _, H, I, E, K, limit, _, vm, fmt, switches = PRESETS[A.preset]
    alpha, beta = SWIGLU.get(A.preset, (1.0, 0.0))
    for name, value in switches.items():
        os.environ.setdefault('XTIER_' + name, value)
    m = m or vm
    name = A.preset
elif A.shape:
    E, H, I, K, fmt = A.shape.split(',')
    E, H, I, K = int(E), int(H), int(I), int(K)
    name = f'{E}x{H}x{I}-{fmt}'
else:
    P.error('--preset or --shape')
m = m or 4
from xtier.planner import probe  # noqa: E402

result = probe.measure(E, H, I, K, fmt, A.threads, m=m, calls=A.calls, slots=A.slots, limit=limit, alpha=alpha,
                       beta=beta)
result['name'] = name
result['switches'] = {k: v for k, v in os.environ.items() if k.startswith('XTIER_')}
out = Path(A.out) if A.out else Path.home() / '.cache' / 'xtier' / f'probe-{name}-{A.threads}t-m{m}.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1) + '\n')
print('written', out)
