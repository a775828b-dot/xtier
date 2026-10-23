"""Tensor-parallel slices on the CPU engine: one real MoE layer split into tp slices of the intermediate size (what
SGLang's loader gives each TP rank: rows of w1 / w3 and their scales, columns of w2 and its scales), each slice
computed in its own process on its own NUMA nodes (xtier.engine.placement.auto_nodes, as each rank would place
itself), the partial outputs summed, against the whole layer computed in one process on every node with the same
kernels. Only the order of the FP32 Down sums differs, so the check is a relative error bound, not bitwise.
--isa avx2 runs both on the AVX2 build (Zen2) and also reports the AVX2 whole layer against the default build's.

  python tests/tp_check.py --preset glm53 --tp 2 [--isa avx2] [--threads 112] [--rank-threads 56]
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

P = argparse.ArgumentParser()
P.add_argument('--preset', default='glm53')
P.add_argument('--tp', type=int, default=2)
P.add_argument('--isa', default='', help='XTIER_ISA of the slice processes (avx2: Zen2 kernels)')
P.add_argument('--threads', type=int, default=112, help='workers of the whole-layer process')
P.add_argument('--rank-threads', type=int, default=56, help='workers per slice process')
P.add_argument('--role', default='main', help=argparse.SUPPRESS)
P.add_argument('--rank', type=int, default=0, help=argparse.SUPPRESS)
P.add_argument('--out', default='', help=argparse.SUPPRESS)
A = P.parse_args()
CASES = (1, 4, 8, 64, 512)


def run_layer():
    """Child: compute the cases on the whole layer or on rank A.rank's slice, save the outputs."""
    import torch

    import layers
    from xtier.engine import CpuPlan

    MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = layers.PRESETS[A.preset]
    for name, value in SWITCHES.items():
        os.environ.setdefault('XTIER_' + name, value)
    G = 32 if FMT == 'mxfp4' else 16
    w13, w2, s13, s2, g13, g2 = layers.load(A.preset)
    threads = A.threads
    if A.role == 'slice':
        n = I // A.tp
        lo, hi = A.rank * n, (A.rank + 1) * n
        w13 = torch.cat([w13[:, lo:hi], w13[:, I + lo:I + hi]], 1).contiguous()
        s13 = torch.cat([s13[:, lo:hi], s13[:, I + lo:I + hi]], 1).contiguous()
        w2 = w2[:, :, lo // 2:hi // 2].contiguous()
        s2 = s2[:, :, lo // G:hi // G].contiguous()
        I, threads = n, A.rank_threads
    plan = CpuPlan(E, threads, [t.data_ptr() for t in (w13, w2, s13, s2, g13, g2)], H, I, LIMIT, FMT)
    g = torch.Generator().manual_seed(20261023)
    outs = []
    for m in CASES:
        x = torch.randn(m, H, generator=g).bfloat16()
        ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
        routing = torch.rand(m, K, generator=g).float()
        out = torch.empty(m, H, dtype=torch.float32)
        plan.forward(m, K, x.data_ptr(), ids.data_ptr(), routing.data_ptr(), out.data_ptr())
        outs.append(out)
    torch.save(outs, A.out)
    print(f'[{A.role} {A.rank}] {plan.nodes()} nodes, XTIER_NODES={os.environ.get("XTIER_NODES", "all")}, '
          f'I={I}, {threads} workers', flush=True)
    plan.close()


def main():
    import torch

    from xtier.engine.placement import _format, auto_nodes, cpu_nodes

    nodes = cpu_nodes()
    tmp = Path(tempfile.mkdtemp(prefix='xtier-tp-'))
    base = [sys.executable, __file__, '--preset', A.preset, '--tp', str(A.tp), '--threads', str(A.threads),
            '--rank-threads', str(A.rank_threads)]
    isa = dict(os.environ, **({'XTIER_ISA': A.isa} if A.isa else {}))
    subprocess.run(base + ['--role', 'whole', '--out', str(tmp / 'whole.pt')], check=True, env=isa)
    if A.isa:
        subprocess.run(base + ['--role', 'whole', '--out', str(tmp / 'default.pt')], check=True)
    procs = []
    for r in range(A.tp):
        env = dict(isa, XTIER_NODES=_format(auto_nodes(r, A.tp, [], nodes)))
        procs.append(subprocess.Popen(base + ['--role', 'slice', '--rank', str(r), '--out', str(tmp / f'r{r}.pt')],
                                      env=env))
    if any(p.wait() for p in procs):
        raise SystemExit('a slice process failed')

    def rel(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-30)), float((a - b).abs().max())

    full = torch.load(tmp / 'whole.pt')
    parts = [torch.load(tmp / f'r{r}.pt') for r in range(A.tp)]
    default = torch.load(tmp / 'default.pt') if A.isa else None
    worst = 0.0
    for i, m in enumerate(CASES):
        err, top = rel(sum(p[i] for p in parts), full[i])
        worst = max(worst, err)
        line = f'm={m:4d}: sum of {A.tp} slices vs whole layer: rel L2 {err:.2e}, max abs {top:.2e}'
        if default is not None:
            line += ' | %s whole vs default build: rel L2 %.2e, max abs %.2e' % (A.isa, *rel(full[i], default[i]))
        print(line, flush=True)
    ok = worst < 1e-5
    print('TP-CHECK', 'PASS' if ok else 'FAIL', f'(worst rel L2 {worst:.2e}, bound 1e-5; {A.isa or "default"} kernels)')


if A.role == 'main':
    main()
else:
    run_layer()
