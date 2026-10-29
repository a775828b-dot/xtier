"""DeepSeek V4.1's 32 x 32 block FP8 projections weight-only below SM89 (xtier/packs/deepseek_v4/w8a16.py) on real
layers against FP64 x @ dequant(W)^T: the decode-sized FP8 einsum and the dequantize + cuBLAS path for larger calls,
with the FP8 decoded in hardware (SM89 and newer) and in software (XTIER_FP8_DECODE=soft, the RTX 30 series path), each
in a child process.

  python tests/ds_w8a16_check.py [--layers 3,20]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--model', default='/data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
P.add_argument('--layers', default='3,20')
P.add_argument('--child', action='store_true')
A = P.parse_args()
NAMES = ('attn.wq_a', 'attn.wq_b', 'attn.wo_b', 'ffn.shared_experts.w1', 'ffn.shared_experts.w2')


def check():
    from xtier.packs.deepseek_v4 import w8a16
    index = json.loads((Path(A.model) / 'model.safetensors.index.json').read_text())['weight_map']
    dev = torch.device('cuda')
    g = torch.Generator(device=dev).manual_seed(20261029)
    worst = 0.0
    for layer in (int(v) for v in A.layers.split(',')):
        for name in NAMES:
            w_name, s_name = f'layers.{layer}.{name}.weight', f'layers.{layer}.{name}.scale'
            if w_name not in index:
                continue
            with safe_open(str(Path(A.model) / index[w_name]), 'pt') as f:
                weight = f.get_tensor(w_name).to(dev)
                scale = f.get_tensor(s_name).to(dev).float()
            n, k = weight.shape
            exact = (weight.double().view(n // 32, 32, k // 32, 32) * scale.double()[:, None, :, None]).view(n, k)
            for rows in (1, 6, 12, 300):
                x = (torch.randn(rows, k, device=dev, generator=g) * 0.5).to(torch.bfloat16)
                out = w8a16.linear(None, x, weight, [32, 32], scale)
                ref = x.double() @ exact.t()
                err = ((out.double() - ref).norm() / ref.norm()).item()
                worst = max(worst, err)
                print(f'  layer {layer} {name} [{n}, {k}] rows {rows:3d}: relative error {err:.2e}')
    return worst


def main():
    if A.child:
        worst = check()
        print(f'WORST {worst:.3e}')
        return
    ok = True
    for mode in ('hw', 'soft'):
        if mode == 'hw' and torch.cuda.get_device_capability() < (8, 9):
            print('XTIER_FP8_DECODE=hw: skipped (no FP8 conversion instructions below SM89)')
            continue
        env = dict(os.environ, XTIER_FP8_DECODE=mode)
        r = subprocess.run([sys.executable, __file__, '--child', '--model', A.model, '--layers', A.layers], env=env,
                           capture_output=True, text=True)
        print(f'XTIER_FP8_DECODE={mode}')
        print(r.stdout.strip())
        if r.returncode:
            print(r.stderr[-2000:])
        worst = float(r.stdout.rsplit('WORST', 1)[1]) if 'WORST' in r.stdout else float('inf')
        ok &= worst < 4e-3   # one BF16 rounding of the output
    print('DS-W8A16-CHECK', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
