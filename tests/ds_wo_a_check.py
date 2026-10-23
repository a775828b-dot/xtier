"""DeepSeek V4.1 wo_a on the Triton grouped einsum (xtier/packs/deepseek_v4/wo_a.py), on the GPU with real layers:
1. a layer's FP8 wo_a and E8M0 scales from the checkpoint, dequantized to BF16 by SGLang's own load-time function,
   goes back to the checkpoint's FP8 bit for bit through the pack's conversion (a stand-in module);
2. the hook's output (FP8 kernel up to 16 tokens, the BF16 kernel for a BF16 weight, dequantized BF16 into SGLang's
   path above) against SGLang's einsum on the loaded BF16 weight, at 1-16 and 64 tokens;
3. the time per call of SGLang's einsum (cuBLAS), the BF16 and the FP8 kernel, in CUDA graphs of 40 calls.
    <SGLang venv>/bin/python tests/ds_wo_a_check.py [--layers 6,14] [--tolerance 1e-3]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from xtier.packs.deepseek_v4 import wo_a  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--model', default='/data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
P.add_argument('--layers', default='6,14')
P.add_argument('--tolerance', type=float, default=1e-3)
P.add_argument('--block-r', type=lambda s: [int(x) for x in s.split(',') if x], default=[],
               help='also time these BLOCK_R tilings')
A = P.parse_args()
from sglang.srt.models.deepseek_v4 import _dequant_fp8  # noqa: E402

index = json.loads((Path(A.model) / 'model.safetensors.index.json').read_text())['weight_map']
G, R = 8, 1024
dev = torch.device('cuda')
failed = []


def tensor(name):
    with safe_open(str(Path(A.model) / index[name]), 'pt') as f:
        return f.get_tensor(name)


def relative(a, b):
    return ((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-30)).item()


def original(o, w, *args, **kwargs):   # SGLang's path off its TP4 fast shapes
    return torch.einsum('tgd,grd->tgr', o, w)


class Attn(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.wo_a = torch.nn.Module()
        self.wo_a.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.o_lora_rank, self.n_local_groups = R, G


def timed(fn, o, weights, *extra, calls=40):
    """us per call in a CUDA graph of `calls` calls cycling over distinct weight copies (out of L2, as 40 layers)."""
    fn(o, weights[0], *(e[0] for e in extra))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            fn(o, weights[i % len(weights)], *(e[i % len(weights)] for e in extra))
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / 10 / calls


for layer in (int(x) for x in A.layers.split(',')):
    w8 = tensor(f'layers.{layer}.attn.wo_a.weight').to(dev)
    s8 = tensor(f'layers.{layer}.attn.wo_a.scale')
    bf16 = _dequant_fp8(w8, s8.to(dev)).to(torch.bfloat16).contiguous()
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(layer + 1)])
    model.layers[layer].self_attn = Attn(bf16.clone())
    wo_a._convert(model, {layer: s8})
    got = model.layers[layer].self_attn.wo_a.weight
    exact = got.dtype == torch.float8_e4m3fn and torch.equal(got.view(torch.uint8), w8.view(torch.uint8))
    print(f'layer {layer}: wo_a {tuple(w8.shape)} scales {tuple(s8.shape)} {s8.dtype}: FP8 back bit for bit: {exact}')
    if not exact:
        failed.append(f'layer {layer} conversion')
        continue
    w = got.view(G, R, -1)
    ref_w = bf16.view(G, R, -1)
    for t in (1, 2, 4, 6, 8, 12, 16, 64):
        o = torch.randn(t, G, w.shape[-1], device=dev, dtype=torch.bfloat16)
        ref = original(o, ref_w)
        e8, e16 = relative(ref, wo_a._apply(original, o, w)), relative(ref, wo_a._apply(original, o, ref_w))
        bitwise = (wo_a._apply(original, o, w) == ref).float().mean().item()
        print(f'  T={t:3d}: FP8 rel {e8:.2e} ({bitwise * 100:.3f}% bitwise), BF16 kernel rel {e16:.2e}')
        if max(e8, e16) > A.tolerance:
            failed.append(f'layer {layer} T={t}')
    copies = 8
    bf16s, fp8s = [ref_w.clone() for _ in range(copies)], [w.clone() for _ in range(copies)]
    scales = [wo_a._scales[got.data_ptr()]] * copies
    for t in (6, 12):
        o = torch.randn(t, G, w.shape[-1], device=dev, dtype=torch.bfloat16)
        us = [timed(original, o, bf16s), timed(wo_a.einsum, o, bf16s), timed(wo_a.einsum, o, fp8s, scales)]
        line = f'  T={t:2d} us/call ({copies} weight copies): cuBLAS einsum {us[0]:.1f}, Triton BF16 {us[1]:.1f}, FP8 {us[2]:.1f}'
        for block_r in A.block_r:
            wo_a.BLOCK_R, keep = block_r, wo_a.BLOCK_R
            line += f'; BLOCK_R {block_r}: BF16 {timed(wo_a.einsum, o, bf16s):.1f}, FP8 {timed(wo_a.einsum, o, fp8s, scales):.1f}'
            wo_a.BLOCK_R = keep
        print(line)
    del bf16s, fp8s
print('FAIL ' + ', '.join(failed) if failed else 'PASS')
sys.exit(1 if failed else 0)
