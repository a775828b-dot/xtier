"""DSpark draft LM head in FP8 (packs/deepseek_v4: _quantize_lm_head, _draft_project) against the BF16 matmul it
replaces: relative L2 error of the logits for DeepSeek V4.1's head shape (FP8 E4M3: about 4%), a CUDA graph replayed
with new activations, and the time per call at the draft's row counts. Top-1 agreement is printed for information
only: random weights and activations give nearly tied logits; what counts is the accept length in the server.

  python tests/draft_lmhead_fp8_check.py [--vocab 129280] [--hidden 5120]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from xtier.packs.deepseek_v4 import _draft_project, _quantize_lm_head  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--vocab', type=int, default=129280)
P.add_argument('--hidden', type=int, default=5120)
A = P.parse_args()
torch.manual_seed(20261023)


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter((torch.randn(A.vocab, A.hidden, device='cuda') * 0.02).bfloat16(),
                                         requires_grad=False)


def bf16(hidden, lm_head):
    return torch.matmul(hidden.to(lm_head.weight.dtype), lm_head.weight.T)


head = Head()
_quantize_lm_head(head)
ok = True
for rows in (1, 5, 10, 64):
    x = torch.randn(rows, A.hidden, device='cuda').bfloat16()
    ref, got = bf16(x, head).float(), _draft_project(bf16, x, head).float()
    err = ((got - ref).norm() / ref.norm()).item()
    top1 = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    ok &= err < 0.05
    print(f'rows {rows:3d}: relative L2 {err:.2e}, top-1 agreement {top1:.2f}')

x = torch.randn(10, A.hidden, device='cuda').bfloat16()
static_in = x.clone()
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    _draft_project(bf16, static_in, head)   # warm
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    static_out = _draft_project(bf16, static_in, head)
for _ in range(3):
    new = torch.randn_like(static_in)
    static_in.copy_(new)
    g.replay()
    direct = _draft_project(bf16, new, head)
    same = torch.equal(static_out, direct)
    ok &= same
    print('graph replay equals a direct call:', same)

for rows in (5, 10):
    x = torch.randn(rows, A.hidden, device='cuda').bfloat16()
    for name, fn in (('bf16', lambda: bf16(x, head)), ('fp8', lambda: _draft_project(bf16, x, head))):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        print(f'rows {rows}: {name} {1e3 * (time.perf_counter() - t0) / 50:.3f} ms')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
