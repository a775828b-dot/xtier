"""QSA indexer prefill logits with the keys split over CTAs (packs/qwen4_exp/qsa_prefill.py) against SGLang's
``tilelang_qsa_mqa_prefill`` at Qwen3.8's indexer shape (4 heads x 128, BF16): bitwise equality of the masked logits
for causal row ranges deep in a long prompt, and the time per call.

  python tests/qsa_prefill_check.py
"""
import os
import sys
import time
from pathlib import Path

# TileLang picks its HIP target first when a ROCm toolkit is installed (the server gets xtier.platform's fix)
os.environ.setdefault('TILELANG_DEFAULT_TARGET', '{"kind": "cuda", "arch": "sm_120"}')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from sglang.srt.layers.attention.qsa import mqa  # noqa: E402

from xtier.packs.qwen4_exp import qsa_prefill  # noqa: E402

HEADS, DIM, RATIO = 4, 128, 4
torch.manual_seed(20261023)


def case(rows, prefix):
    """rows query rows at prompt positions prefix .. prefix + rows; compressed keys up to the last one."""
    keys = (prefix + rows) // RATIO
    q = torch.randn(rows, HEADS, DIM, device='cuda').bfloat16()
    k = torch.randn(keys, 1, DIM, device='cuda').bfloat16()
    pos = torch.arange(prefix, prefix + rows, device='cuda')
    starts = torch.zeros(rows, dtype=torch.int32, device='cuda')
    ends = ((pos + 1) // RATIO).to(torch.int32)
    return q, k, starts, ends


def timed(fn, *args):
    fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        out = fn(*args)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) / 3 * 1e3


ok = True
for rows, prefix in ((352, 380000), (1000, 100000), (64, 8192), (4096, 30000), (37, 500000)):
    q, k, starts, ends = case(rows, prefix)
    ref, t_ref = timed(mqa.tilelang_qsa_mqa_prefill, q, k, starts, ends)
    got, t_got = timed(lambda *a: qsa_prefill.qsa_mqa_prefill(mqa.tilelang_qsa_mqa_prefill, *a), q, k, starts, ends)
    same = torch.equal(ref, got)
    ok &= same
    print(f'{rows:5d} rows x {k.shape[0]:6d} keys: identical {same}; {t_ref:7.2f} -> {t_got:6.2f} ms')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
