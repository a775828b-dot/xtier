"""FP8 KV reads of the Qwen pack's QSA attention where Triton has no float8e4nv (packs/qwen4_exp/qsa_fp4.py, RTX 30
series) at Qwen3.8's QSA shape (2 KV heads x 256): the decode gather (compact and page-aligned zero-filled layouts) and
the prefill packing read from the pool's bytes with the software E4M3 decode must equal SGLang's own kernels
(qwen_sparse_kv_extraction_compact_triton, pack_qsa_prefill_kv) on the FP8 pool, bitwise. Runs the software decode on
any GPU (XTIER_FP8_DECODE=soft is set here); SGLang's kernels need a GPU with FP8 in Triton (SM89 or newer).

  python tests/qsa_fp8_check.py
"""
import os
import sys
from pathlib import Path

os.environ['XTIER_FP8_DECODE'] = 'soft'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from sglang.srt.layers.attention.qsa.sparse_attn import (pack_qsa_prefill_kv,  # noqa: E402
                                                         qwen_sparse_kv_extraction_compact_triton)

from xtier import arch  # noqa: E402
from xtier.packs.qwen4_exp import qsa_fp4  # noqa: E402

HEADS, DIM, ROWS, DEV = 2, 256, 70000, 'cuda'
torch.manual_seed(20261029)
assert not arch.HW_FP8


def fp8(x):
    return x.to(torch.float8_e4m3fn).contiguous()


# K / V of ROWS tokens with per-token magnitudes over 4 decades (E4M3 subnormals included)
k8, v8 = (fp8(torch.randn(ROWS, HEADS, DIM, device=DEV) * 10 ** torch.empty(ROWS, 1, 1, device=DEV).uniform_(-3, 1))
          for _ in range(2))
raw = (k8.view(torch.uint8), v8.view(torch.uint8), None, None, None, None)
ok = True

batch, topk, width = 3, 2048, 120000
req_to_token = torch.randint(0, ROWS, (4, width), dtype=torch.int32, device=DEV)
req_indices = torch.tensor([2, 0, 3], dtype=torch.int64, device=DEV)
seq_lens = torch.tensor([50000, 3000, 69000], dtype=torch.int32, device=DEV)
indices = torch.full((batch, topk), -1, dtype=torch.int32, device=DEV)
counts = []
for b, n in enumerate((2048, 1500, 2000)):
    indices[b, :n] = torch.randperm(int(seq_lens[b]), device=DEV)[:n].to(torch.int32)
    counts.append(n)
cu_k = torch.tensor([0] + counts, device=DEV).cumsum(0).to(torch.int32)

for label, cu, zero_fill in (('compact', cu_k, 0), ('page-aligned', torch.arange(batch + 1, device=DEV, dtype=torch.int32) * 2112, 2112)):
    cap = int(cu[-1]) if not zero_fill else batch * zero_fill
    outs = []
    for fn in ('sglang', 'xtier'):
        out_k = torch.full((cap, HEADS, DIM), float('nan'), dtype=torch.bfloat16, device=DEV)
        out_v = out_k.clone()
        args = (req_to_token, req_indices, indices, seq_lens, cu, out_k, out_v, batch, topk)
        if fn == 'sglang':
            qwen_sparse_kv_extraction_compact_triton(k8, v8, *args, zero_fill_cols=zero_fill)
        else:
            qsa_fp4.gather(raw, *args, zero_fill_cols=zero_fill)
        outs.append((out_k, out_v))
    same = all(torch.equal(a.nan_to_num(7.0).view(torch.int16), b.nan_to_num(7.0).view(torch.int16))
               for a, b in zip(*outs))
    ok &= same
    print(f'decode gather, {label}: software E4M3 identical to SGLang on the FP8 pool: {same}')

p_req = torch.tensor([1, 3], dtype=torch.int64, device=DEV)
p_lens = [37000, 64000]
p_cu = torch.tensor([0] + p_lens, device=DEV).cumsum(0).to(torch.int32)
ref = pack_qsa_prefill_kv(k=k8, v=v8, req_to_token=req_to_token, req_indices=p_req, cu_k=p_cu,
                          total_k=sum(p_lens), max_k=max(p_lens), output_dtype=torch.bfloat16)
got = qsa_fp4.pack(raw, req_to_token, p_req, p_cu, sum(p_lens), max(p_lens), torch.bfloat16)
same = torch.equal(ref[0].view(torch.int16), got[0].view(torch.int16)) and torch.equal(ref[1].view(torch.int16),
                                                                                       got[1].view(torch.int16))
ok &= same
print(f'prefill packing of {p_lens} tokens: software E4M3 identical to SGLang on the FP8 pool: {same}')
print('QSA-FP8-CHECK', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
