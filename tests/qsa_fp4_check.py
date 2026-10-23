"""NVFP4 KV reads of the Qwen pack's QSA attention (packs/qwen4_exp/qsa_fp4.py) at Qwen3.8's QSA shape (2 KV heads x
256): the decode gather (compact and page-aligned zero-filled layouts) and the prefill packing read from the packed
pool must equal SGLang's own kernels (qwen_sparse_kv_extraction_compact_triton, pack_qsa_prefill_kv) run on the pool
dequantized by SGLang (NVFP4KVQuantizeUtil, the pool's format), bitwise; and the time of the decode gather.

  CUDA_HOME=<nvcc toolkit> python tests/qsa_fp4_check.py   (FlashInfer builds its NVFP4 quantize kernel on first use)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from sglang.srt.layers.attention.qsa.sparse_attn import (pack_qsa_prefill_kv,  # noqa: E402
                                                         qwen_sparse_kv_extraction_compact_triton)
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil  # noqa: E402

from xtier.packs.qwen4_exp import qsa_fp4  # noqa: E402

HEADS, DIM, ROWS, DEV = 2, 256, 70000, 'cuda'
torch.manual_seed(20261023)
one = torch.ones(1, dtype=torch.float32, device=DEV)


def fp4_pool():
    """Packed K / V of ROWS tokens (per-token magnitudes spread over 3 decades) and their SGLang dequantization."""
    out = []
    for _ in range(2):
        x = (torch.randn(ROWS, HEADS, DIM, device=DEV) * 10 ** torch.empty(ROWS, 1, 1, device=DEV).uniform_(-2, 1))
        q, sf, _ = NVFP4KVQuantizeUtil.quantize(x.bfloat16().contiguous(), one)
        out.append((q.view(torch.uint8), sf.view(torch.uint8), NVFP4KVQuantizeUtil.dequantize(q, sf, one)))
    (kq, ksf, kref), (vq, vsf, vref) = out
    return (kq, vq, ksf, vsf, one, one), kref.contiguous(), vref.contiguous()


raw, kref, vref = fp4_pool()
ok = True

# requests: slots of the pool in random order, sequence lengths, top-k positions (valid ones first, then -1)
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
        ok_k = torch.full((cap, HEADS, DIM), float('nan'), dtype=torch.bfloat16, device=DEV)
        ok_v = ok_k.clone()
        args = (req_to_token, req_indices, indices, seq_lens, cu, ok_k, ok_v, batch, topk)
        if fn == 'sglang':
            qwen_sparse_kv_extraction_compact_triton(kref, vref, *args, zero_fill_cols=zero_fill)
        else:
            qsa_fp4.gather(raw, *args, zero_fill_cols=zero_fill)
        outs.append((ok_k, ok_v))
    same = all(torch.equal(a.nan_to_num(7.0), b.nan_to_num(7.0)) for a, b in zip(*outs))
    ok &= same
    print(f'decode gather, {label}: identical to SGLang on the dequantized pool: {same}')

torch.cuda.synchronize()
out_k = torch.empty((batch * 2112, HEADS, DIM), dtype=torch.bfloat16, device=DEV)
out_v = torch.empty_like(out_k)
cu = torch.arange(batch + 1, device=DEV, dtype=torch.int32) * 2112
for fn, label in ((lambda: qwen_sparse_kv_extraction_compact_triton(kref, vref, req_to_token, req_indices, indices,
                                                                    seq_lens, cu, out_k, out_v, batch, topk,
                                                                    zero_fill_cols=2112), 'SGLang, BF16 pool'),
                  (lambda: qsa_fp4.gather(raw, req_to_token, req_indices, indices, seq_lens, cu, out_k, out_v, batch,
                                          topk, 2112), 'xtier, NVFP4 pool')):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        fn()
    torch.cuda.synchronize()
    print(f'decode gather of {batch} x {topk} rows, {label}: {(time.perf_counter() - t0) / 200 * 1e6:.1f} us')

# prefill packing: whole sequences of two requests
p_req = torch.tensor([1, 3], dtype=torch.int64, device=DEV)
p_lens = [37000, 64000]
p_cu = torch.tensor([0] + p_lens, device=DEV).cumsum(0).to(torch.int32)
ref = pack_qsa_prefill_kv(k=kref, v=vref, req_to_token=req_to_token, req_indices=p_req, cu_k=p_cu,
                          total_k=sum(p_lens), max_k=max(p_lens), output_dtype=torch.bfloat16)
got = qsa_fp4.pack(raw, req_to_token, p_req, p_cu, sum(p_lens), max(p_lens), torch.bfloat16)
same = torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
ok &= same
print(f'prefill packing of {p_lens} tokens: identical to SGLang on the dequantized pool: {same}')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
