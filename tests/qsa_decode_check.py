"""The SM8x QSA decode kernels (xtier/packs/qwen4_exp/qsa_decode.py) against an FP64 reference: Qwen3.8's shape (24
query heads, 2 KV heads, head_dim 256, top-2048 packed rows), rows with 0 .. 2048 keys, eager and in a CUDA graph.
SGLang's Triton decode (the ROCm branch, which rounds the scaled query to BF16) is measured on the same rows for
comparison.

  python tests/qsa_decode_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def reference(q, k, v, cu_k, scale):
    rows, heads, dim = q.shape
    group = heads // k.shape[1]
    out = torch.zeros((rows, heads, dim), dtype=torch.float64)
    qd, kd, vd = q.double().cpu(), k.double().cpu(), v.double().cpu()
    for r in range(rows):
        a, b = int(cu_k[r]), int(cu_k[r + 1])
        if a == b:
            continue
        for h in range(heads):
            s = (kd[a:b, h // group] @ qd[r, h]) * scale
            out[r, h] = torch.softmax(s, 0) @ vd[a:b, h // group]
    return out


def flash_reference(q, k, v, cu_k, scale):
    """The precision of the flash kernels (FA2 / FA4 / trtllm-gen): FP32 scores, P rounded to BF16 for the PV
    product, FP32 sums, one pass with the row max."""
    rows, heads, dim = q.shape
    group = heads // k.shape[1]
    out = torch.zeros((rows, heads, dim), dtype=torch.float64)
    qf, kf, vf = q.float().cpu(), k.float().cpu(), v.float().cpu()
    for r in range(rows):
        a, b = int(cu_k[r]), int(cu_k[r + 1])
        if a == b:
            continue
        for h in range(heads):
            s = (kf[a:b, h // group] @ qf[r, h]) * scale
            p = torch.exp(s - s.max())
            o = (p.to(torch.bfloat16).float() @ vf[a:b, h // group]) / p.sum()
            out[r, h] = o.to(torch.bfloat16).double()
    return out


if __name__ == '__main__':
    import torch

    from xtier.packs.qwen4_exp import qsa_decode
    from sglang.srt.layers.attention.qsa.sparse_attn import sparse_gqa_packed_decode_triton

    dev = torch.device('cuda')
    g = torch.Generator(device='cpu').manual_seed(20261029)
    heads, kv_heads, dim, topk = 24, 2, 256, 2048
    counts = [2048, 1, 0, 37, 128, 129, 1999, 2048]
    rows = len(counts)
    cu_k = torch.zeros(rows + 1, dtype=torch.int32)
    cu_k[1:] = torch.cumsum(torch.tensor(counts), 0)
    cu_q = torch.arange(rows + 1, dtype=torch.int32)
    total = int(cu_k[-1])
    q = (torch.randn((rows, heads, dim), generator=g) * 2).to(torch.bfloat16).to(dev)
    k = (torch.randn((total, kv_heads, dim), generator=g) * 2).to(torch.bfloat16).to(dev)
    v = torch.randn((total, kv_heads, dim), generator=g).to(torch.bfloat16).to(dev)
    scale = dim ** -0.5
    cu_k, cu_q = cu_k.to(dev), cu_q.to(dev)
    want = reference(q, k, v, cu_k.cpu(), scale)

    ok = True
    got = qsa_decode.varlen_decode(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=1,
                                   max_seqlen_k=topk, softmax_scale=scale, causal=True)
    torch.cuda.synchronize()
    err = (got.double().cpu() - want).abs()
    flash = (flash_reference(q, k, v, cu_k.cpu(), scale) - want).abs()
    finite = bool(torch.isfinite(got).all())
    empty_zero = bool((got[2] == 0).all())
    print(f'split-key decode vs FP64: max |err| {err.max().item():.3g}, mean {err.mean().item():.3g}; '
          f'flash precision (BF16 P) vs FP64: max {flash.max().item():.3g}, mean {flash.mean().item():.3g}; '
          f'finite {finite}; empty row zero {empty_zero}')
    # same precision class as the flash kernels: only the order of the sums and where P is rounded differ
    ok &= (err.max() <= 1.5 * flash.max() and err.mean() <= 1.25 * flash.mean()) and finite and empty_zero

    # SGLang's Triton decode on the same packed rows (relative indices as its ROCm branch builds them)
    rel = torch.arange(topk, dtype=torch.int32, device=dev).expand(rows, -1)
    valid = (cu_k[1:] - cu_k[:-1])[:, None]
    rel = rel.masked_fill(rel >= valid, -1).contiguous()
    theirs = sparse_gqa_packed_decode_triton(q.contiguous(), k, v, rel, cu_q, cu_k, valid[:, 0].contiguous(), scale)
    nonempty = torch.tensor([c > 0 for c in counts])
    e2 = (theirs.double().cpu() - want).abs()[nonempty]
    print(f'SGLang Triton decode vs FP64 (non-empty rows): max |err| {e2.max().item():.3g}, '
          f'mean {e2.mean().item():.3g}; ours on the same rows: mean {err[nonempty].mean().item():.3g}')

    # a CUDA graph capture and replay on new values gives the eager result bit for bit
    qs, ks, vs = q.clone(), k.clone(), v.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = qsa_decode.varlen_decode(q=qs, k=ks, v=vs, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=1,
                                       max_seqlen_k=topk, softmax_scale=scale, causal=True)
    q2 = torch.randn_like(q.float()).to(torch.bfloat16)
    qs.copy_(q2)
    graph.replay()
    eager = qsa_decode.varlen_decode(q=q2, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=1,
                                     max_seqlen_k=topk, softmax_scale=scale, causal=True)
    torch.cuda.synchronize()
    same = torch.equal(out.view(torch.int16), eager.view(torch.int16))
    print(f'CUDA graph replay vs eager: {"bitwise identical" if same else "DIFFERENT"}')
    ok &= same
    print('QSA-DECODE-CHECK', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)
