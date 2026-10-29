"""DeepSeek V4.1 sparse MLA below SM90 (xtier.packs.deepseek_v4.sparse_mla) on random V4 KV pages against an FP64
reference, next to SGLang's SM120 paths where this GPU runs them: FlashInfer's SM120 kernel (production) and SGLang's
Triton fallback. Decode-sized calls (split keys, FP32 merge) and prefill-sized ones, window keys only and window plus
compressed keys, holes (-1) and rows without any key.

  python tests/ds_sparse_mla_check.py [--soft]     # --soft: FP8 decoded in software (the RTX 30 series path)
  python tests/ds_sparse_mla_check.py --bench      # time per call of a few tilings (and FlashInfer's on SM120)
"""
import math
import os
import sys
from pathlib import Path

if '--soft' in sys.argv:
    os.environ['XTIER_FP8_DECODE'] = 'soft'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from xtier.packs.deepseek_v4 import sparse_mla  # noqa: E402

HEADS, DIM, NOPE = 64, 512, 448
SCALE = DIM ** -0.5


def pack(latent, page_size):
    """V4 pages of FP32 latents [tokens, 512]: (cache view [pages, page_size, 1, 584], exact decoded keys FP64)."""
    tokens = latent.shape[0]
    pages = tokens // page_size
    nope = latent[:, :NOPE].reshape(tokens, 7, 64)
    exp = torch.ceil(torch.log2(nope.abs().amax(-1).clamp_min(1e-30) / 448)).clamp(-126, 127)
    fp8 = (nope / torch.exp2(exp)[..., None]).to(torch.float8_e4m3fn)
    rope = latent[:, NOPE:].to(torch.bfloat16)
    page_bytes = math.ceil(page_size * 584 / 576) * 576
    buf = torch.zeros(pages, page_bytes, dtype=torch.uint8, device=latent.device)
    data = buf[:, :page_size * 576].view(pages, page_size, 576)
    data[..., :NOPE] = fp8.view(torch.uint8).view(pages, page_size, NOPE)
    data[..., NOPE:] = rope.view(torch.uint8).view(pages, page_size, 128)
    buf[:, page_size * 576:page_size * 584].view(pages, page_size, 8)[..., :7] = (exp + 127).to(torch.uint8).view(
        pages, page_size, 7)
    keys = torch.cat([(fp8.double() * torch.exp2(exp.double())[..., None]).reshape(tokens, NOPE), rope.double()], 1)
    return buf.as_strided((pages, page_size, 1, 584), (page_bytes, 584, 584, 1)), keys


def reference(q, keys1, idx1, len1, keys2, idx2, len2, sink):
    """FP64: softmax over the valid keys of both lists plus the sink; rows without keys 0."""
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    for r0 in range(0, q.shape[0], 32):
        r = slice(r0, r0 + 32)
        ks, valid = [], []
        for keys, idx, length in ((keys1, idx1, len1), (keys2, idx2, len2)):
            if idx is None:
                continue
            pos = torch.arange(idx.shape[1], device=q.device)
            v = (idx[r] >= 0) & (pos[None, :] < length[r, None])
            ks.append(keys[idx[r].clamp_min(0).long()])
            valid.append(v)
        k = torch.cat(ks, 1)
        v = torch.cat(valid, 1)
        s = torch.einsum('rhd,rnd->rhn', q[r].double(), k) * SCALE
        s = s.masked_fill(~v[:, None, :], float('-inf'))
        m = s.amax(-1, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = torch.exp(s - m)
        total = p.sum(-1, keepdim=True) + torch.exp(sink.double()[None, :, None] - m)
        o = torch.einsum('rhn,rnd->rhd', p, k) / total
        out[r] = torch.where(v.any(1)[:, None, None], o, torch.zeros_like(o))
    return out


def errors(out, ref):
    out = out.reshape(ref.shape).double()
    if not torch.isfinite(out).all():
        return float('inf'), float('inf')
    d = out - ref
    return (d.abs().max() / ref.abs().max()).item(), (d.square().mean().sqrt() / ref.square().mean().sqrt()).item()


def bench(c1, c2, c1_64, c2_64, sink, sm120):
    """Time per call with every row's 128 window keys and 512 compressed keys valid, for a few tilings."""
    import triton.runtime.errors
    import triton.testing
    dev = c1.device
    for rows in (1, 12, 4096):
        q = (torch.randn(rows, 1, HEADS, DIM, device=dev) * 2).to(torch.bfloat16)
        idx1 = torch.randint(0, 16384, (rows, 1, 128), device=dev, dtype=torch.int32)
        idx2 = torch.randint(0, 8192, (rows, 1, 512), device=dev, dtype=torch.int32)
        len1 = torch.full((rows,), 128, device=dev, dtype=torch.int32)
        len2 = torch.full((rows,), 512, device=dev, dtype=torch.int32)
        line = []
        configs = os.environ.get('XTIER_BENCH_TILES', '32/32/8,16/16/4,32/16/8,64/16/8')
        for hb, bn, warps in (tuple(int(v) for v in c.split('/')) for c in configs.split(',')):
            sparse_mla.HEADS_PER_PROGRAM, sparse_mla.BLOCK_N, sparse_mla.NUM_WARPS = hb, bn, warps
            try:
                ms = triton.testing.do_bench(lambda: sparse_mla.sparse_mla(q, c1, idx1, len1, sink, SCALE, c2, idx2,
                                                                           len2))
                line.append(f'{hb}/{bn}/{warps}w {ms * 1e3:8.1f}')
            except triton.runtime.errors.OutOfResources:
                line.append(f'{hb}/{bn}/{warps}w  no fit')
        if sm120:
            from sglang.kernels.ops.attention import flash_mla_sm120 as fm
            ms = triton.testing.do_bench(lambda: fm.flash_mla_with_kvcache_sm120(
                q=q, k_cache=c1_64, head_dim_v=DIM, softmax_scale=SCALE, indices=idx1, topk_length=len1,
                attn_sink=sink, extra_k_cache=c2_64, extra_indices_in_kvcache=idx2, extra_topk_length=len2))
            line.append(f'FlashInfer {ms * 1e3:8.1f}')
        print(f'{rows:5d} rows, us per call (heads per program / keys per step / warps): ' + ' | '.join(line))


def main():
    dev = torch.device('cuda')
    g = torch.Generator(device=dev).manual_seed(20261029)
    # the GPU generation, as tools/arch_check.py --emulate shows it to SGLang and XTier
    sm = int(os.environ.get('XTIER_EMULATE_SM') or '%d%d' % torch.cuda.get_device_capability())
    sm120 = sm == 120
    # window cache: 16384 tokens in pages of 256; compressed cache: 8192 slots in pages of 128 (64 for FlashInfer)
    lat1 = torch.randn(16384, DIM, device=dev, generator=g) * torch.rand(16384, 1, device=dev, generator=g) * 4
    lat2 = torch.randn(8192, DIM, device=dev, generator=g) * torch.rand(8192, 1, device=dev, generator=g) * 4
    c1, k1 = pack(lat1, 256)
    c2, k2 = pack(lat2, 128)
    c1_64, _ = pack(lat1, 64)
    c2_64, _ = pack(lat2, 64)
    sink = torch.randn(HEADS, device=dev, generator=g)
    if '--bench' in sys.argv:
        return bench(c1, c2, c1_64, c2_64, sink, sm120)
    ok = True
    print(f'{"rows":>5s} {"extra":>5s} | {"xtier max":>9s} {"rms":>8s} | {"FlashInfer":>10s} {"rms":>8s} | '
          f'{"SGL triton":>10s} {"rms":>8s}')
    for rows in (1, 2, 12, 300, 4096):
        for extra in (False, True):
            q = (torch.randn(rows, 1, HEADS, DIM, device=dev, generator=g) * 2).to(torch.bfloat16)
            # every row sees at least its newest window key (as in serving: the window holds the query's token)
            idx1 = torch.randint(0, 16384, (rows, 1, 128), device=dev, generator=g, dtype=torch.int32)
            idx1[torch.rand(idx1.shape, device=dev, generator=g) < 0.1] = -1
            idx1[:, 0, 0] = torch.randint(0, 16384, (rows,), device=dev, generator=g, dtype=torch.int32)
            len1 = torch.randint(1, 129, (rows,), device=dev, generator=g, dtype=torch.int32)
            idx2 = len2 = None
            if extra:
                idx2 = torch.randint(0, 8192, (rows, 1, 512), device=dev, generator=g, dtype=torch.int32)
                idx2[torch.rand(idx2.shape, device=dev, generator=g) < 0.1] = -1
                len2 = torch.randint(0, 513, (rows,), device=dev, generator=g, dtype=torch.int32)
                len2[: max(1, rows // 8)] = 0
            ref = reference(q.squeeze(1), k1, idx1.squeeze(1), len1, k2, None if idx2 is None else idx2.squeeze(1),
                            len2, sink)
            out, _ = sparse_mla.sparse_mla(q, c1, idx1, len1, sink, SCALE, c2 if extra else None, idx2, len2)
            mine = errors(out, ref)
            # rows without any key (graph padding) write 0
            empty, _ = sparse_mla.sparse_mla(q, c1, idx1, torch.zeros_like(len1), sink, SCALE,
                                             c2 if extra else None, idx2, None if len2 is None else torch.zeros_like(len2))
            zero = bool((empty == 0).all())
            fi = tr = (float('nan'), float('nan'))
            if sm120:
                from sglang.kernels.ops.attention import flash_mla_sm120 as fm
                kw = dict(q=q, k_cache=c1_64, head_dim_v=DIM, softmax_scale=SCALE, indices=idx1, topk_length=len1,
                          attn_sink=sink, extra_k_cache=c2_64 if extra else None, extra_indices_in_kvcache=idx2,
                          extra_topk_length=len2)
                fi = errors(fm.flash_mla_with_kvcache_sm120(**kw)[0], ref)
            if sm >= 89:   # SGLang's fallback loads float8e4nv
                from sglang.kernels.ops.attention.flash_mla_sm120_triton import flash_mla_sparse_decode_triton
                tr = errors(flash_mla_sparse_decode_triton(q, c1, idx1, len1, sink, DIM, SCALE,
                                                           c2 if extra else None, idx2, len2)[0], ref)
            good = zero and mine[0] < 1e-2 and mine[1] < 4e-3 and not (sm120 and mine[1] > fi[1] * 1.05 + 1e-5)
            ok &= good
            print(f'{rows:5d} {str(extra):>5s} | {mine[0]:9.2e} {mine[1]:8.2e} | {fi[0]:10.2e} {fi[1]:8.2e} | '
                  f'{tr[0]:10.2e} {tr[1]:8.2e} {"" if zero else "  rows without keys: not 0"}'
                  f'{"" if good else "  <-- FAIL"}')
    print('DS-SPARSE-MLA-CHECK', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
