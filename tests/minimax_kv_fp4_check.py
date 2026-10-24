"""NVFP4 KV cache of the MiniMax pack (packs/minimax_m3/kv_fp4.py) on SGLang's MSA-v1 NVFP4 pool layout, at
MiniMax-M3.1's shape (4 KV heads x 128, one 128-dim index K head, 128-token pages):

- layout: SGLang's quantizer (NVFP4KVQuantizeUtil, FlashInfer's nvfp4_kv_quantize) read with kv_fp4's convention
  (byte i = values 2i, 2i+1, scale i // 8) equals SGLang's dequantization, bitwise;
- decode: the main-head sparse decode and the index scores / top-k against SGLang's Triton kernels on the cache
  dequantized to BF16;
- prefill: the copies (dequantize_context) are bit-equal to the dequantized cache, and the hooked minimax_sparse_prefill
  on the NVFP4 cache equals SGLang's minimax_sparse_prefill on the dequantized cache (index, top-k and main kernels);
- store: SGLang's fused nvfp4_store_kv_index writes the same bytes as the pool's eager path (the pack enables it on
  SM120, where SGLang does not).

  CUDA_HOME=<nvcc toolkit> python tests/minimax_kv_fp4_check.py   (on SGLang's M3.1 branch; FlashInfer builds its
  NVFP4 kernels on first use)
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from xtier.packs.minimax_m3 import kv_fp4

torch.manual_seed(0)
dev = torch.device('cuda')
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=dev)
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
PAGE, D, KH, QH, IH = 128, 128, 4, 64, 4


def quant(x):
    """x [slots, heads, D] FP32 -> page payload [pages, heads, 128, D/2] uint8, scale bytes [pages, heads, 128, D/16],
    dequantized [slots, heads, D] (amax / 6 E4M3 scales, nearest E2M1)."""
    shape = x.shape
    g = x.reshape(-1, 16)
    s = (g.abs().amax(-1, keepdim=True) / 6).clamp_min(2 ** -9).to(torch.float8_e4m3fn)
    v = g / s.float()
    mag = (v.abs().clamp(max=6)[..., None] - E2M1).abs().argmin(-1)
    neg = v < 0
    deq = (E2M1[mag] * torch.where(neg, -1.0, 1.0) * s.float()).reshape(shape)
    code = (mag | (neg.long() << 3)).reshape(shape)
    pay = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    sc = s.view(torch.uint8).reshape(*shape[:-1], D // 16)
    pages = shape[0] // PAGE
    lay = lambda t: t.view(pages, PAGE, t.shape[1], t.shape[2]).permute(0, 2, 1, 3).contiguous()
    return lay(pay), lay(sc), deq


def mapping(seq_lens):
    """req_to_token with each request's pages in random physical order; the number of pages."""
    blocks = [-(-s // PAGE) for s in seq_lens]
    pages = sum(blocks) + 1
    perm = torch.randperm(pages - 1, device=dev) + 1
    r2t = torch.zeros(len(seq_lens), 16384, dtype=torch.int32, device=dev)
    off = 0
    for r, n in enumerate(blocks):
        for blk in range(n):
            pg = int(perm[off + blk])
            r2t[r, blk * PAGE:(blk + 1) * PAGE] = torch.arange(pg * PAGE, (pg + 1) * PAGE, device=dev)
        off += n
    return r2t, pages


def check_layout():
    from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

    gs = torch.ones(1, dtype=torch.float32, device=dev)
    ok = True
    for shape, scale in (((4096, KH, D), 3.0), ((4096, 1, D), 0.02), ((7, KH, D), 40.0)):
        x = (torch.randn(*shape, device=dev) * scale).to(torch.bfloat16)
        pay, sc, _ = NVFP4KVQuantizeUtil.quantize(x, gs)
        vals = torch.stack([LUT[(pay & 15).long()], LUT[(pay >> 4).long()]], -1).reshape(*pay.shape[:-1], -1)
        mine = vals * sc.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
        ok &= torch.equal(mine, NVFP4KVQuantizeUtil.dequantize(pay, sc, gs).float())
    print('layout: kv_fp4 reading of SGLang\'s quantizer = SGLang\'s dequantization:', ok, flush=True)
    return ok


def check_decode(seq_lens, topk=16):
    from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import flash_decode_with_topk_idx
    from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import flash_decode_with_gqa_share_sparse

    b = len(seq_lens)
    r2t, pages = mapping(seq_lens)
    kp, ks, kd = quant(torch.randn(pages * PAGE, KH, D, device=dev))
    vp, vs, vd = quant(torch.randn(pages * PAGE, KH, D, device=dev))
    ip, isc, idq = quant(torch.randn(pages * PAGE, 1, D, device=dev))
    seq = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    slot_ids = torch.arange(b, dtype=torch.int32, device=dev)
    q = torch.randn(b, QH, D, device=dev).to(torch.bfloat16)
    idx = torch.full((KH, b, topk), -1, dtype=torch.int32, device=dev)
    for r, s in enumerate(seq_lens):
        for h in range(KH):
            pick = torch.randperm(-(-s // PAGE), device=dev)[:topk]
            idx[h, r, :len(pick)] = pick.to(torch.int32)
    ref = flash_decode_with_gqa_share_sparse(q, None, kd.to(torch.bfloat16), vd.to(torch.bfloat16), r2t, seq, slot_ids,
                                             PAGE, idx, page_size=PAGE)
    got = kv_fp4.fp4_sparse_decode(q, kp, ks, vp, vs, r2t, slot_ids, idx, seq, PAGE)
    err = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    iq = torch.randn(b, IH, D, device=dev).to(torch.bfloat16)
    kw = dict(sink=None, v_cache=None, req_to_token=r2t, seq_lens=seq, max_seqlen=max(seq_lens), slot_ids=slot_ids,
              block_size=PAGE, topk=topk, init_blocks=0, local_blocks=1, score_type='max', disable_index_value=True,
              page_size=PAGE)
    t_ref = flash_decode_with_topk_idx(q=iq, k_cache=idq.to(torch.bfloat16), **kw)[1]
    t_got = kv_fp4.fp4_decode_topk_idx(iq, ip[:, 0], isc[:, 0], **kw)[1]
    same = all(set(a.tolist()) == set(c.tolist()) for a, c in zip(t_ref.reshape(-1, topk), t_got.reshape(-1, topk)))
    ok = err < 2e-3 and same
    print(f'decode {seq_lens}: main rel L2 {err:.1e}, index top-k equal {same}', 'PASS' if ok else 'FAIL', flush=True)
    return ok


def check_prefill(prefix_lens, ext_lens):
    from sglang.srt.layers.attention.minimax_sparse_ops import minimax_sparse as ops

    seq_lens = [p + e for p, e in zip(prefix_lens, ext_lens)]
    b = len(seq_lens)
    r2t, pages = mapping(seq_lens)
    kp, ks, kd = quant(torch.randn(pages * PAGE, KH, D, device=dev))
    vp, vs, vd = quant(torch.randn(pages * PAGE, KH, D, device=dev))
    ip, isc, idq = quant(torch.randn(pages * PAGE, 1, D, device=dev) * 0.3)
    ip, isc = ip[:, 0], isc[:, 0]   # the index pool has no head dimension
    kp._xtier_fp4, vp._xtier_fp4, ip._xtier_fp4 = ks, vs, isc   # as attach_scales leaves the pool's views
    total = sum(ext_lens)
    q = torch.randn(total, QH, D, device=dev).to(torch.bfloat16)
    iq = torch.randn(total, IH, D, device=dev).to(torch.bfloat16)
    cu = torch.tensor([0] + list(itertools.accumulate(ext_lens)), dtype=torch.int32, device=dev)
    seq = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    pre = torch.tensor(prefix_lens, dtype=torch.int32, device=dev)
    slot_ids = torch.arange(b, dtype=torch.int32, device=dev)
    rows = -(-max(seq_lens) // PAGE) * PAGE
    got = kv_fp4.dequantize_context(kp, ks, r2t, slot_ids, seq, rows).float()
    want = torch.stack([kd[r2t[r, :rows].long()] for r in range(b)]).reshape(b * rows, KH, D)
    copy_ok = all(torch.equal(got[r * rows:r * rows + s], want[r * rows:r * rows + s]) for r, s in enumerate(seq_lens))
    args = (cu, seq, pre, max(ext_lens), max(seq_lens), 1, PAGE, 16, 0, 1)
    kw = dict(score_type='max', disable_index_value=True, use_msa=False, seqlens_cpu=ext_lens, page_size=PAGE,
              return_topk_idx=True)
    orig = ops.minimax_sparse_prefill
    _, o_hook, t_hook = kv_fp4._sparse_prefill(orig, q, kp, vp, None, iq, ip, None, None, r2t, slot_ids, *args, **kw)
    _, o_ref, t_ref = orig(q, kd.to(torch.bfloat16), vd.to(torch.bfloat16), None, iq, idq.to(torch.bfloat16), None,
                           None, r2t, slot_ids, *args, **kw)
    err = ((o_hook.float() - o_ref.float()).norm() / o_ref.float().norm()).item()
    same = all(set(a.tolist()) == set(c.tolist()) for a, c in zip(t_hook.reshape(-1, 16), t_ref.reshape(-1, 16)))
    ok = copy_ok and err < 1e-3 and same
    print(f'prefill prefix {prefix_lens} ext {ext_lens}: copies bit-equal {copy_ok}, rel L2 {err:.1e}, top-k equal '
          f'{same}', 'PASS' if ok else 'FAIL', flush=True)
    return ok


def check_store():
    from sglang.kernels.ops.kvcache.nvfp4_store_kv_index import nvfp4_store_kv_index
    from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4IndexKPool, MiniMaxM3Nvfp4MainKVPool

    def pools():
        kw = dict(size=8192, page_size=PAGE, dtype=torch.bfloat16, layer_num=2, device='cuda', enable_memory_saver=False,
                  start_layer=0, end_layer=2)
        return (MiniMaxM3Nvfp4MainKVPool(head_num=KH, head_dim=D, **kw),
                MiniMaxM3Nvfp4IndexKPool(head_num=1, head_dim=D, **kw))

    (ref_main, ref_index), (main, index) = pools(), pools()
    ok = True
    for n in (1, 6, 12, 1000, 4096):
        proj = (torch.randn(n, (2 * KH + 1) * D + 512, device=dev) * 2).to(torch.bfloat16)   # strided K/V views
        k, v = proj[:, :KH * D].view(n, KH, D), proj[:, KH * D:2 * KH * D].view(n, KH, D)
        ik = proj[:, 2 * KH * D:(2 * KH + 1) * D]
        loc = torch.randperm(8192, device=dev)[:n]
        for layer in range(2):
            ref_main.set_kv_buffer(None, loc, k, v, layer_id_override=layer)
            ref_index.set_k_buffer(layer, loc, ik)
            kpay, vpay, ksc, vsc = main.get_raw_kv_buffers(layer)
            ipay, isc = index.get_raw_k_buffers(layer)
            nvfp4_store_kv_index(k, v, ik, kpay, ksc, vpay, vsc, ipay, isc, loc, main.global_scale, page_size=PAGE)
        ok &= torch.equal(ref_main.slab, main.slab) and torch.equal(ref_index.slab, index.slab)
    print('store: fused nvfp4_store_kv_index bytes = the pool\'s eager path:', ok, flush=True)
    return ok


oks = [check_layout(), check_decode([4000, 2500, 1300]), check_decode([8000]), check_decode([128, 129, 300]),
       check_prefill([0], [1500]), check_prefill([1000], [500]), check_prefill([0, 300], [900, 600]),
       check_prefill([5000], [3000]), check_store()]
print('MINIMAX-KV-FP4-CHECK', 'PASS' if all(oks) else 'FAIL')
