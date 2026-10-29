"""GLM-5.3's DSA indexer logits in torch below SM90 (xtier/packs/glm5_next/indexer_torch.py) against FP64 on random
FP8 inputs, next to SGLang's torch fallbacks (BF16 per-head scores): prefill (fp8_mqa_logits) and decode over pages
(fp8_paged_mqa_logits).

  python tests/glm_indexer_torch_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from xtier.packs.glm5_next import indexer_torch  # noqa: E402

H, D, BLOCK = 32, 128, 64


def rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm()).item()


def main():
    dev = torch.device('cuda')
    g = torch.Generator(device=dev).manual_seed(20261029)
    ok = True
    # prefill: 300 queries over 1000 keys, ranges [ks, ke)
    q = (torch.randn(300, H, D, device=dev, generator=g) * 2).to(torch.float8_e4m3fn)
    k = (torch.randn(1000, D, device=dev, generator=g) * 2).to(torch.float8_e4m3fn)
    k_scale = torch.rand(1000, device=dev, generator=g) + 0.5
    w = torch.randn(300, H, device=dev, generator=g)
    ks = torch.randint(0, 400, (300,), device=dev, generator=g, dtype=torch.int32)
    ke = ks + torch.randint(1, 600, (300,), device=dev, generator=g, dtype=torch.int32)
    scores = torch.einsum('qhd,kd->qhk', q.double(), k.double()).relu()
    ref = (scores * w.double().unsqueeze(2)).sum(1) * k_scale.double()
    pos = torch.arange(1000, device=dev)
    ref = ref.masked_fill(~((pos >= ks[:, None]) & (pos < ke[:, None])), 0.0)
    mine = indexer_torch.fp8_mqa_logits(q, (k, k_scale), w, ks, ke)
    from sglang.srt.layers.attention.dsa.utils import _fp8_mqa_logits_torch
    sgl = _fp8_mqa_logits_torch(q, (k, k_scale), w, ks, ke)
    e1, e2 = rel(mine, ref), rel(sgl, ref)
    ok &= e1 < 1e-5
    print(f'prefill 300 x 1000: xtier FP32 {e1:.2e}, SGLang torch (BF16 scores) {e2:.2e}')
    # decode: 3 rows over 8 pages of 64 slots in a pool of 40 pages, lengths up to 512
    pages = 40
    kv = torch.empty(pages, BLOCK * (D + 4), dtype=torch.uint8, device=dev)
    vals = (torch.randn(pages, BLOCK, D, device=dev, generator=g) * 2).to(torch.float8_e4m3fn)
    scl = torch.rand(pages, BLOCK, device=dev, generator=g) + 0.5
    kv[:, :BLOCK * D] = vals.view(torch.uint8).reshape(pages, -1)
    kv[:, BLOCK * D:] = scl.view(torch.uint8).reshape(pages, -1)
    kv4 = kv.view(pages, BLOCK, 1, D + 4)
    table = torch.randperm(pages, device=dev, generator=g)[:24].view(3, 8).to(torch.int32)
    lens = torch.tensor([512, 300, 77], device=dev, dtype=torch.int32)
    qd = (torch.randn(3, 1, H, D, device=dev, generator=g) * 2).to(torch.float8_e4m3fn)
    wd = torch.randn(3, H, device=dev, generator=g)
    kv_rows = vals[table.long()].reshape(3, 8 * BLOCK, D).double()
    sc_rows = scl[table.long()].reshape(3, 8 * BLOCK).double()
    refd = (torch.einsum('bhd,btd->bht', qd[:, 0].double(), kv_rows).relu() * wd.double().unsqueeze(2)).sum(1) * sc_rows
    refd = refd.masked_fill(torch.arange(8 * BLOCK, device=dev)[None, :] >= lens[:, None], 0.0)
    mined = indexer_torch.fp8_paged_mqa_logits(qd, kv4, wd, lens.view(3, 1), table, None, 8 * BLOCK)
    from sglang.srt.layers.attention.dsv4.indexer import fp8_paged_mqa_logits_torch
    sgld = fp8_paged_mqa_logits_torch(qd, kv4.view(torch.uint8), wd, lens, table, None, 8 * BLOCK, clean_logits=False)
    e3, e4 = rel(mined, refd), rel(sgld[:, :8 * BLOCK], refd)
    ok &= e3 < 1e-5
    print(f'decode 3 rows x 8 pages: xtier FP32 {e3:.2e}, SGLang torch (BF16 scores) {e4:.2e}')
    print('GLM-INDEXER-TORCH-CHECK', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
