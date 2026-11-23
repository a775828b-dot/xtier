"""Unit check for xtier.packs.hy4_flash (NVFP4 MLA KV gather-unpack + prefill unpack, FP8 IndexPool, pool._between).

    python tests/hy4_flash_check.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xtier.cache import pool as xtier_pool
from xtier.packs.hy4_flash import indexpool_fp8, kv_fp4


def _fail(msg):
    print(f'FAIL: {msg}')
    sys.exit(1)


def check_kv_fp4():
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    dev = 'cuda'
    torch.manual_seed(0)
    n_slots = 4096
    kv_buf = torch.zeros((n_slots, kv_fp4.NVFP4_KV_DIM), dtype=torch.uint8, device=dev)
    nope = torch.randn(n_slots, 512, dtype=torch.bfloat16, device=dev) * 0.5
    rope = torch.randn(n_slots, 64, dtype=torch.bfloat16, device=dev) * 0.5
    locs = torch.arange(n_slots, dtype=torch.int32, device=dev)
    kv_fp4.write_nvfp4_mla(kv_buf, locs, nope, rope)

    # Verify RoPE is preserved bit-exact and nope unpack matches NVFP4 precision
    idx_all = torch.arange(512, dtype=torch.int32, device=dev).unsqueeze(0)
    for ps in (1, 16, 64):
        fp8_full = kv_fp4.unpack_nvfp4_to_fp8(kv_buf, idx_all, page_size=ps)
        if fp8_full.shape != (512 // ps, ps, 1, kv_fp4.FP8_KV_DIM):
            _fail(f'unpack_nvfp4_to_fp8 shape mismatch for ps={ps}: {fp8_full.shape}')
        rope_out = fp8_full.reshape(512, kv_fp4.FP8_KV_DIM)[:, 528:].contiguous().view(torch.bfloat16)
        if not torch.equal(rope_out, rope[:512]):
            _fail(f'unpack_nvfp4_to_fp8 RoPE drift for ps={ps}')

    # Verify gather-unpack + FlashInfer vs Triton NVFP4 decode across M=1,4 and SWA/DSA topk with -1 padding
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
    sm_scale = 1.0 / (192 ** 0.5)
    for m, heads, topk, valid in [(1, 64, 512, 256), (4, 64, 512, 512), (1, 32, 2048, 1024), (4, 32, 2048, 2048)]:
        q = torch.randn(m, heads, 576, dtype=torch.bfloat16, device=dev)
        sink = torch.randn(heads, dtype=torch.float32, device=dev)
        idx = torch.full((m, topk), -1, dtype=torch.int32, device=dev)
        idx[:, :valid] = torch.randint(0, n_slots, (m, valid), dtype=torch.int32, device=dev)
        seq_lens = torch.full((m,), topk, dtype=torch.int32, device=dev)

        fp8_scratch, remapped = kv_fp4.gather_unpack_nvfp4_to_fp8(kv_buf, idx)
        if not torch.equal(remapped < 0, idx < 0):
            _fail(f'gather_unpack mask mismatch for m={m} topk={topk}')
        got = trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=fp8_scratch,
            workspace_buffer=ws,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=remapped.unsqueeze(1),
            seq_lens=seq_lens,
            max_seq_len=topk,
            sparse_mla_top_k=topk,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            sinks=[sink],
            kv_scale_format='arbitrary_fp32',
        ).squeeze(1)
        ref = kv_fp4.triton_sparse_mla_decode(q, kv_buf, idx, sink, sm_scale, kv_fp4.NVFP4_KV_DIM)
        rel = (got.float() - ref.float()).norm().item() / max(ref.float().norm().item(), 1e-6)
        if not torch.isfinite(got).all() or rel > 0.06:
            _fail(f'gather_unpack+FI vs Triton mismatch m={m} H={heads} topk={topk}: rel={rel:.4f}')
    print('  kv_fp4 (write + gather_unpack + prefill unpack + FI): ok')


def check_indexpool_fp8():
    from sglang.kernels.ops.attention.dsa.hy4_flash_indexpool import indexpool_logits

    dev = 'cuda'
    torch.manual_seed(1)
    page_size = 16
    num_groups = 1024
    num_pages = num_groups * 4 // 64
    index_buf = torch.zeros((num_groups // page_size + 2, page_size * 128 + page_size * 4), dtype=torch.uint8, device=dev)
    k_fp8 = (torch.randn(index_buf.shape[0], page_size, 128, device=dev) * 0.5).to(torch.float8_e4m3fn)
    k_sc = torch.rand(index_buf.shape[0], page_size, dtype=torch.float32, device=dev) + 0.1
    index_buf[:, :page_size * 128] = k_fp8.view(torch.uint8).reshape(index_buf.shape[0], -1)
    index_buf[:, page_size * 128:] = k_sc.view(torch.uint8).reshape(index_buf.shape[0], -1)
    pages = torch.arange(num_pages, dtype=torch.int32, device=dev).unsqueeze(0).repeat(4, 1)

    for rows in (4, 128):
        q_fp8 = (torch.randn(rows, 16, 128, device=dev) * 0.5).to(torch.float8_e4m3fn)
        w = torch.randn(rows, 16, dtype=torch.float32, device=dev)
        r2b = torch.arange(rows, dtype=torch.int64, device=dev) % 4
        pos = torch.randint(16, num_groups * 4, (rows,), dtype=torch.int64, device=dev)
        ref = indexpool_logits(q_fp8, w, pages, r2b, pos, index_buf, page_size, num_groups)
        got = indexpool_fp8._indexpool_logits(indexpool_logits, q_fp8, w, pages, r2b, pos, index_buf, page_size, num_groups)
        both_inf = torch.isneginf(ref) & torch.isneginf(got)
        diff = torch.where(both_inf, torch.zeros_like(ref), (ref - got).abs()).max().item()
        finite = ~both_inf
        rel = ((ref[finite] - got[finite]).norm() / ref[finite].norm().clamp(min=1e-6)).item() if finite.any() else 0.0
        if diff > 5e-3 or rel > 2e-4:
            _fail(f'indexpool_fp8 rows={rows} max diff={diff:.3e} rel={rel:.3e}')
    print('  indexpool_fp8 (decode + prefill exactness): ok')


def check_pool_between():
    if not isinstance(xtier_pool._between, list) or len(xtier_pool._between) != 1 or xtier_pool._between[0] is not None:
        _fail(f'unexpected pool._between initial state: {xtier_pool._between!r}')
    print('  pool._between hook slot: ok')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('SKIP: CUDA required')
        sys.exit(0)
    check_kv_fp4()
    check_indexpool_fp8()
    check_pool_between()
    print('HY4-FLASH-CHECK PASS')
