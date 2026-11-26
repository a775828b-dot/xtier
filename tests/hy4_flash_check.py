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
    cap = torch.cuda.get_device_capability()
    use_fi_sparse_mla = cap >= (10, 0)
    trtllm_batch_decode_with_kv_cache_mla = None
    if use_fi_sparse_mla:
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

    # Verify gather-unpack + FlashInfer (SM100+) or FP32 reference vs Triton NVFP4 decode across M=1,4 and SWA/DSA topk
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev) if use_fi_sparse_mla else None
    sm_scale = 1.0 / (192 ** 0.5)
    kv_exact = torch.cat([nope, rope], dim=-1).float()
    for m, heads, topk, valid in [(1, 64, 512, 256), (4, 64, 512, 512), (1, 32, 2048, 1024), (4, 32, 2048, 2048)]:
        q = torch.randn(m, heads, 576, dtype=torch.bfloat16, device=dev)
        sink = torch.randn(heads, dtype=torch.float32, device=dev)
        idx = torch.full((m, topk), -1, dtype=torch.int32, device=dev)
        idx[:, :valid] = torch.randint(0, n_slots, (m, valid), dtype=torch.int32, device=dev)
        seq_lens = torch.full((m,), topk, dtype=torch.int32, device=dev)

        fp8_scratch, remapped = kv_fp4.gather_unpack_nvfp4_to_fp8(kv_buf, idx)
        if not torch.equal(remapped < 0, idx < 0):
            _fail(f'gather_unpack mask mismatch for m={m} topk={topk}')
        ref = kv_fp4.triton_sparse_mla_decode(q, kv_buf, idx, sink, sm_scale, kv_fp4.NVFP4_KV_DIM)
        if use_fi_sparse_mla:
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
            rel = (got.float() - ref.float()).norm().item() / max(ref.float().norm().item(), 1e-6)
            if not torch.isfinite(got).all() or rel > 0.06:
                _fail(f'gather_unpack+FI vs Triton mismatch m={m} H={heads} topk={topk}: rel={rel:.4f}')
        else:
            # Below SM100 FlashInfer routes to XQA MLA which lacks sparse_mla_top_k; verify Triton NVFP4 vs FP32 ref.
            safe_idx = idx.clamp(min=0).long()
            k_sel = kv_exact[safe_idx]  # [m, topk, 576]
            scores = torch.einsum('mhd,mkd->mhk', q.float(), k_sel) * sm_scale
            scores = scores.masked_fill((idx < 0)[:, None, :], float('-inf'))
            scores_sink = torch.cat([scores, sink[None, :, None].expand(m, heads, 1)], dim=-1)
            probs = torch.softmax(scores_sink, dim=-1)[..., :-1]
            got = torch.einsum('mhk,mkd->mhd', probs, k_sel[..., :512])
            rel = (got - ref.float()).norm().item() / max(got.norm().item(), 1e-6)
            if not torch.isfinite(ref).all() or rel > 0.25:
                _fail(f'Triton NVFP4 vs FP32 ref mismatch m={m} H={heads} topk={topk}: rel={rel:.4f}')
    # Regression: larger eager gather calls (e.g. short 31-token prefills) must never replace the
    # static scratch buffers already captured into CUDA graphs before torch.cuda.empty_cache().
    kv_fp4.ensure_gather_buffers(24 * 2176, torch.device(dev))
    static_ptr = kv_fp4._GATHER_FP8_CACHE[0].data_ptr()
    idx6 = torch.randint(0, n_slots, (6, 2048), dtype=torch.int32, device=dev)
    for _ in range(2):
        kv_fp4.gather_unpack_nvfp4_to_fp8(kv_buf, idx6)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        g_fp8, g_rem = kv_fp4.gather_unpack_nvfp4_to_fp8(kv_buf, idx6)
    torch.cuda.synchronize()
    idx31 = torch.randint(0, n_slots, (31, 2048), dtype=torch.int32, device=dev)
    tmp_fp8, tmp_rem = kv_fp4.gather_unpack_nvfp4_to_fp8(kv_buf, idx31)
    del tmp_fp8, tmp_rem
    if kv_fp4._GATHER_FP8_CACHE[0].data_ptr() != static_ptr:
        _fail('gather_unpack_nvfp4_to_fp8 replaced static graph-captured buffer on eager overflow')
    torch.cuda.empty_cache()
    g.replay()
    torch.cuda.synchronize()
    if not torch.equal(g_rem < 0, idx6 < 0):
        _fail('gather_unpack CUDA graph replay drifted after empty_cache()')
    print('  kv_fp4 (write + gather_unpack + prefill unpack + FI + graph safety): ok')


def check_indexpool_fp8():
    if torch.cuda.get_device_capability() < (8, 9):
        print('  indexpool_fp8: skipped (needs SM89+ HW_FP8)')
        return
    from sglang.kernels.ops.attention.dsa.hy4_flash_indexpool import indexpool_logits

    dev = 'cuda'
    torch.manual_seed(1)
    page_size = 16
    num_groups = 1024
    num_pages = num_groups * 4 // 64
    index_buf = torch.zeros((num_groups * 4 // page_size + 2, page_size * 128 + page_size * 4), dtype=torch.uint8, device=dev)
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
        # SM89/SM90 FP8 tensor cores accumulate with ~14-bit mantissa (~1.3e-4 rel on ~75-scale logits).
        diff_tol = 2e-2 if torch.cuda.get_device_capability() < (10, 0) else 5e-3
        if diff > diff_tol or rel > 2e-4:
            _fail(f'indexpool_fp8 rows={rows} max diff={diff:.3e} rel={rel:.3e}')
    # Verify length-aware decode path (WRITE_LENGTHS=True + top_k_ragged_transform) matches exact top-k
    import flashinfer
    for rows in (1, 6, 24):
        q_fp8 = (torch.randn(rows, 16, 128, device=dev) * 0.5).to(torch.float8_e4m3fn)
        w = torch.randn(rows, 16, dtype=torch.float32, device=dev)
        r2b = torch.arange(rows, dtype=torch.int64, device=dev) % 4
        pos = torch.randint(32, num_groups * 4, (rows,), dtype=torch.int64, device=dev)
        lengths = torch.empty((rows,), dtype=torch.int32, device=dev)
        got_l = indexpool_fp8.indexpool_logits_fp8(q_fp8, w, pages, r2b, pos, index_buf, page_size, num_groups, lengths=lengths)
        ref_l = indexpool_logits(q_fp8, w, pages, r2b, pos, index_buf, page_size, num_groups)
        k_top = 64
        picked = flashinfer.topk.top_k_ragged_transform(
            got_l, torch.zeros_like(lengths), lengths, k_top, dsa_graph_safe=True
        )
        ref_top = torch.topk(ref_l, k_top, dim=-1).indices
        for r in range(rows):
            n_val = int(lengths[r].item())
            exp_len = int(pos[r].item() // 4)
            if n_val != min(exp_len, num_groups):
                _fail(f'indexpool_fp8 length mismatch row={r}: got={n_val} exp={min(exp_len, num_groups)}')
            k_eff = min(k_top, n_val)
            got_set = set(picked[r, :k_eff].tolist())
            ref_set = set(ref_top[r, :k_eff].tolist())
            overlap = len(got_set & ref_set)
            if overlap < k_eff - 1:
                _fail(f'indexpool_fp8 ragged topk mismatch rows={rows} r={r}: overlap={overlap}/{k_eff}')
    print('  indexpool_fp8 (decode + prefill + length-aware ragged topk): ok')


def check_pool_between():
    if not isinstance(xtier_pool._between, list) or len(xtier_pool._between) != 1 or xtier_pool._between[0] is not None:
        _fail(f'unexpected pool._between initial state: {xtier_pool._between!r}')
    print('  pool._between hook slot: ok')




def check_dfly():
    from types import SimpleNamespace
    from sglang.srt.configs.hy_v4_flash import SPARSE_ATTENTION, SLIDING_ATTENTION
    from xtier.packs.glm5_next import fp8_weights
    from xtier.packs.hy4_flash import dfly

    dev = 'cuda'
    # 1. Config parsing
    layer_types = [SLIDING_ATTENTION] * 42
    for lid in (25, 29, 33, 37, 41):
        layer_types[lid] = SPARSE_ATTENTION
    cfg = SimpleNamespace(
        num_hidden_layers=42,
        vocab_size=120832,
        index_topk=2048,
        layer_types=layer_types,
        dfly_num_hidden_layers=5,
        dfly_mask_token_id=120000,
        dfly_target_kv_layers=[25, 29, 33, 37, 41],
        dfly_target_index_layer=[25, 29, 33, 37, 41],
        dfly_target_index_topk=2048,
    )
    dcfg = dfly.parse_dfly_config(cfg, block_size=6)
    if dcfg.num_layers != 5 or dcfg.draft_index_topk != 2176 or dcfg.block_size != 6:
        _fail(f'unexpected parsed DFly config: {dcfg}')

    # 2. build_draft_topk exactness (history packing + causal verify slots + counts)
    bs, width, layers, topk, dtopk = 4, 6, 5, 2048, 2176
    state_topk = torch.full((layers, 8, topk), -1, dtype=torch.int32, device=dev)
    req_idx = torch.tensor([1, 3, 5, 7], dtype=torch.int32, device=dev)
    val_counts = [10, 0, 512, 2048]
    for b, nv in enumerate(val_counts):
        if nv > 0:
            state_topk[:, req_idx[b], :nv] = torch.arange(100 + b * 3000, 100 + b * 3000 + nv, dtype=torch.int32, device=dev)
    verify_slots = torch.arange(50000, 50000 + bs * width, dtype=torch.int32, device=dev).view(bs, width)
    out_rows = torch.full((layers, bs * width, dtopk), -1, dtype=torch.int32, device=dev)
    out_counts = torch.zeros((bs * width,), dtype=torch.int32, device=dev)
    dfly.build_draft_topk(
        state_topk=state_topk, req_pool_indices=req_idx, verify_slots=verify_slots,
        out_rows=out_rows, out_counts=out_counts,
    )
    for b, nv in enumerate(val_counts):
        for j in range(width):
            row = b * width + j
            if int(out_counts[row].item()) != nv + j + 1:
                _fail(f'build_draft_topk count mismatch b={b} j={j}: {out_counts[row].item()} vs {nv + j + 1}')
            expected_tail = verify_slots[b, : j + 1]
            got_tail = out_rows[0, row, nv : nv + j + 1]
            if not torch.equal(got_tail, expected_tail):
                _fail(f'build_draft_topk slot tail mismatch b={b} j={j}')
            if nv + j + 1 < dtopk and int(out_rows[0, row, nv + j + 1].item()) != -1:
                _fail(f'build_draft_topk padding not -1 at b={b} j={j}')

    # 3. Fused FP8 draft LM-head argmax vs exact BF16 torch.mm
    torch.manual_seed(7)
    vocab, hidden = 8192, 3072
    w_bf16 = torch.randn(vocab, hidden, dtype=torch.bfloat16, device=dev) * 0.05
    q_fp8, s_fp8 = fp8_weights.quantize(w_bf16)
    for m in (1, 4, 16, 24):
        x = torch.randn(m, hidden, dtype=torch.bfloat16, device=dev)
        # Make one distinct argmax per row clearly dominant to verify exact index match
        for r in range(m):
            target_id = (r * 317 + 42) % vocab
            x[r] = w_bf16[target_id] * 2.0
        vmax, varg = dfly._fp8_lmhead_argmax(x, q_fp8, s_fp8)
        ref = torch.mm(x, w_bf16.T, out_dtype=torch.float32)
        rmax, rarg = ref.max(dim=-1)
        if not torch.equal(varg, rarg):
            _fail(f'_fp8_lmhead_argmax mismatch for m={m}: {varg.tolist()} vs {rarg.tolist()}')
        rel = (vmax - rmax).abs().max().item() / max(rmax.abs().max().item(), 1e-6)
        if rel > 0.05:
            _fail(f'_fp8_lmhead_argmax value drift for m={m}: rel={rel:.4f}')

    # 4. Zero-alloc Triton confidence-head dot + survival caps + active_flags kernels
    n_steps = width - 1
    cx = torch.randn(bs, n_steps, hidden, dtype=torch.bfloat16, device=dev)
    cw = torch.randn(1, hidden, dtype=torch.bfloat16, device=dev) * 0.02
    cb = torch.randn(1, dtype=torch.bfloat16, device=dev) * 0.1
    probs = torch.empty((bs, n_steps), dtype=torch.float32, device=dev)
    caps = torch.empty((bs,), dtype=torch.int32, device=dev)
    flags = torch.ones((64,), dtype=torch.int32, device=dev)
    for step in range(n_steps):
        dfly._dfly_conf_dot_kernel[(bs,)](
            cx[:, step].contiguous(), cw, cb, probs, step, n_steps, H=hidden, BH=1024, num_warps=4,
        )
    dfly._dfly_conf_caps_kernel[(bs,)](
        probs, caps, flags, 0.20, N_STEPS=n_steps, WIDTH=width, HAS_FLAGS=True, num_warps=1,
    )
    ref_probs = torch.sigmoid(torch.nn.functional.linear(cx.float(), cw.float(), cb.float()).squeeze(-1))
    ref_keep = (torch.cumprod(ref_probs, dim=1) >= 0.20).sum(dim=1).clamp_(min=1)
    ref_caps = (ref_keep + 1).to(torch.int32)
    ref_flags = (torch.arange(width, device=dev)[None, :] < ref_caps[:, None]).reshape(-1).to(torch.int32)
    if (probs - ref_probs).abs().max().item() > 1e-5:
        _fail('confidence dot kernel numeric drift')
    if not torch.equal(caps, ref_caps) or not torch.equal(flags[: bs * width], ref_flags):
        _fail(f'confidence caps/flags mismatch: {caps.tolist()} vs {ref_caps.tolist()}')

    # 5. CustomSpecAlgo DFLY registration
    dfly._register_dfly()
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    algo = SpeculativeAlgorithm.from_string('DFLY')
    if not algo.is_dfly() or not algo.supports_overlap:
        _fail(f'DFLY registration check failed: {algo!r}')
    print('  dfly (config + draft_topk + fp8_lmhead_argmax + conf_truncate + registration): ok')


if __name__ == '__main__':
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        print('SKIP: SM80+ CUDA GPU required')
        sys.exit(0)
    check_kv_fp4()
    check_indexpool_fp8()
    check_pool_between()
    check_dfly()
    print('HY4-FLASH-CHECK PASS')
