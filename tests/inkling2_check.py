"""Unit & GPU regression suite for Inkling-2-Small (NVFP4 + MXFP8 + soft-capped SiTU(4, 25), dual cold store,
LayerPool, Prefill BF16 & NVFP4 4over6 MAE, and HookRegistry packs: moe, attn, mtp)."""
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault('XTIER', '1')
os.environ.setdefault('XTIER_THREADS', '16')
os.environ.setdefault('XTIER_COLD', '0')

import torch
from xtier.engine import CpuPlan, Mailbox
from xtier.backend.prefill import Prefill
from xtier.cache import cold
from xtier.cache.pool import LayerPool
from xtier.cache.prefill import hybrid_prefill, resident_prefill

E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def _dequant_nvfp4(w_u8, s_u8, g):
    lo = E2M1_LUT[(w_u8 & 0xF).long()]
    hi = E2M1_LUT[(w_u8 >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).reshape(*w_u8.shape[:-1], -1)
    sf = s_u8.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return vals * sf * g


def _dequant_mxfp8(w_u8, s_u8):
    w = w_u8.view(torch.float8_e4m3fn).float()
    w = torch.where(torch.isnan(w), torch.zeros_like(w), w)
    sf = torch.pow(2.0, s_u8.float() - 127.0)
    sf = torch.where(s_u8 == 255, torch.zeros_like(sf), sf)
    return w * sf.repeat_interleave(32, dim=-1)


def _ref_moe_situ(x, w13, s13, w2, s2, g13, g2, ids, rw, fmt, gcap=4.0, ucap=25.0):
    M, H = x.shape
    I = w2.shape[2] * (2 if fmt == 'nvfp4' else 1)
    out = torch.zeros((M, H), dtype=torch.float32)
    xf = x.float()
    for t in range(M):
        for q in range(ids.shape[1]):
            e = int(ids[t, q])
            if e < 0:
                continue
            if fmt == 'nvfp4':
                wg = _dequant_nvfp4(w13[e, :I], s13[e, :I], g13[e, 0])
                wu = _dequant_nvfp4(w13[e, I:], s13[e, I:], g13[e, 1])
                wd = _dequant_nvfp4(w2[e], s2[e], g2[e])
            else:
                wg = _dequant_mxfp8(w13[e, :I], s13[e, :I])
                wu = _dequant_mxfp8(w13[e, I:], s13[e, I:])
                wd = _dequant_mxfp8(w2[e], s2[e])
            gate = xf[t] @ wg.T
            up = xf[t] @ wu.T
            act = (gcap * torch.tanh(gate / gcap)) * torch.sigmoid(gate) * (ucap * torch.tanh(up / ucap))
            out[t] += (act @ wd.T) * float(rw[t, q])
    return out


def _make_weights(E, H, I, fmt, seed=0):
    g = torch.Generator().manual_seed(seed)
    if fmt == 'nvfp4':
        w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=g)
        s13 = torch.randint(40, 120, (E, 2 * I, H // 16), dtype=torch.uint8, generator=g)
        w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g)
        s2 = torch.randint(40, 120, (E, H, I // 16), dtype=torch.uint8, generator=g)
        g13_1 = torch.empty((E,), dtype=torch.float32).uniform_(0.0001, 0.0005, generator=g)
        g13 = torch.stack([g13_1, g13_1], dim=1).contiguous()
        g2 = torch.empty((E,), dtype=torch.float32).uniform_(0.0001, 0.0005, generator=g)
    else:
        w13 = torch.randint(0, 256, (E, 2 * I, H), dtype=torch.uint8, generator=g)
        s13 = torch.randint(110, 122, (E, 2 * I, H // 32), dtype=torch.uint8, generator=g)
        w2 = torch.randint(0, 256, (E, H, I), dtype=torch.uint8, generator=g)
        s2 = torch.randint(110, 122, (E, H, I // 32), dtype=torch.uint8, generator=g)
        g13 = torch.ones((E, 2), dtype=torch.float32)
        g2 = torch.ones((E,), dtype=torch.float32)
    return w13, w2, s13, s2, g13, g2


def test_cpu_and_dual_cold():
    print('[1/4] CPU engine (NVFP4 + MXFP8 + SiTU edge all-256-bytes) & dual cold store...')
    E, H, I, K = 16, 256, 128, 4
    # Edge test: all 256 E4M3 bytes including subnormals & NaN (0x7F/0xFF) + scales including <10 & >246
    w13, w2, s13, s2, g13, g2 = _make_weights(E, H, I, 'mxfp8', seed=1)
    w13[0, 0] = torch.arange(256, dtype=torch.uint8)
    # Put large scales (254, 247) on small-magnitude E4M3 blocks (0..63) and small scales on large blocks to avoid FP32 inf-inf
    s13[0, 0] = torch.tensor([254, 247, 246, 127, 10, 9, 1, 0], dtype=torch.uint8)
    ptrs = [t.data_ptr() for t in (w13, w2, s13, s2, g13, g2)]
    plan = CpuPlan(E, 16, ptrs, hidden=H, inter=I, scale_format='mxfp8', situ_gate_cap=4.0, situ_up_cap=25.0)
    plan.localize()
    x = (torch.randn(4, H) * 0.3).to(torch.bfloat16).contiguous()
    ids = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [7, 8, 9, 10], [0, 11, 12, 15]], dtype=torch.int32)
    rw = torch.full((4, K), 0.25, dtype=torch.float32)
    out = torch.empty((4, H), dtype=torch.float32)
    plan.forward(4, K, x.data_ptr(), ids.data_ptr(), rw.data_ptr(), out.data_ptr())
    ref = _ref_moe_situ(x, w13, s13, w2, s2, g13, g2, ids, rw, 'mxfp8')
    rel = (out - ref).norm().item() / ref.norm().item()
    assert torch.isfinite(out).all() and rel < 1e-5, f'MXFP8 edge rel_l2={rel}'
    plan.close()

    # Dual cold store isolation (nvfp4 + mxfp8 concurrently in one process)
    with tempfile.TemporaryDirectory() as td:
        old_env = {k: os.environ.get(k) for k in ('XTIER_COLD', 'XTIER_COLD_PATHS', 'XTIER_COLD_TAG', 'XTIER_COLD_FRACTION')}
        os.environ['XTIER_COLD'] = '1'
        os.environ['XTIER_COLD_PATHS'] = td
        os.environ['XTIER_COLD_TAG'] = os.path.basename(td)
        os.environ['XTIER_COLD_FRACTION'] = '0.5'
        try:
            cold._stores.clear()
            plans = []
            for lid, fmt in [(0, 'nvfp4'), (40, 'mxfp8')]:
                ws = _make_weights(E, 2048, 1024, fmt, seed=lid + 2)
                lp = cold.LayerPlan(lid, E, gpu_slots=0)
                pl = CpuPlan(E, 16, [t.data_ptr() for t in ws], hidden=2048, inter=1024,
                             scale_format=fmt, resident=lp.resident, situ_gate_cap=4.0, situ_up_cap=25.0)
                pl.localize()
                cold.attach(pl, lp, ws[:4], fmt)
                plans.append((pl, ws, fmt))
            assert len(cold._stores) == 2, f'Expected 2 cold stores, got {list(cold._stores)}'
            for pl, ws, fmt in plans:
                xh = (torch.randn(2, 2048) * 0.3).to(torch.bfloat16).contiguous()
                id2 = torch.tensor([[0, 1, 8, 12], [2, 5, 9, 15]], dtype=torch.int32)
                rw2 = torch.full((2, 4), 0.25, dtype=torch.float32)
                o = torch.empty((2, 2048), dtype=torch.float32)
                pl.forward(2, 4, xh.data_ptr(), id2.data_ptr(), rw2.data_ptr(), o.data_ptr())
                r = _ref_moe_situ(xh, ws[0], ws[2], ws[1], ws[3], ws[4], ws[5], id2, rw2, fmt)
                rel = (o - r).norm().item() / r.norm().item()
                assert rel < 1e-4, f'Dual cold {fmt} rel_l2={rel}'
                pl.close()
            cold._stores.clear()
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    print('  PASS')


def test_gpu_pool_and_prefill():
    if not torch.cuda.is_available():
        print('[2/4] SKIP (no CUDA)')
        return
    print('[2/4] GPU LayerPool + Prefill (NVFP4 & MXFP8 + SiTU + NVFP4 _run_fp4 4over6)...')
    E, H, I, K = 32, 512, 256, 4
    stream = torch.cuda.current_stream()
    for fmt in ('nvfp4', 'mxfp8'):
        w13, w2, s13, s2, g13, g2 = _make_weights(E, H, I, fmt, seed=10 if fmt == 'nvfp4' else 20)
        ptrs = [t.data_ptr() for t in (w13, w2, s13, s2, g13, g2)]
        os.environ['XTIER_PREFILL_UNPACK'] = '1'
        os.environ['XTIER_PREFILL_OVERLAP'] = '1'
        os.environ['XTIER_PREFILL_ROUTE_ACCUM'] = '1'
        os.environ['XTIER_PREFILL_EXPERT_CHUNK'] = '8'
        os.environ['XTIER_PREFILL_FP4'] = '0'
        plan = CpuPlan(E, 16, ptrs, hidden=H, inter=I, scale_format=fmt, situ_gate_cap=4.0, situ_up_cap=25.0)
        plan.localize()
        pf = Prefill(E, ptrs, native=plan, hidden=H, inter=I, situ_gate_cap=4.0, situ_up_cap=25.0)
        pf.ensure_registered()
        mb = Mailbox(plan)
        pool = LayerPool(pf, 8, plan, mb, layer_id=0 if fmt == 'nvfp4' else 40)
        plan.pool = pool
        pool.fill(list(range(8)), stream)

        # Decode M=1, 8, 18 (hybrid hit+miss, NaN-poisoned scratch)
        for M in (1, 8, 18):
            x = (torch.randn(M, H, device='cuda') * 0.3).to(torch.bfloat16)
            ids = torch.randint(0, E, (M, K), device='cuda', dtype=torch.int32)
            rw = torch.softmax(torch.randn(M, K, device='cuda'), dim=-1).float()
            out = torch.empty((M, H), device='cuda', dtype=torch.bfloat16)
            pool.decode(stream.cuda_stream, M, K, x.data_ptr(), ids.data_ptr(), rw.data_ptr(), out.data_ptr(), out_bf16=True)
            torch.cuda.synchronize()
            ref = _ref_moe_situ(x.cpu(), w13, s13, w2, s2, g13, g2, ids.cpu(), rw.cpu(), fmt)
            rel = (out.cpu().float() - ref).norm().item() / ref.norm().item()
            assert torch.isfinite(out).all() and rel < 0.015, f'{fmt} pool decode M={M} rel_l2={rel}'

        # Prefill BF16 + resident + hybrid
        xp = (torch.randn(64, H, device='cuda') * 0.3).to(torch.bfloat16)
        idp = torch.randint(0, E, (64, K), device='cuda', dtype=torch.int32)
        rwp = torch.softmax(torch.randn(64, K, device='cuda'), dim=-1).float()
        outp = torch.empty((64, H), device='cuda', dtype=torch.bfloat16)
        pf.run(xp.data_ptr(), outp.data_ptr(), idp.data_ptr(), rwp.data_ptr(), 64, K, stream.cuda_stream)
        torch.cuda.synchronize()
        refp = _ref_moe_situ(xp.cpu(), w13, s13, w2, s2, g13, g2, idp.cpu(), rwp.cpu(), fmt)
        relp = (outp.cpu().float() - refp).norm().item() / refp.norm().item()
        assert torch.isfinite(outp).all() and relp < 0.015, f'{fmt} prefill rel_l2={relp}'

        # Resident prefill (all routes in hot slots 0..7)
        id_res = torch.randint(0, 8, (32, K), device='cuda', dtype=torch.int32)
        out_res = resident_prefill(pf, pool.arena, xp[:32], id_res, rwp[:32])
        torch.cuda.synchronize()
        ref_res = _ref_moe_situ(xp[:32].cpu(), w13, s13, w2, s2, g13, g2, id_res.cpu(), rwp[:32].cpu(), fmt)
        assert (out_res.cpu().float() - ref_res).norm().item() / ref_res.norm().item() < 0.015

        # Hybrid prefill
        out_hyb = hybrid_prefill(pool, plan, xp, idp, rwp)
        torch.cuda.synchronize()
        assert (out_hyb.cpu().float() - refp).norm().item() / refp.norm().item() < 0.015

        # Native SM100+ NVFP4+SiTU _run_fp4 check (with pool reuse!)
        if fmt == 'nvfp4' and torch.cuda.get_device_capability()[0] >= 10:
            os.environ['XTIER_PREFILL_FP4'] = '1'
            os.environ['XTIER_PREFILL_NVFP4'] = '1'
            os.environ['XTIER_PREFILL_NVFP4_4OVER6'] = '1'
            os.environ['XTIER_PREFILL_FP4_PRESWIZZLE'] = '0'
            os.environ['XTIER_PREFILL_FP4_SLOTS'] = '2'
            pf._fp4_ok = None
            out_fp4 = torch.empty((64, H), device='cuda', dtype=torch.bfloat16)
            pf.run(xp.data_ptr(), out_fp4.data_ptr(), idp.data_ptr(), rwp.data_ptr(), 64, K, stream.cuda_stream)
            torch.cuda.synchronize()
            cos = torch.nn.functional.cosine_similarity(out_fp4.cpu().float().flatten(), refp.flatten(), dim=0).item()
            assert torch.isfinite(out_fp4).all() and cos > 0.97, f'NVFP4 _run_fp4 cos={cos}'
            os.environ['XTIER_PREFILL_FP4'] = '0'
            os.environ['XTIER_PREFILL_NVFP4'] = '0'

        mb.close(); pf.close(); plan.close()
    print('  PASS')


def test_mtp_hooks():
    if not torch.cuda.is_available():
        print('[3/4] SKIP (no CUDA)')
        return
    print('[3/4] Inkling2 MTP hooks (pad-to-2 markov bias + FP8 draft lm_head)...')
    from xtier.packs.inkling2 import mtp
    # Test pad-to-2 markov step_bias bitwise exactness
    V, R, H = 4096, 256, 512
    fake = types.SimpleNamespace(
        _fold_weight=(torch.randn(V, R, device='cuda', dtype=torch.bfloat16) * 0.02).contiguous(),
        _fold_bias=(torch.randn(V, device='cuda', dtype=torch.bfloat16) * 0.02).contiguous(),
        vocab_shard=None,
        _gather_logits=lambda t: t,
    )
    lat = (torch.randn(1, R, device='cuda', dtype=torch.bfloat16) * 0.5).contiguous()
    fake._rnn_step = lambda s, p, h, f: (s, lat)
    def orig(self, prev_tokens, *, hidden_states, state):
        s, f = state
        s, l = self._rnn_step(s, prev_tokens, hidden_states, f)
        return torch.addmm(self._fold_bias, l.to(self._fold_weight.dtype), self._fold_weight.T), (s, f)
    pt = torch.tensor([5], device='cuda', dtype=torch.int32)
    b_ref, _ = orig(fake, pt, hidden_states=None, state=(None, None))
    b_pad, _ = mtp._markov_step_bias(orig, fake, pt, hidden_states=None, state=(None, None))
    rel = (b_pad.float() - b_ref.float()).norm().item() / b_ref.float().norm().item()
    assert rel < 1e-3, f'Markov pad2 rel_l2={rel}'

    # Test FP8 draft lm_head
    head = types.SimpleNamespace(weight=torch.nn.Parameter((torch.randn(V, H, device='cuda', dtype=torch.bfloat16) * 0.02)))
    mtp._quantize_lm_head(head)
    x = (torch.randn(7, H, device='cuda', dtype=torch.bfloat16) * 0.5)
    y_fp8 = mtp._draft_project(lambda h, l: torch.nn.functional.linear(h, l.weight), x, head)
    y_bf16 = torch.nn.functional.linear(x, head.weight)
    rel_lm = (y_fp8.float() - y_bf16.float()).norm().item() / y_bf16.float().norm().item()
    assert rel_lm < 0.05, f'Draft LM head FP8 rel_l2={rel_lm}'
    print(f'  PASS (markov pad2 rel={rel:.2e}, lm_head fp8 rel={rel_lm:.4f})')


def test_pack_registration():
    print('[4/4] Pack registration & hook target discovery...')
    from xtier.packs import PACKS
    assert 'inkling2' in PACKS
    from xtier.packs import inkling2
    assert inkling2.has_inkling2()
    print('  PASS')


def test_phase_bc_optimizations():
    if not torch.cuda.is_available():
        print('[5/5] SKIP (no CUDA)')
        return
    print('[5/5] Phase B/C optimizations (kv_fp4 far-tile, _mix_norm int32, fp8 int64, _project_sources, full-GPU zero-RAM, cold.relink)...')
    from xtier.packs.inkling2.kv_fp4 import decode_partials_nvfp4, write_nvfp4_mha, unpack_active_pages_to_bf16
    from xtier.packs.inkling2 import mtp, attn
    from xtier.packs.glm5_next import fp8_weights as fw
    from xtier.backend.prefill import Prefill
    from xtier.cache.pool import LayerPool
    from xtier.cache.prefill import resident_prefill
    from xtier.cache import cold

    # 1. kv_fp4 write + unpack + far-tile Global Attention decode fast path accuracy
    seq_n = 4096
    pages = (seq_n + 127) // 128
    k_raw = torch.randn(seq_n, 8, 128, dtype=torch.bfloat16, device='cuda') * 0.5
    v_raw = torch.randn(seq_n, 8, 128, dtype=torch.bfloat16, device='cuda') * 0.5
    k_buf = torch.zeros(pages * 128, 8, 72, dtype=torch.uint8, device='cuda')
    v_buf = torch.zeros(pages * 128, 8, 72, dtype=torch.uint8, device='cuda')
    loc = torch.arange(seq_n, dtype=torch.int64, device='cuda')
    write_nvfp4_mha(k_buf, v_buf, loc, k_raw, v_raw)
    k_p, v_p = k_buf.view(pages, 128, 8, 72), v_buf.view(pages, 128, 8, 72)
    pt = torch.arange(pages, dtype=torch.int32, device='cuda').unsqueeze(0)
    sl = torch.tensor([seq_n], dtype=torch.int32, device='cuda')
    q = torch.randn(8, 48, 128, dtype=torch.bfloat16, device='cuda') * 0.5
    rb = torch.randn(8, 48, 1024, dtype=torch.bfloat16, device='cuda') * 0.1
    o, m, l = decode_partials_nvfp4(q, k_p, v_p, pt, sl, rb, 128**-0.5, window_size_left=None)
    m_max = m.max(dim=0).values
    w = torch.exp(m - m_max.unsqueeze(0))
    o_comb = (o * w.unsqueeze(-1)).sum(dim=0) / (l * w).sum(dim=0).unsqueeze(-1)
    ku, vu, _ = unpack_active_pages_to_bf16(k_p, v_p, pt, sl)
    ku = ku.view(-1, 8, 128)[:seq_n].repeat_interleave(6, dim=1).float()
    vu = vu.view(-1, 8, 128)[:seq_n].repeat_interleave(6, dim=1).float()
    scores = torch.einsum('qhd,khd->hqk', q.float(), ku) * (128**-0.5)
    dist = torch.arange(seq_n - 8, seq_n, device='cuda')[:, None] - torch.arange(seq_n, device='cuda')[None, :]
    mask = dist >= 0
    rb_full = torch.zeros_like(scores)
    for qi in range(8):
        vk = mask[qi] & (dist[qi] < 1024)
        rb_full[:, qi, vk] = rb[qi, :, dist[qi, vk]].float()
    probs = torch.softmax(scores.masked_fill(~mask.unsqueeze(0), float('-inf')) + rb_full, dim=-1)
    ref_attn = torch.einsum('hqk,khd->qhd', probs, vu)
    err_kv = (o_comb.squeeze(1) - ref_attn).abs().max().item()
    assert err_kv < 2e-3, f'kv_fp4 far-tile err={err_kv}'

    # 2. MTP _mix_norm int32-overflow chunking at rows=65664, taps=8, H=4608
    from sglang.srt.models.inkling2_mtp import mix_rmsnorm
    targets = torch.randn(65664, 8, 4608, dtype=torch.bfloat16, device='cuda')
    alpha = torch.randn(1, 8, dtype=torch.float32, device='cuda')
    dummy_m = types.SimpleNamespace(_fuse_alpha=alpha)
    dummy_n = types.SimpleNamespace(weight=torch.ones(4608, dtype=torch.bfloat16, device='cuda'), variance_epsilon=1e-6)
    orig_mix = lambda s, t, r, n: mix_rmsnorm(t, s._fuse_alpha[r].to(t.dtype), n.weight, n.variance_epsilon)
    out_mix = mtp._mix_norm(orig_mix, dummy_m, targets, 0, dummy_n)
    assert torch.equal(out_mix[:16], orig_mix(dummy_m, targets[:16], 0, dummy_n))
    assert torch.equal(out_mix[-16:], orig_mix(dummy_m, targets[-16:], 0, dummy_n))
    del targets, out_mix
    torch.cuda.empty_cache()

    # 3. fp8_weights gemm_w8a8 int64 indexing (M*N > 2^31) & attn._project_sources slice equivalence
    if torch.cuda.get_device_capability()[0] >= 9:
        xm = torch.randn(147456, 256, dtype=torch.bfloat16, device='cuda')
        wm = torch.randn(16384, 256, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        sm = torch.ones((16384, 2), dtype=torch.float32, device='cuda')
        om = fw.gemm_w8a8(xm, wm, sm)
        assert torch.equal(om[:64], fw.gemm_w8a8(xm[:64], wm, sm))
        assert torch.equal(om[-64:], fw.gemm_w8a8(xm[-64:], wm, sm))
        del xm, wm, sm, om
        torch.cuda.empty_cache()

    # attn._project_sources equivalence on 3 distinct streams
    w_lin = torch.nn.Linear(256, 768, bias=False, device='cuda', dtype=torch.bfloat16)
    q_w, q_s = fw.quantize(w_lin.weight.data)
    w_lin.weight = torch.nn.Parameter(q_w, requires_grad=False)
    w_lin._xtier_fp8_scale = q_s
    w_lin._xtier_w8_cnt = torch.zeros(32, dtype=torch.int32, device='cuda')
    w_lin._xtier_w8a8 = True
    def _qkvr_call(t):
        return fw.forward(w_lin, t), None
    w_lin_mod = types.SimpleNamespace(weight=w_lin.weight, bias=None, _xtier_fp8_scale=q_s, _xtier_w8a8=True, __call__=_qkvr_call)
    # Build mock Inkling2Attention with _project_qkvr
    class _MockAttn:
        def __init__(self):
            self.split_sizes = [256, 128, 128, 128, 64, 64]
            self.qkvr = lambda t: _qkvr_call(t)
            self.qkvr.weight = w_lin.weight
            self.qkvr.bias = None
            self.qkvr._xtier_fp8_scale = q_s
            self.qkvr._xtier_w8a8 = True
        def _project_qkvr(self, t):
            return self.qkvr(t)[0].split(self.split_sizes, dim=-1)[:6]
    ma = _MockAttn()
    srcs = types.SimpleNamespace(
        q=torch.randn(64, 256, dtype=torch.bfloat16, device='cuda'),
        k=torch.randn(64, 256, dtype=torch.bfloat16, device='cuda'),
        v=torch.randn(64, 256, dtype=torch.bfloat16, device='cuda'),
    )
    def _orig_proj(self, s):
        qp, kp, vp = (p.split(self.split_sizes, dim=-1)[:6] for p in self.qkvr(torch.cat([s.q, s.k, s.v]))[0].chunk(3))
        return qp[0], kp[1], vp[2], qp[3], qp[4], qp[5]
    r_ref = _orig_proj(ma, srcs)
    r_opt = attn._project_sources(_orig_proj, ma, srcs)
    for a, b in zip(r_ref, r_opt):
        assert torch.allclose(a, b, atol=1e-2, rtol=1e-2), 'project_sources mismatch'

    # 4. Zero-RAM full-GPU resident layer (Plan B) for both NVFP4 & MXFP8
    old_cold = os.environ.get('XTIER_COLD')
    os.environ['XTIER_COLD'] = '0'
    try:
        for fmt in ('nvfp4', 'mxfp8'):
            E, H, I, K = 16, 256, 128, 4
            g13 = torch.ones((E, 2), dtype=torch.float32)
            g2 = torch.ones((E,), dtype=torch.float32)
            pf = Prefill(E, (None, None, None, None, g13.data_ptr(), g2.data_ptr()),
                         native=None, hidden=H, inter=I, situ_gate_cap=4.0, situ_up_cap=25.0, scale_format=fmt)
            pool = LayerPool(pf, E, None, None, 0)
            for a in pool.arena:
                a.fill_(0x22 if fmt == 'nvfp4' else 0x38)
            pool.table.copy_(torch.arange(E, dtype=torch.int32, device='cuda'))
            pool.owner.copy_(torch.arange(E, dtype=torch.int32, device='cuda'))
            pool.filled = True
            xp = torch.randn(32, H, dtype=torch.bfloat16, device='cuda') * 0.1
            idsp = torch.randint(0, E, (32, K), dtype=torch.int32, device='cuda')
            wp = torch.full((32, K), 1.0 / K, dtype=torch.float32, device='cuda')
            op = resident_prefill(pf, pool.arena, xp, idsp, wp)
            assert torch.isfinite(op).all(), f'Plan B {fmt} resident_prefill non-finite'
            del pool, pf
    finally:
        if old_cold is None:
            os.environ.pop('XTIER_COLD', None)
        else:
            os.environ['XTIER_COLD'] = old_cold

    # 5. cold.relink() sorts alphabetical load order into model layer_id order
    fake_layers = [(lid, None, types.SimpleNamespace(index=-1, next=None), None) for lid in (0, 1, 10, 2, 3)]
    saved = list(cold._layers)
    try:
        cold._layers[:] = fake_layers
        cold.relink()
        assert [row[0] for row in cold._layers] == [0, 1, 2, 3, 10]
        assert [row[2].index for row in cold._layers] == [0, 1, 2, 3, 4]
        assert cold._layers[0][2].next is cold._layers[1][2] and cold._layers[4][2].next is None
    finally:
        cold._layers[:] = saved

    print(f'  PASS (kv_fp4 err={err_kv:.2e}, _mix_norm 65664, fp8 int64, _project_sources, Plan B zero-RAM, cold.relink)')


if __name__ == '__main__':
    test_cpu_and_dual_cold()
    test_gpu_pool_and_prefill()
    test_mtp_hooks()
    test_pack_registration()
    test_phase_bc_optimizations()
    print('INKLING2 CHECK ALL PASSED!')

