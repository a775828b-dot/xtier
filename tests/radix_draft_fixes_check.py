"""Tests for Wave 1 platform fixes in xtier.platform (zero server required).

Verifies:
  1. SWA radix insert full span (Lsglang #37): keeps swa_branching_seqlen & evicted_seqlen
     metadata while returning None so radix insert spans the full request.
  2. Draft prewarm helper grids (_draft_prewarm_batch_sizes, _draft_track_intervals).
  3. XTIER_DRAFT_PREWARM opt-out cleanly skips EAGLE_CAPTURE_GRAPHS hook registration.
  4. Live GPU eager draft kernel prewarm (_prewarm_eager_draft_kernels) executes every
     specialization without mutating req_to_token or triggering Xid 31 / illegal address.
  5. Live GPU QSA draft-graph metadata prewarm (_prewarm_qsa_draft_graph_metadata) compiles
     and launches on cloned buffers, leaving live graph metadata bit-for-bit untouched.
  6. _eagle_capture_cuda_graphs ordering: eager prewarm before original(), QSA prewarm after.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.model_executor.forward_mode import ForwardMode
from xtier import platform


def _check(cond: bool, msg: str) -> None:
    if not cond:
        print(f'FAIL: {msg}', file=sys.stderr)
        sys.exit(1)


class _DummyRegistry:
    def __init__(self):
        self.items = []
        self.fns = {}

    def register(self, target, fn, htype):
        self.items.append((target, htype))
        self.fns[target] = fn


class _DummyHookType:
    AROUND = 'AROUND'
    AFTER = 'AFTER'


class _FakeInsertParams:
    def __init__(self):
        self.evicted = {}
        self.swa_branching_seqlen = None

    def set_evicted_seqlen(self, component_type, seqlen):
        self.evicted[component_type] = seqlen


def _make_swa_component(*, is_mamba: bool = False, is_eagle: bool = False):
    return SimpleNamespace(
        component_type='swa',
        window_sub_pool='win_pool',
        cache=SimpleNamespace(is_mamba_enabled=is_mamba),
        tree_core=SimpleNamespace(is_eagle=is_eagle),
    )


def _make_swa_req(*, branching_seqlen, protected_len: int = 0, evicted_seqlen: int = 512):
    return SimpleNamespace(
        swa_branching_seqlen=branching_seqlen,
        kv=SimpleNamespace(
            cache_protected_len=protected_len,
            get_evicted_seqlen=lambda pool: evicted_seqlen,
        ),
    )


def _make_qsa_step_metadata(bs: int, max_pages: int, ratio: int, pool, device, fill_val: int):
    indexer_metadata = SimpleNamespace(
        token_to_kv_pool=pool,
        graph_compressed_lengths=torch.full((bs,), fill_val, dtype=torch.int32, device=device),
        graph_write_locs=torch.full((bs,), fill_val + 1, dtype=torch.int32, device=device),
        graph_compressed_page_table=torch.full((bs, max_pages), fill_val + 2, dtype=torch.int32, device=device),
        decode_logical_positions=torch.full((bs,), fill_val + 3, dtype=torch.int32, device=device),
        pending_ring_slots=torch.full((bs,), fill_val + 4, dtype=torch.int64, device=device),
        graph_ring_group_locs=torch.full((bs * ratio,), fill_val + 5, dtype=torch.int32, device=device),
        graph_prefix_lengths=torch.full((bs,), fill_val + 6, dtype=torch.int32, device=device),
    )
    return SimpleNamespace(
        sequence_lengths=torch.full((bs,), fill_val + 7, dtype=torch.int32, device=device),
        row_req_pool_indices=torch.full((bs,), fill_val + 8, dtype=torch.int32, device=device),
        indexer_metadata=indexer_metadata,
    )


def _metadata_buffers(meta):
    im = meta.indexer_metadata
    return (
        meta.sequence_lengths,
        meta.row_req_pool_indices,
        im.graph_compressed_lengths,
        im.graph_write_locs,
        im.graph_compressed_page_table,
        im.decode_logical_positions,
        im.pending_ring_slots,
        im.graph_ring_group_locs,
        im.graph_prefix_lengths,
    )


def main():
    old_prewarm = os.environ.get('XTIER_DRAFT_PREWARM')
    try:
        # 1. Registration checks (default ON vs XTIER_DRAFT_PREWARM=0 opt-out)
        os.environ.pop('XTIER_DRAFT_PREWARM', None)
        reg_on = _DummyRegistry()
        fixes_on = platform.install(reg_on, _DummyHookType)
        _check('swa-radix-full-insert' in fixes_on, f'missing swa-radix-full-insert in {fixes_on}')
        _check(platform.SWA_PREPARE_CACHE in reg_on.fns, 'SWA_PREPARE_CACHE not registered')
        _check('eagle-draft-prewarm' in fixes_on, f'missing eagle-draft-prewarm in {fixes_on}')
        _check(platform.EAGLE_CAPTURE_GRAPHS in reg_on.fns, 'EAGLE_CAPTURE_GRAPHS not registered')

        os.environ['XTIER_DRAFT_PREWARM'] = '0'
        reg_off = _DummyRegistry()
        fixes_off = platform.install(reg_off, _DummyHookType)
        _check('swa-radix-full-insert' in fixes_off, 'SWA fix must remain registered when draft prewarm is OFF')
        _check('eagle-draft-prewarm' not in fixes_off, 'eagle-draft-prewarm must skip when XTIER_DRAFT_PREWARM=0')
        _check(platform.EAGLE_CAPTURE_GRAPHS not in reg_off.fns, 'EAGLE_CAPTURE_GRAPHS must not register when OFF')
        os.environ['XTIER_DRAFT_PREWARM'] = '1'

        # 2. SWA radix insert full span (Lsglang #37)
        orig_swa = SWAComponent.prepare_for_caching_req
        comp_std = _make_swa_component(is_mamba=False, is_eagle=False)
        req_branch = _make_swa_req(branching_seqlen=4096, protected_len=0, evicted_seqlen=512)

        ip_orig = _FakeInsertParams()
        raw_ret = orig_swa(comp_std, req_branch, ip_orig, 8192, True)
        _check(raw_ret == 4096, f'expected unpatched SWA prepare_for_caching_req to truncate at 4096, got {raw_ret}')

        ip_hook = _FakeInsertParams()
        hook_ret = platform._swa_prepare_for_caching_req(orig_swa, comp_std, req_branch, ip_hook, 8192, True)
        _check(hook_ret is None, f'hooked SWA prepare_for_caching_req must return None, got {hook_ret}')
        _check(ip_hook.swa_branching_seqlen == 4096, f'swa_branching_seqlen must be preserved, got {ip_hook.swa_branching_seqlen}')
        _check(ip_hook.evicted.get('swa') == 512, f'evicted_seqlen must be preserved, got {ip_hook.evicted}')

        # EAGLE bigram mode (+1 raw token) also suppresses truncation while preserving boundary
        comp_eagle = _make_swa_component(is_mamba=False, is_eagle=True)
        ip_eagle = _FakeInsertParams()
        _check(orig_swa(comp_eagle, req_branch, _FakeInsertParams(), 8192, True) == 4097, 'expected raw eagle len 4097')
        ret_eagle = platform._swa_prepare_for_caching_req(orig_swa, comp_eagle, req_branch, ip_eagle, 8192, True)
        _check(ret_eagle is None and ip_eagle.swa_branching_seqlen == 4096, 'EAGLE SWA branch must not truncate')

        # Mamba hybrid or no branching_seqlen -> unchanged None and evicted_seqlen preserved
        comp_mamba = _make_swa_component(is_mamba=True, is_eagle=False)
        ip_mamba = _FakeInsertParams()
        ret_mamba = platform._swa_prepare_for_caching_req(orig_swa, comp_mamba, req_branch, ip_mamba, 8192, True)
        _check(
            ret_mamba is None and ip_mamba.swa_branching_seqlen is None and ip_mamba.evicted.get('swa') == 512,
            f'Mamba hybrid path must stay untouched, got ret={ret_mamba}, ip={ip_mamba.__dict__}',
        )

        # Non-branching return (hypothetical future path without swa_branching_seqlen) passes through
        ip_other = _FakeInsertParams()
        ret_other = platform._swa_prepare_for_caching_req(
            lambda self, req, ip, n, fin: 2048, comp_std, req_branch, ip_other, 8192, True
        )
        _check(ret_other == 2048, f'non-branch return must pass through unchanged, got {ret_other}')

        # 3. Prewarm helper grids
        _check(platform._draft_prewarm_batch_sizes(None) == (1,), 'None cap -> (1,)')
        _check(platform._draft_prewarm_batch_sizes(0) == (1,), '0 cap -> (1,)')
        _check(platform._draft_prewarm_batch_sizes(4) == (1, 2, 3, 4), 'cap=4 -> 1..4')
        _check(platform._draft_prewarm_batch_sizes(16) == tuple(range(1, 17)), 'cap=16 -> 1..16')
        _check(
            platform._draft_prewarm_batch_sizes(64) == tuple(range(1, 17)) + (32, 64),
            f'cap=64 mismatch: {platform._draft_prewarm_batch_sizes(64)}',
        )
        _check(
            platform._draft_prewarm_batch_sizes(512) == tuple(range(1, 17)) + (32, 64, 128, 256),
            f'cap=512 mismatch: {platform._draft_prewarm_batch_sizes(512)}',
        )
        _check(platform._draft_track_intervals(None) == (0,), 'page_size=None -> (0,)')
        ti = platform._draft_track_intervals(64)
        _check(0 in ti, f'track intervals must always include 0, got {ti}')

        # 4 & 5. Live GPU tests (eager draft prewarm + QSA draft-graph metadata prewarm)
        if torch.cuda.is_available():
            dev = torch.device('cuda')
            req_to_token = torch.arange(4 * 512, dtype=torch.int32, device=dev).reshape(4, 512)
            r2t_ref = req_to_token.clone()

            # topk != 1 skips cleanly
            _check(
                platform._prewarm_eager_draft_kernels(
                    req_to_token, topk=2, num_steps=3, num_draft_tokens=4, batch_sizes=(1,), track_intervals=(0,)
                ) == 0,
                'topk != 1 must skip eager prewarm',
            )

            # Live CUDA eager draft prewarm (bs=1,2,4; steps=3; width=4; track_intervals=(0,256))
            launches = platform._prewarm_eager_draft_kernels(
                req_to_token,
                topk=1,
                num_steps=3,
                num_draft_tokens=4,
                batch_sizes=(1, 2, 4),
                track_intervals=(0, 256),
            )
            torch.cuda.synchronize()
            # Per bs: 2 windows * 2 mrope (4) + 2*2*2 build_chain_tree (8) + 1 extend_len + 2 extend_layout + 2 verify_commit = 17
            _check(launches == 3 * 17, f'expected 51 eager draft kernel launches, got {launches}')
            _check(torch.equal(req_to_token, r2t_ref), 'eager draft prewarm must not mutate req_to_token')

            # Live CUDA QSA draft-graph metadata prewarm on cloned buffers
            pool = SimpleNamespace(qsa_compress_ratio=4, qsa_compressed_page_size=16)
            max_pages = 8
            ratio = 4
            b0_meta = {
                (ForwardMode.DECODE, 1): _make_qsa_step_metadata(1, max_pages, ratio, pool, dev, 10),
                (ForwardMode.DECODE, 2): _make_qsa_step_metadata(2, max_pages, ratio, pool, dev, 30),
            }
            b1_meta = {
                (ForwardMode.DECODE, 1): _make_qsa_step_metadata(1, max_pages, ratio, pool, dev, 50),
                (ForwardMode.DECODE, 2): _make_qsa_step_metadata(2, max_pages, ratio, pool, dev, 70),
            }
            snapshots = [
                tuple(t.clone() for t in _metadata_buffers(m))
                for d in (b0_meta, b1_meta)
                for m in d.values()
            ]
            fake_qsa = SimpleNamespace(
                _draft_graph_metadata_args={},
                attn_backends=[
                    SimpleNamespace(req_to_token=req_to_token, _cuda_graph_metadata=b0_meta),
                    SimpleNamespace(req_to_token=req_to_token, _cuda_graph_metadata=b1_meta),
                ],
            )
            qsa_launches = platform._prewarm_qsa_draft_graph_metadata(fake_qsa)
            torch.cuda.synchronize()
            _check(qsa_launches == 2, f'expected 2 QSA metadata launches (bs=1,2), got {qsa_launches}')
            after_metas = [m for d in (b0_meta, b1_meta) for m in d.values()]
            for meta, snap in zip(after_metas, snapshots):
                for cur, ref in zip(_metadata_buffers(meta), snap):
                    _check(torch.equal(cur, ref), 'QSA draft-graph metadata prewarm mutated live graph buffer!')
            _check(torch.equal(req_to_token, r2t_ref), 'QSA prewarm must not mutate req_to_token')

            # 6. _eagle_capture_cuda_graphs ordering check (eager before original, QSA after original)
            events = []
            orig_eager = platform._prewarm_eager_draft_kernels
            orig_qsa = platform._prewarm_qsa_draft_graph_metadata
            try:
                def _stub_eager(*a, **kw):
                    events.append('eager')
                    return 1

                def _stub_qsa(backend):
                    events.append(('qsa', backend is fake_qsa))
                    return 2

                platform._prewarm_eager_draft_kernels = _stub_eager
                platform._prewarm_qsa_draft_graph_metadata = _stub_qsa

                worker = SimpleNamespace(
                    topk=1,
                    speculative_num_steps=3,
                    speculative_num_draft_tokens=4,
                    draft_runner=SimpleNamespace(
                        server_args=SimpleNamespace(max_running_requests=4),
                        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
                    ),
                    target_worker=SimpleNamespace(
                        model_runner=SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(page_size=64)),
                    ),
                    draft_attn_backend=None,
                )

                def _fake_capture(self):
                    events.append('capture')
                    self.draft_attn_backend = fake_qsa
                    return 'CAPTURED'

                out = platform._eagle_capture_cuda_graphs(_fake_capture, worker)
                _check(out == 'CAPTURED', f'expected original return value, got {out}')
                _check(
                    events == ['eager', 'capture', ('qsa', True)],
                    f'unexpected capture hook ordering: {events}',
                )
            finally:
                platform._prewarm_eager_draft_kernels = orig_eager
                platform._prewarm_qsa_draft_graph_metadata = orig_qsa

        print('PASS: tests/radix_draft_fixes_check.py')
    finally:
        if old_prewarm is None:
            os.environ.pop('XTIER_DRAFT_PREWARM', None)
        else:
            os.environ['XTIER_DRAFT_PREWARM'] = old_prewarm


if __name__ == '__main__':
    main()
