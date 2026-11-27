"""Fast mock tests for xtier.kvstore.bridge (zero server required).

Verifies:
  1. Zero hooks registered when XTIER_KV_OFFLOAD=0 (upgrade convenience red line).
  2. Seven hooks registered when XTIER_KV_OFFLOAD=1 (incl. container cgroup fallback,
     mixed device/host HiCache prefix promise fix, and Python TreeCore pin).
  3. Packed sub-byte MHA normalization (e.g. NVFP4 uint8 declaring unpacked head_dim).
  4. Generic heterogeneous draft compatibility detection (MHA dtype/dim/scales mismatch).
  5. Exact data_ptr() coverage check catches uncovered scale/indexer tensors and disables storage under strict mode.
  6. FullComponent.finalize_match_result_in_tree_core counts only evicted nodes (value is None) and auto-retires.
  7. tree_core_registry.resolve_tree_core_backend pins rust -> python while FullComponent lacks the guard, and auto-retires.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, NamedTuple

import torch

from sglang.srt.mem_cache.device_pool_info import (
    DevicePoolInfo,
    KVBufferLayout,
    KVBuffers,
    KVScaleBuffers,
    KVScaleLayout,
    MHABufferInfo,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from xtier.kvstore import bridge


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


class _FakePool:
    def __init__(self, *, dtype=torch.uint8, head_num=4, head_dim=64, storage_dim=None, with_scales=False):
        sdim = head_dim if storage_dim is None else storage_dim
        k = torch.empty((128, head_num, sdim), dtype=dtype)
        v = torch.empty((128, head_num, sdim), dtype=dtype)
        self.k_buffer = [k]
        self.v_buffer = [v]
        scales = None
        if with_scales:
            ks = torch.empty((128, head_num, 4), dtype=torch.uint8)
            vs = torch.empty((128, head_num, 4), dtype=torch.uint8)
            self.k_scale_buffer = [ks]
            self.v_scale_buffer = [vs]
            scales = KVScaleBuffers(k=(ks,), v=(vs,), layout=KVScaleLayout.TOKEN)
        self._info = DevicePoolInfo(
            pool_name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            layer_ids=(0,),
            buffer_info=MHABufferInfo(
                buffers=KVBuffers(k=(k,), v=(v,), k_data_ptrs=None, v_data_ptrs=None, layout=KVBufferLayout.NHD),
                page_size=64,
                head_num=head_num,
                head_dim=head_dim,
                v_head_dim=head_dim,
                dtype=dtype,
                device=k.device,
                scales=scales,
            ),
        )

    def get_device_pool_infos(self):
        return (self._info,)


class _FakeMatchResult(NamedTuple):
    best_match_node: Any
    last_device_node: Any
    host_hit_length: int


def _make_node(parent, *, value_len: int | None, host_len: int | None, ct: str = 'full'):
    cd = SimpleNamespace(
        value=None if value_len is None else torch.empty((value_len,), dtype=torch.int64),
        host_value=None if host_len is None else torch.empty((host_len,), dtype=torch.int64),
    )
    return SimpleNamespace(parent=parent, component_data={ct: cd})


def main():
    old = os.environ.get('XTIER_KV_OFFLOAD')
    try:
        # 1. Default OFF -> zero hooks
        os.environ['XTIER_KV_OFFLOAD'] = '0'
        reg0 = _DummyRegistry()
        names0 = bridge.install(reg0, _DummyHookType)
        _check(names0 == [] and len(reg0.items) == 0, f'expected 0 hooks when OFF, got {reg0.items}')

        # 2. ON -> 7 generic hooks (incl. cgroup fallback, FullComponent host-hit fix, TreeCore python pin)
        os.environ['XTIER_KV_OFFLOAD'] = '1'
        reg1 = _DummyRegistry()
        names1 = bridge.install(reg1, _DummyHookType)
        _check(names1 == ['kvstore.bridge'] and len(reg1.items) == 7, f'expected 7 hooks when ON, got {reg1.items}')
        cg_hook = reg1.fns['sglang.srt.mem_cache.host_memory._cgroup_memory_headroom']
        def _bad_cg():
            raise ValueError('Cannot resolve cgroup memory path: /../../jupyter-children')
        _check(cg_hook(_bad_cg) is None, 'cgroup hook must swallow relative-path ValueError and return None')
        _check(cg_hook(lambda: 4096) == 4096, 'cgroup hook must pass through normal headroom')

        # 3. Packed sub-byte MHA normalization (e.g. NVFP4 uint8 [m, 2, 128] declaring head_dim=256)
        packed_tgt = _FakePool(dtype=torch.uint8, head_num=2, head_dim=256, storage_dim=128, with_scales=True)
        norm_bi, fixed = bridge._normalize_mha_packed_dims(packed_tgt.get_device_pool_infos()[0].buffer_info)
        _check(fixed and norm_bi.head_dim == 128 and norm_bi.v_head_dim == 128, f'expected packed dim 256->128, got {(fixed, norm_bi)}')
        unpacked_fp8 = _FakePool(dtype=torch.float8_e4m3fn, head_num=2, head_dim=256, storage_dim=256, with_scales=False)
        norm_fp8, fixed_fp8 = bridge._normalize_mha_packed_dims(unpacked_fp8.get_device_pool_infos()[0].buffer_info)
        _check(not fixed_fp8 and norm_fp8.head_dim == 256, 'unpacked fp8 must be unchanged')

        # 4. Heterogeneous draft check (compatible vs dtype/scale/dim/packed mismatch)
        tgt = _FakePool(dtype=torch.uint8, head_dim=64, with_scales=True)
        d_same = _FakePool(dtype=torch.uint8, head_dim=64, with_scales=True)
        d_fp8 = _FakePool(dtype=torch.float8_e4m3fn, head_dim=64, with_scales=False)
        d_bf16 = _FakePool(dtype=torch.bfloat16, head_dim=128, with_scales=False)
        d_unpacked_u8 = _FakePool(dtype=torch.uint8, head_num=2, head_dim=256, storage_dim=256, with_scales=True)

        ok_same, _ = bridge._draft_pack_compatible(tgt, d_same)
        ok_fp8, r_fp8 = bridge._draft_pack_compatible(tgt, d_fp8)
        ok_bf16, r_bf16 = bridge._draft_pack_compatible(tgt, d_bf16)
        ok_packed, r_packed = bridge._draft_pack_compatible(packed_tgt, d_unpacked_u8)
        _check(ok_same, 'identical geometry should be pack-compatible')
        _check(not ok_fp8 and 'geometry' in r_fp8, f'expected mismatch for fp8 draft, got {(ok_fp8, r_fp8)}')
        _check(not ok_bf16 and 'geometry' in r_bf16, f'expected mismatch for bf16 draft, got {(ok_bf16, r_bf16)}')
        _check(not ok_packed and 'geometry' in r_packed, f'expected packed dim mismatch, got {(ok_packed, r_packed)}')

        # 5. Coverage check catches uncovered tensor and disables storage
        if torch.cuda.is_available():
            dev = torch.device('cuda')
            k = torch.empty((128, 4, 64), dtype=torch.uint8, device=dev)
            v = torch.empty((128, 4, 64), dtype=torch.uint8, device=dev)
            ks = torch.empty((128, 4, 4), dtype=torch.uint8, device=dev)
            vs = torch.empty((128, 4, 4), dtype=torch.uint8, device=dev)
            dp = SimpleNamespace(k_buffer=[k], v_buffer=[v], k_scale_buffer=[ks], v_scale_buffer=[vs])

            class _FakeStorage:
                def __init__(self):
                    self.disabled_reason = None

                def disable(self, reason):
                    self.disabled_reason = reason

            class MHATokenToKVPoolHost:
                pass

            # Without scale sidecar -> must detect uncovered ks/vs and disable storage!
            st1 = _FakeStorage()
            hpg1 = SimpleNamespace(entry_map={
                'kv': SimpleNamespace(name='kv', host_pool=MHATokenToKVPoolHost(), device_pool=dp, packed_draft_device_pools=()),
            })
            cache1 = SimpleNamespace(host_pool_group=hpg1, cache_controller=SimpleNamespace(storage_backend=st1))
            ok1 = bridge._verify_coverage(cache1, [('kvcache', dp)], context='unit-missing-scales')
            _check(not ok1 and st1.disabled_reason is not None, 'strict coverage must disable storage when scales uncovered')

            # With scale sidecar binding ks/vs -> must pass!
            st2 = _FakeStorage()
            scale_hp = SimpleNamespace(device_buffers=[ks, vs])
            hpg2 = SimpleNamespace(entry_map={
                'kv': SimpleNamespace(name='kv', host_pool=MHATokenToKVPoolHost(), device_pool=dp, packed_draft_device_pools=()),
                'kv_scale': SimpleNamespace(name='kv_scale', host_pool=scale_hp, device_pool=dp, packed_draft_device_pools=()),
            })
            cache2 = SimpleNamespace(host_pool_group=hpg2, cache_controller=SimpleNamespace(storage_backend=st2))
            ok2 = bridge._verify_coverage(cache2, [('kvcache', dp)], context='unit-complete')
            _check(ok2 and st2.disabled_reason is None, f'complete coverage should pass, got reason={st2.disabled_reason}')

        # 6. Mixed device/host HiCache prefix promise fix (Lsglang #40)
        full_hook = reg1.fns['sglang.srt.mem_cache.unified_cache.components.full.FullComponent.finalize_match_result_in_tree_core']
        root = _make_node(None, value_len=0, host_len=None)
        # Write-through node: resident on GPU (value != None) AND backed on host (host_value != None)
        n_dev = _make_node(root, value_len=3584, host_len=3584)
        # Evicted nodes: only on host (value is None)
        n_ev1 = _make_node(n_dev, value_len=None, host_len=5120)
        n_ev2 = _make_node(n_ev1, value_len=None, host_len=5120)
        fake_comp = SimpleNamespace(component_type='full', tree_core=SimpleNamespace(root_node=root))

        res_mixed = _FakeMatchResult(best_match_node=n_ev2, last_device_node=root, host_hit_length=0)
        orig_fn = FullComponent.finalize_match_result_in_tree_core
        fixed_res = full_hook(orig_fn, fake_comp, res_mixed, None, [], 0)
        _check(
            fixed_res.host_hit_length == 10240,
            f'mixed write-through chain must count only evicted nodes (10240), got {fixed_res.host_hit_length}',
        )
        if bridge._source_needs_full_host_hit_fix(orig_fn):
            buggy_res = orig_fn(fake_comp, res_mixed, None, [], 0)
            _check(
                buggy_res.host_hit_length == 13824,
                f'expected unpatched upstream to overcount 13824, got {buggy_res.host_hit_length}',
            )

        # Resident-only chain with write-through host copies must promise 0 host tokens
        res_resident = _FakeMatchResult(best_match_node=n_dev, last_device_node=root, host_hit_length=0)
        out_resident = full_hook(orig_fn, fake_comp, res_resident, None, [], 0)
        _check(
            out_resident is res_resident and out_resident.host_hit_length == 0,
            f'resident-only write-through node must not add host_hit_length, got {out_resident}',
        )

        # Existing larger host_hit_length from another component is preserved via max()
        res_hi = _FakeMatchResult(best_match_node=n_ev2, last_device_node=root, host_hit_length=20000)
        out_hi = full_hook(orig_fn, fake_comp, res_hi, None, [], 0)
        _check(out_hi.host_hit_length == 20000, f'expected max(20000, 10240)=20000, got {out_hi.host_hit_length}')

        # Auto-retire when upstream adds `.value is None` guard
        def _upstream_patched(self, result, params, value_chunks, best_value_len):
            cd = result.best_match_node.component_data[self.component_type]
            if cd.value is None and cd.host_value is not None:
                return 'UPSTREAM_DELEGATED'
            return result

        _check(
            not bridge._source_needs_full_host_hit_fix(_upstream_patched),
            'source detector must recognize patched .value is None guard',
        )
        _check(
            full_hook(_upstream_patched, fake_comp, res_mixed, None, [], 0) == 'UPSTREAM_DELEGATED',
            'hook must delegate directly once upstream carries .value is None guard',
        )

        # 7. Python TreeCore pin while FullComponent lacks guard (bypasses unpatched Rust full.rs)
        tc_hook = reg1.fns['sglang.srt.mem_cache.unified_cache.tree_core_registry.resolve_tree_core_backend']
        if bridge._full_component_needs_host_hit_fix():
            _check(
                tc_hook(lambda name, params: 'rust', 'rust', None) == 'python',
                'rust TreeCore must pin to python while FullComponent lacks value-is-None guard',
            )
        _check(
            tc_hook(lambda name, params: 'python', 'python', None) == 'python',
            'python backend must pass through unchanged',
        )
        _check(
            tc_hook(lambda name, params: 'custom', 'custom', None) == 'custom',
            'custom backend must pass through unchanged',
        )
        orig_needs = bridge._full_component_needs_host_hit_fix
        try:
            bridge._full_component_needs_host_hit_fix = lambda: False
            _check(
                tc_hook(lambda name, params: 'rust', 'rust', None) == 'rust',
                'rust backend must pass through once upstream FullComponent is fixed',
            )
        finally:
            bridge._full_component_needs_host_hit_fix = orig_needs

        print('PASS: tests/kvstore_bridge_check.py')
    finally:
        if old is None:
            os.environ.pop('XTIER_KV_OFFLOAD', None)
        else:
            os.environ['XTIER_KV_OFFLOAD'] = old


if __name__ == '__main__':
    main()
