"""Unit tests for xtier.kvstore (Python binding + HiCacheStorage v2 semantics: ALL_PAGES, TRAILING_PAGES, BLOCK_GROUPS).

Requires no GPU and no running model server. Uses small temporary O_DIRECT files on NVMe (or XTIER_TEST_KV_DIR).
"""

import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from xtier.kvstore.backend import XTierNVMeStorage


class _FakeHostPool:
    def __init__(self, num_pages: int, page_size: int, bytes_per_page: int, split_kv: bool = False):
        self.page_size = page_size
        self.layout = 'page_first'
        self.dtype = torch.uint8
        self.size_per_token = bytes_per_page // page_size
        self._split_kv = split_kv
        if split_kv:
            half = bytes_per_page // 2
            self.k_buf = torch.empty((num_pages, half), dtype=torch.uint8)
            self.v_buf = torch.empty((num_pages, half), dtype=torch.uint8)
            self.kv_buffer = [self.k_buf, self.v_buf]
        else:
            self.kv_buffer = torch.empty((num_pages, bytes_per_page), dtype=torch.uint8)

    def get_page_buffer_meta(self, indices: torch.Tensor):
        pages = [int(indices[i].item()) // self.page_size for i in range(0, indices.numel(), self.page_size)]
        ptrs, sizes = [], []
        if self._split_kv:
            half = self.k_buf.shape[1]
            for p in pages:
                ptrs.append(self.k_buf[p].data_ptr())
                sizes.append(half)
                ptrs.append(self.v_buf[p].data_ptr())
                sizes.append(half)
        else:
            nb = self.kv_buffer.shape[1]
            for p in pages:
                ptrs.append(self.kv_buffer[p].data_ptr())
                sizes.append(nb)
        return ptrs, sizes


def make_config(paths: list[str], tp_size: int = 1) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name='unit-test',
        extra_config={
            'paths': paths,
            'size_gib': 0,  # overridden below via direct NativeKVStore size when needed
        },
    )


def _check(cond: bool, msg: str) -> None:
    if not cond:
        print(f'FAIL: {msg}', file=sys.stderr)
        sys.exit(1)


def main():
    base = Path(os.environ.get('XTIER_TEST_KV_DIR', '/data/fast/cold1/xtier-kv'))
    tmp = Path(tempfile.mkdtemp(prefix='py-kvcheck-', dir=str(base)))
    try:
        dirs = [str(tmp / f'd{i}') for i in range(4)]

        # 1. TP > 1 gracefully disables
        cfg_tp2 = make_config(dirs, tp_size=2)
        cfg_tp2.extra_config['size_gib'] = 1
        st_tp2 = XTierNVMeStorage(cfg_tp2)
        _check(st_tp2.disabled, 'tp_size=2 must disable v1 offload')
        st_tp2.close()

        # 2. Full v2 multi-pool test: KV (ALL_PAGES), INDEXER (ALL_PAGES), SWA (TRAILING_PAGES), C128 (BLOCK_GROUPS)
        cfg = make_config(dirs, tp_size=1)
        cfg.extra_config = {'paths': dirs, 'size_gib': 1, 'region_mib': 32, 'write_qd': 4}
        st = XTierNVMeStorage(cfg)
        _check(not st.disabled, f'storage disabled unexpectedly: {st._disabled_reason}')

        kv_pool = _FakeHostPool(16, 64, 65536, split_kv=False)
        idx_pool = _FakeHostPool(16, 64, 16384, split_kv=False)
        swa_pool = _FakeHostPool(16, 64, 26624, split_kv=True)   # non-4K split iovs -> bounce path!
        c128_pool = _FakeHostPool(16, 64, 8192, split_kv=False)

        st.register_mem_pool_host(kv_pool)
        st.register_mem_host_pool_v2(idx_pool, PoolName.INDEXER)
        st.register_mem_host_pool_v2(swa_pool, PoolName.SWA)
        st.register_mem_host_pool_v2(c128_pool, PoolName.DEEPSEEK_V4_C128)
        st.on_host_pools_registered()

        keys = [f'{i:064x}' for i in range(1, 9)]  # 8 pages
        kv_pool.kv_buffer[:8].random_(0, 255)
        idx_pool.kv_buffer[:8].random_(0, 255)
        swa_pool.k_buf[:8].random_(0, 255)
        swa_pool.v_buf[:8].random_(0, 255)
        c128_pool.kv_buffer[:8].random_(0, 255)

        ref_kv = kv_pool.kv_buffer[:8].clone()
        ref_idx = idx_pool.kv_buffer[:8].clone()
        ref_swa_k = swa_pool.k_buf[:8].clone()
        ref_swa_v = swa_pool.v_buf[:8].clone()
        ref_c128 = c128_pool.kv_buffer[:8].clone()

        def idx_range(start_page, count):
            return torch.arange(start_page * 64, (start_page + count) * 64, dtype=torch.int64)

        # Write KV pages 0..7
        res_kv = st.batch_set_v1(keys, idx_range(0, 8))
        _check(all(res_kv), f'batch_set_v1 failed: {res_kv}')

        # R1 check via Python: immediately clobber source buffer after batch_set_v1 returns!
        kv_pool.kv_buffer[:8].fill_(0xEE)

        # Write INDEXER pages 0..5 (only 6 pages -> caps ALL_PAGES at 6)
        # Write SWA pages 2,3 and 5 (window=2 -> restorable at prefix_len 4 (pages 2,3) only! Not 5 or 6!)
        # Write C128 block endpoints at page 1 (block 2) and page 3 (block 4) -> BLOCK_GROUPS(block=2) gives 2, 4!
        t_idx = PoolTransfer(name=PoolName.INDEXER, host_indices=idx_range(0, 6), keys=keys[:6], hit_policy=PoolHitPolicy.ALL_PAGES)
        t_swa_a = PoolTransfer(name=PoolName.SWA, host_indices=idx_range(2, 2), keys=keys[2:4], hit_policy=PoolHitPolicy.TRAILING_PAGES)
        t_swa_b = PoolTransfer(name=PoolName.SWA, host_indices=idx_range(5, 1), keys=keys[5:6], hit_policy=PoolHitPolicy.TRAILING_PAGES)
        t_c128 = PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C128,
            host_indices=torch.cat([idx_range(1, 1), idx_range(3, 1)]),
            keys=[keys[1], keys[3]],
            hit_policy=PoolHitPolicy.BLOCK_GROUPS,
            block_pages=2,
        )
        res_v2 = st.batch_set_v2([t_idx, t_swa_a, t_c128])
        _check(
            all(res_v2[PoolName.INDEXER]) and all(res_v2[PoolName.SWA]) and all(res_v2[PoolName.DEEPSEEK_V4_C128]),
            f'set_v2: {res_v2}',
        )
        _check(all(st.batch_set_v2([t_swa_b])[PoolName.SWA]), 'set_v2 swa_b')

        # Query batch_exists_v2 with ALL_PAGES + TRAILING_PAGES(len=2) + BLOCK_GROUPS(block=2)
        q_transfers = [
            PoolTransfer(name=PoolName.INDEXER, hit_policy=PoolHitPolicy.ALL_PAGES),
            PoolTransfer(name=PoolName.SWA, keys=['', ''], hit_policy=PoolHitPolicy.TRAILING_PAGES),
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C128, hit_policy=PoolHitPolicy.BLOCK_GROUPS, block_pages=2),
        ]
        hit = st.batch_exists_v2(keys, q_transfers)
        # KV=8, INDEXER=6 -> 1..6; SWA(trailing=2) -> {4} (pages 2,3 present; page 4 missing so 5,6 invalid);
        # C128(block=2) -> {2, 4}; intersection = [4]!
        _check(
            hit.kv_hit_pages == 4 and hit.restorable_prefix_pages == [4],
            f'batch_exists_v2 mismatch: got kv_hit={hit.kv_hit_pages}, restorable={hit.restorable_prefix_pages}, extra={hit.extra_pool_hit_pages}',
        )
        _check(hit.extra_pool_hit_pages[PoolName.INDEXER] == 6, f'indexer hit={hit.extra_pool_hit_pages}')
        _check(hit.extra_pool_hit_pages[PoolName.SWA] == 4, f'swa hit={hit.extra_pool_hit_pages}')
        _check(hit.extra_pool_hit_pages[PoolName.DEEPSEEK_V4_C128] == 4, f'c128 hit={hit.extra_pool_hit_pages}')

        # Read back into clean slots 8..15 and verify byte-exact equality!
        kv_pool.kv_buffer[8:16].zero_()
        idx_pool.kv_buffer[8:14].zero_()
        swa_pool.k_buf[8:10].zero_()
        swa_pool.v_buf[8:10].zero_()
        c128_pool.kv_buffer[8:10].zero_()

        _check(all(st.batch_get_v1(keys, idx_range(8, 8))), 'batch_get_v1')
        _check(torch.equal(kv_pool.kv_buffer[8:16], ref_kv), 'KV roundtrip mismatch (R1!)')

        get_res = st.batch_get_v2([
            PoolTransfer(name=PoolName.INDEXER, host_indices=idx_range(8, 6), keys=keys[:6]),
            PoolTransfer(name=PoolName.SWA, host_indices=idx_range(8, 2), keys=keys[2:4]),
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C128, host_indices=idx_range(8, 2), keys=[keys[1], keys[3]]),
        ])
        _check(
            all(get_res[PoolName.INDEXER]) and all(get_res[PoolName.SWA]) and all(get_res[PoolName.DEEPSEEK_V4_C128]),
            f'get_res={get_res}',
        )
        _check(torch.equal(idx_pool.kv_buffer[8:14], ref_idx[:6]), 'INDEXER roundtrip mismatch')
        _check(
            torch.equal(swa_pool.k_buf[8:10], ref_swa_k[2:4]) and torch.equal(swa_pool.v_buf[8:10], ref_swa_v[2:4]),
            'SWA bounce roundtrip mismatch',
        )
        _check(
            torch.equal(c128_pool.kv_buffer[8], ref_c128[1]) and torch.equal(c128_pool.kv_buffer[9], ref_c128[3]),
            'C128 roundtrip mismatch',
        )

        # Clear test
        st.clear()
        _check(st.batch_exists(keys) == 0, 'clear did not reset exists')
        st.close()

        # 3. LogicalHostPool anchor (e.g. DeepSeek-V4 where KV is 0-byte anchor and sidecars hold data)
        from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool

        st_log = XTierNVMeStorage(cfg)
        _check(not st_log.disabled, 'logical anchor storage disabled')
        logical_kv = LogicalHostPool(16 * 64, 64, layout='page_first')
        c1_pool = _FakeHostPool(16, 64, 32768, split_kv=False)
        c1_pool.kv_buffer[:4].random_(0, 255)
        ref_c1 = c1_pool.kv_buffer[:4].clone()
        st_log.register_mem_pool_host(logical_kv)
        st_log.register_mem_host_pool_v2(c1_pool, PoolName.DEEPSEEK_V4_C1)
        st_log.on_host_pools_registered()
        _check(all(st_log.batch_set_v1(keys[:4], idx_range(0, 4))), 'logical batch_set_v1 must return True')
        t_c1 = PoolTransfer(name=PoolName.DEEPSEEK_V4_C1, host_indices=idx_range(0, 4), keys=keys[:4], hit_policy=PoolHitPolicy.ALL_PAGES)
        _check(all(st_log.batch_set_v2([t_c1])[PoolName.DEEPSEEK_V4_C1]), 'logical sidecar set_v2')
        hit_l = st_log.batch_exists_v2(keys[:6], [PoolTransfer(name=PoolName.DEEPSEEK_V4_C1, hit_policy=PoolHitPolicy.ALL_PAGES)])
        _check(hit_l.kv_hit_pages == 4 and hit_l.restorable_prefix_pages == [1, 2, 3, 4], f'logical exists_v2 mismatch: {hit_l}')
        c1_pool.kv_buffer[8:12].zero_()
        _check(all(st_log.batch_get_v1(keys[:4], idx_range(8, 4))), 'logical batch_get_v1 must return True')
        _check(all(st_log.batch_get_v2([PoolTransfer(name=PoolName.DEEPSEEK_V4_C1, host_indices=idx_range(8, 4), keys=keys[:4])])[PoolName.DEEPSEEK_V4_C1]), 'logical sidecar get_v2')
        _check(torch.equal(c1_pool.kv_buffer[8:12], ref_c1), 'logical sidecar roundtrip mismatch')
        st_log.close()

        print('PASS: tests/kvstore_check.py (TP guard, R1, ALL_PAGES, TRAILING_PAGES, BLOCK_GROUPS, LogicalHostPool, direct+bounce, clear)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
