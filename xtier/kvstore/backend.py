"""SGLang HiCacheStorage v2 backend backed by XTier's multi-drive O_DIRECT + io_uring KV store.

Loaded via SGLang's standard dynamic backend mechanism (zero SGLang source patches):
  --hicache-storage-backend dynamic
  --hicache-storage-backend-extra-config '{"backend_name":"xtier_nvme","module_path":"xtier.kvstore.backend","class_name":"XTierNVMeStorage"}'
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
    l3_page_span,
)
from xtier import kvstore as kv_cfg
from xtier.kvstore.native import NativeKVStore

logger = logging.getLogger(__name__)


class XTierNVMeStorage(HiCacheStorage):
    """Model-agnostic HiCacheStorage v2 backend over striped O_DIRECT/io_uring NVMe pool files."""

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        extra_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.storage_config = storage_config
        self.extra_config = dict(storage_config.extra_config or {})
        if extra_kwargs:
            self.extra_config.update(extra_kwargs)

        self.enable_storage_metrics = bool(storage_config.enable_storage_metrics)
        self.mem_pool_host = None
        self.registered_pools: dict[str, Any] = {}
        self._pool_ids: dict[str, int] = {str(PoolName.KV): 0}
        self._pool_direct_meta: dict[str, bool] = {}
        self._pool_logical: dict[str, bool] = {}
        self._kv_anchor_is_logical: bool = False
        self._pool_lock = threading.Lock()
        self._io_log_counter = 0

        self._metrics_lock = threading.Lock()
        self._prefetch_pgs: list[int] = []
        self._backup_pgs: list[int] = []
        self._prefetch_bandwidth: list[float] = []
        self._backup_bandwidth: list[float] = []

        self._disabled_reason: Optional[str] = None
        self._store: Optional[NativeKVStore] = None

        if storage_config.tp_size > 1:
            self.disable(f'v1 supports tp_size=1 only (got tp_size={storage_config.tp_size})')
            return

        dirs = self._resolve_paths()
        total_gib = int(self.extra_config.get('size_gib', kv_cfg.size_gib()))
        bytes_per_drive = (total_gib * (1024 ** 3)) // max(1, len(dirs))
        region_mib = int(self.extra_config.get('region_mib', kv_cfg.region_mib()))
        write_qd = int(self.extra_config.get('write_qd', kv_cfg.write_qd()))
        read_qd = int(self.extra_config.get('read_qd', kv_cfg.read_qd()))
        relocate_pct = int(self.extra_config.get('relocate_pct', kv_cfg.relocate_pct()))

        try:
            self._store = NativeKVStore(
                dirs,
                bytes_per_drive,
                region_bytes=region_mib * 1024 * 1024,
                write_qd=write_qd,
                read_qd=read_qd,
                relocate_pct=relocate_pct,
                pin_numa=True,
            )
            atexit.register(self.close)
            logger.info(
                'xtier kvstore: opened %d drives x %.1f GiB (total %.1f GiB, region=%d MiB, write_qd=%d, read_qd=%d)',
                len(dirs),
                bytes_per_drive / (1024 ** 3),
                (bytes_per_drive * len(dirs)) / (1024 ** 3),
                region_mib,
                write_qd,
                read_qd,
            )
        except Exception as e:
            self.disable(f'failed to open native NVMe store: {e}')

    def _resolve_paths(self) -> list[str]:
        p = self.extra_config.get('paths')
        if isinstance(p, (list, tuple)):
            return [str(x) for x in p if x]
        if isinstance(p, str) and p:
            return [x for x in p.split(':') if x]
        return kv_cfg.paths()

    @property
    def disabled(self) -> bool:
        return self._disabled_reason is not None or self._store is None

    def disable(self, reason: str) -> None:
        if self._disabled_reason is None:
            self._disabled_reason = reason
            logger.warning('xtier kvstore: disabling NVMe KV offload: %s', reason)
        if self._store is not None:
            self._store.close()
            self._store = None

    def _pool_id(self, name: str) -> int:
        key = str(name)
        pid = self._pool_ids.get(key)
        if pid is not None:
            return pid
        with self._pool_lock:
            pid = self._pool_ids.get(key)
            if pid is None:
                pid = len(self._pool_ids)
                self._pool_ids[key] = pid
            return pid

    # ----- host pool registration -------------------------------------------

    def register_mem_pool_host(self, mem_pool_host) -> None:
        self.mem_pool_host = mem_pool_host
        self.register_mem_host_pool_v2(mem_pool_host, PoolName.KV)

    def register_mem_host_pool_v2(self, host_pool, host_pool_name) -> None:
        name = str(host_pool_name)
        self.registered_pools[name] = host_pool
        self._pool_id(name)
        is_logical = (
            getattr(host_pool, 'kv_buffer', None) is None
            and getattr(host_pool, 'layer_num', 1) == 0
            and getattr(host_pool, 'size_per_token', 1) == 0
        )
        self._pool_logical[name] = is_logical
        if name == str(PoolName.KV):
            self.mem_pool_host = host_pool
            self._kv_anchor_is_logical = is_logical
        if is_logical:
            self._pool_direct_meta[name] = False
            logger.info(
                'xtier kvstore: registered pool=%s (logical anchor, 0 payload bytes, page_span=%d)',
                name, l3_page_span(host_pool),
            )
            return
        self._pool_direct_meta[name] = self._probe_direct_meta(name, host_pool)

    def _probe_direct_meta(self, name: str, host_pool) -> bool:
        try:
            span = l3_page_span(host_pool)
            probe = torch.arange(span, dtype=torch.int64)
            meta = host_pool.get_page_buffer_meta(probe)
            if meta is None:
                return False
            ptrs, sizes = meta
            if not ptrs or len(ptrs) != len(sizes):
                return False
            aligned = all((int(p) % 4096 == 0 and int(s) % 4096 == 0) for p, s in zip(ptrs, sizes))
            total_b = sum(int(s) for s in sizes)
            logger.info(
                'xtier kvstore: registered pool=%s layout=%s page_span=%d iovs/page=%d page_bytes=%d direct_4k=%s',
                name, getattr(host_pool, 'layout', '?'), span, len(ptrs), total_b, aligned,
            )
            return True
        except Exception as e:
            logger.info('xtier kvstore: pool=%s get_page_buffer_meta unavailable (%s); will use flat copy fallback', name, e)
            return False

    def on_host_pools_registered(self) -> None:
        total_pinned = 0
        seen = set()
        for name, pool in self.registered_pools.items():
            for attr in ('kv_buffer', 'index_k_with_scale_buffer', 'conv_buffer', 'temporal_buffer'):
                t = getattr(pool, attr, None)
                if isinstance(t, torch.Tensor) and id(t) not in seen:
                    seen.add(id(t))
                    total_pinned += t.numel() * t.element_size()
                elif isinstance(t, (list, tuple)):
                    for sub in t:
                        if isinstance(sub, torch.Tensor) and id(sub) not in seen:
                            seen.add(id(sub))
                            total_pinned += sub.numel() * sub.element_size()
        logger.info(
            'xtier kvstore: %d host pools registered, staging pinned memory=%.2f GiB, status=%s',
            len(self.registered_pools),
            total_pinned / (1024 ** 3),
            'disabled (' + self._disabled_reason + ')' if self.disabled else 'active',
        )

    # ----- metrics ----------------------------------------------------------

    def _record_metrics(self, *, prefetch: bool, pages: int, num_bytes: int, elapsed_s: float) -> None:
        if not self.enable_storage_metrics or pages <= 0 or elapsed_s <= 0:
            return
        bw = (num_bytes / 1e9) / elapsed_s
        with self._metrics_lock:
            if prefetch:
                self._prefetch_pgs.append(pages)
                self._prefetch_bandwidth.append(bw)
            else:
                self._backup_pgs.append(pages)
                self._backup_bandwidth.append(bw)

    def get_stats(self):
        if not self.enable_storage_metrics:
            return None
        from sglang.srt.observability.metrics_collector import StorageMetrics

        with self._metrics_lock:
            out = StorageMetrics(
                prefetch_pgs=self._prefetch_pgs,
                backup_pgs=self._backup_pgs,
                prefetch_bandwidth=self._prefetch_bandwidth,
                backup_bandwidth=self._backup_bandwidth,
            )
            self._prefetch_pgs = []
            self._backup_pgs = []
            self._prefetch_bandwidth = []
            self._backup_bandwidth = []
        return out

    def native_stats(self) -> dict[str, int]:
        return self._store.stats() if self._store is not None else {}

    # ----- existence queries (ALL_PAGES, TRAILING_PAGES, BLOCK_GROUPS) ------

    def exists(self, key: str) -> bool:
        if self.disabled:
            return False
        return self._store.exists(self._pool_id(str(PoolName.KV)), [key])[0]

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        if self.disabled or not keys:
            return 0
        bm = self._store.exists(self._pool_id(str(PoolName.KV)), keys)
        for i, ok in enumerate(bm):
            if not ok:
                return i
        return len(keys)

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """Exact v2 multi-pool prefix existence check supporting ALL_PAGES, TRAILING_PAGES, and BLOCK_GROUPS."""
        if self.disabled or not keys:
            return PoolTransferResult(0, {}, [])

        if self._kv_anchor_is_logical:
            kv_pages = len(keys) if pool_transfers else 0
        else:
            kv_pages = self.batch_exists(keys, extra_info)
        if kv_pages == 0:
            return PoolTransferResult(0, {}, [])

        hit_count: dict[Any, int] = {PoolName.KV: kv_pages}
        restorable: list[int] = list(range(1, kv_pages + 1))
        prefix_keys = keys[:kv_pages]

        for transfer in pool_transfers or []:
            if not restorable:
                break
            name = str(transfer.name)
            pid = self._pool_id(name)
            page_exists = self._store.exists(pid, prefix_keys)

            boundary = 0
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                try:
                    boundary = page_exists.index(False)
                except ValueError:
                    boundary = kv_pages
                pool_restorable_set = set(range(1, boundary + 1))
            elif transfer.hit_policy == PoolHitPolicy.BLOCK_GROUPS:
                # Reference: sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py:1271
                block = max(1, int(transfer.block_pages))
                complete = 0
                for idx in range(block - 1, kv_pages, block):
                    if not page_exists[idx]:
                        break
                    complete += 1
                boundary = complete * block
                pool_restorable_set = set(range(block, boundary + 1, block))
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                pool_restorable_set = set()
                for prefix_len in range(kv_pages, 0, -1):
                    start_i = max(0, prefix_len - trailing)
                    if all(page_exists[i] for i in range(start_i, prefix_len)):
                        pool_restorable_set.add(prefix_len)
                        if boundary == 0:
                            boundary = prefix_len
            else:
                raise ValueError(f'Unsupported pool hit policy: {transfer.hit_policy}')

            if boundary:
                hit_count[transfer.name] = boundary
            restorable = [p for p in restorable if p in pool_restorable_set]

        final_pages = restorable[-1] if restorable else 0
        return PoolTransferResult(final_pages, hit_count, restorable)

    # ----- page buffer extraction & batched transfers -----------------------

    def _build_direct_entries(
        self, pool_name: str, host_pool, keys: List[str], host_indices: torch.Tensor
    ) -> Optional[tuple[list[tuple[int, str, list[tuple[int, int]]]], int]]:
        n = len(keys)
        if n == 0:
            return [], 0
        ptrs, sizes = host_pool.get_page_buffer_meta(host_indices)
        if not ptrs or len(ptrs) != len(sizes) or len(ptrs) % n != 0:
            logger.error(
                'xtier kvstore: invalid get_page_buffer_meta for pool=%s (%d ptrs for %d keys)',
                pool_name, len(ptrs) if ptrs else 0, n,
            )
            return None
        per_page = len(ptrs) // n
        pid = self._pool_id(pool_name)
        entries = []
        total_bytes = 0
        for i in range(n):
            base = i * per_page
            iovs = [(int(ptrs[base + j]), int(sizes[base + j])) for j in range(per_page) if int(sizes[base + j]) > 0]
            total_bytes += sum(sz for _, sz in iovs)
            entries.append((pid, keys[i], iovs))
        return entries, total_bytes

    def _transfer_flat_fallback(
        self, pool_name: str, host_pool, keys: List[str], host_indices: torch.Tensor, *, write: bool
    ) -> tuple[List[bool], int]:
        span = l3_page_span(host_pool)
        pid = self._pool_id(pool_name)
        tensors = []
        entries = []
        total_bytes = 0
        for i, key in enumerate(keys):
            page_off = int(host_indices[i * span].item())
            if write:
                t = host_pool.get_data_page(page_off, flat=True).contiguous()
            else:
                t = host_pool.get_dummy_flat_data_page().contiguous()
            tensors.append((page_off, t))
            nb = t.numel() * t.element_size()
            total_bytes += nb
            entries.append((pid, key, [(t.data_ptr(), nb)]))
        res = self._store.put_batch(entries) if write else self._store.get_batch(entries)
        if not write:
            for ok, (page_off, t) in zip(res, tensors):
                if ok:
                    host_pool.set_from_flat_data_page(page_off, t)
        return res, total_bytes

    def _transfer_pool_v1(
        self, pool_name: str, host_pool, keys: List[str], host_indices: Optional[torch.Tensor], *, write: bool
    ) -> List[bool]:
        if self.disabled or not keys:
            return [False] * len(keys)
        if self._pool_logical.get(pool_name, False):
            return [True] * len(keys)
        span = l3_page_span(host_pool)
        expected = len(keys) * span
        if host_indices is None or host_indices.numel() != expected:
            logger.error(
                'xtier kvstore: %s indices mismatch for pool=%s: expected %d, got %s',
                'write' if write else 'read', pool_name, expected,
                host_indices.numel() if host_indices is not None else None,
            )
            return [False] * len(keys)

        t0 = time.perf_counter()
        if self._pool_direct_meta.get(pool_name, False):
            built = self._build_direct_entries(pool_name, host_pool, keys, host_indices)
            if built is None:
                return [False] * len(keys)
            entries, total_bytes = built
            res = self._store.put_batch(entries) if write else self._store.get_batch(entries)
        else:
            res, total_bytes = self._transfer_flat_fallback(pool_name, host_pool, keys, host_indices, write=write)

        ok_pages = sum(1 for x in res if x)
        elapsed = time.perf_counter() - t0
        self._record_metrics(
            prefetch=not write,
            pages=ok_pages,
            num_bytes=total_bytes if ok_pages == len(keys) else 0,
            elapsed_s=elapsed,
        )
        if not write and ok_pages > 0 and elapsed > 0:
            self._io_log_counter += 1
            if self._io_log_counter <= 5 or (self._io_log_counter % 50 == 0):
                logger.info(
                    'xtier kvstore: prefetch_v1 #%d pool=%s pages=%d bytes=%.2f MiB bw=%.2f GB/s',
                    self._io_log_counter, pool_name, ok_pages,
                    total_bytes / (1024 ** 2), (total_bytes / 1e9) / elapsed,
                )
        return res

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self._transfer_pool_v1(str(PoolName.KV), self.mem_pool_host, keys, host_indices, write=False)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self._transfer_pool_v1(str(PoolName.KV), self.mem_pool_host, keys, host_indices, write=True)

    def _batch_io_v2(self, transfers: List[PoolTransfer], *, write: bool) -> dict[str, List[bool]]:
        if self.disabled:
            return {t.name: [False] * len(t.keys or []) for t in transfers}

        t0 = time.perf_counter()
        results: dict[str, List[bool]] = {}
        direct_entries: list[tuple[int, str, list[tuple[int, int]]]] = []
        direct_slices: list[tuple[str, int, int]] = []
        total_bytes = 0

        for transfer in transfers:
            name = str(transfer.name)
            keys = transfer.keys or []
            if not keys:
                results[transfer.name] = []
                continue
            if self._pool_logical.get(name, False):
                results[transfer.name] = [True] * len(keys)
                continue
            host_pool = self.registered_pools.get(name)
            if host_pool is None:
                logger.error('xtier kvstore: unregistered host pool %s', name)
                results[transfer.name] = [False] * len(keys)
                continue
            span = l3_page_span(host_pool)
            expected = len(keys) * span
            host_indices = transfer.host_indices
            if host_indices is None or host_indices.numel() != expected:
                logger.error(
                    'xtier kvstore: %s v2 indices mismatch for pool=%s: expected %d, got %s',
                    'write' if write else 'read', name, expected,
                    host_indices.numel() if host_indices is not None else None,
                )
                results[transfer.name] = [False] * len(keys)
                continue

            if self._pool_direct_meta.get(name, False):
                built = self._build_direct_entries(name, host_pool, keys, host_indices)
                if built is None:
                    results[transfer.name] = [False] * len(keys)
                    continue
                entries, nb = built
                start = len(direct_entries)
                direct_entries.extend(entries)
                direct_slices.append((transfer.name, start, len(direct_entries)))
                total_bytes += nb
            else:
                res, nb = self._transfer_flat_fallback(name, host_pool, keys, host_indices, write=write)
                results[transfer.name] = res
                total_bytes += nb

        if direct_entries:
            flat_res = self._store.put_batch(direct_entries) if write else self._store.get_batch(direct_entries)
            for orig_name, start, end in direct_slices:
                results[orig_name] = flat_res[start:end]

        pages = max((sum(1 for ok in r if ok) for r in results.values()), default=0)
        elapsed = time.perf_counter() - t0
        self._record_metrics(prefetch=not write, pages=pages, num_bytes=total_bytes, elapsed_s=elapsed)
        if not write and pages > 0 and elapsed > 0:
            self._io_log_counter += 1
            if self._io_log_counter <= 5 or (self._io_log_counter % 50 == 0):
                logger.info(
                    'xtier kvstore: prefetch_v2 #%d pools=%s pages=%d bytes=%.2f MiB bw=%.2f GB/s',
                    self._io_log_counter, [str(t.name) for t in transfers], pages,
                    total_bytes / (1024 ** 2), (total_bytes / 1e9) / elapsed,
                )
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, write=False)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, write=True)

    # ----- legacy v0 ABC methods --------------------------------------------

    def get(self, key: str, target_location: Optional[torch.Tensor] = None, target_sizes=None):
        if self.disabled or target_location is None:
            return None
        t = target_location if target_location.is_contiguous() else target_location.contiguous()
        ok = self._store.get_batch([(0, key, [(t.data_ptr(), t.numel() * t.element_size())])])[0]
        if not ok:
            return None
        if t is not target_location:
            target_location.copy_(t)
        return target_location

    def batch_get(self, keys: List[str], target_locations=None, target_sizes=None):
        if self.disabled or not keys or target_locations is None:
            return [None] * len(keys)
        if self._kv_anchor_is_logical:
            return list(target_locations)
        entries = []
        tensors = []
        for k, loc in zip(keys, target_locations):
            t = loc if loc.is_contiguous() else loc.contiguous()
            tensors.append((loc, t))
            entries.append((0, k, [(t.data_ptr(), t.numel() * t.element_size())]))
        res = self._store.get_batch(entries)
        out = []
        for ok, (loc, t) in zip(res, tensors):
            if ok:
                if t is not loc:
                    loc.copy_(t)
                out.append(loc)
            else:
                out.append(None)
        return out

    def set(self, key: str, value: Optional[torch.Tensor] = None, target_location=None, target_sizes=None) -> bool:
        if self.disabled or value is None:
            return False
        t = value if value.is_contiguous() else value.contiguous()
        return self._store.put_batch([(0, key, [(t.data_ptr(), t.numel() * t.element_size())])])[0]

    def batch_set(self, keys: List[str], values=None, target_locations=None, target_sizes=None) -> bool:
        if self.disabled or not keys or values is None:
            return False
        if self._kv_anchor_is_logical:
            return True
        entries = []
        keep = []
        for k, v in zip(keys, values):
            t = v if v.is_contiguous() else v.contiguous()
            keep.append(t)
            entries.append((0, k, [(t.data_ptr(), t.numel() * t.element_size())]))
        return all(self._store.put_batch(entries))

    def clear(self) -> bool:
        if self._store is not None:
            self._store.clear()
            logger.info('xtier kvstore: cleared volatile NVMe KV index')
        return True

    def close(self) -> None:
        if self._store is not None:
            st = self._store.stats()
            logger.info(
                'xtier kvstore: closing (puts=%d, hits=%d, misses=%d, recycled=%d, relocated=%d, direct=%d, bounce=%d)',
                st.get('puts_ok', 0), st.get('gets_hit', 0), st.get('gets_miss', 0),
                st.get('regions_recycled', 0), st.get('hot_relocated', 0),
                st.get('direct_ios', 0), st.get('bounce_ios', 0),
            )
            self._store.close()
            self._store = None
