"""Model-agnostic SGLang HiCache bridge for XTier NVMe KV offload.

Registered ONLY when ``XTIER_KV_OFFLOAD=1``. Contains zero model-name branches; every hook triggers solely on generic
device KV pool properties and auto-retires (no-op) if upstream SGLang adds native support:

  1. Heterogeneous packed draft fallback (``EagleDraftWorkerBase._build_hicache_draft_plan``):
     If SGLang chooses ``HiCacheDraftMode.PACKED`` for a draft whose KV buffer geometry/dtype/scales differs from the
     target pool, fall back to SGLang's own ``HiCacheDraftMode.SIDECAR`` path.
  2. MHA per-token scale bridge (``build_kv_host_pool`` + ``attach_hybrid_pool_to_unified_cache``):
     If ``MHATokenToKVPoolHost`` rejects ``KVScaleLayout.TOKEN``, build the host KV pool over raw K/V payload buffers
     and attach matching K/V scale sidecars via ``DeepSeekV4PagedHostPool(page_aligned_only=True)``.
  3. Missing declared sidecars (``attach_hybrid_pool_to_unified_cache``):
     If the anchor device pool's ``get_device_pool_infos()`` declares sidecars (e.g. ``PoolName.INDEXER`` under SWA)
     absent from ``host_pool_group.entry_map``, build and register them via SGLang's own ``_build_sidecar_host_pool``.
  4. Exact ``data_ptr()`` coverage verifier (``attach_hybrid_pool_to_unified_cache`` & ``register_hicache_draft_pools``):
     Verify every persistent device KV/indexer/scale/state/draft buffer is bound by a registered host pool; if any
     tensor is uncovered and ``XTIER_KV_STRICT_COVERAGE=1``, disable ``XTierNVMeStorage`` rather than corrupting KV.
  5. Container cgroup headroom fallback (``host_memory._cgroup_memory_headroom``):
     In nested-cgroup containers (e.g. Colab / Docker) where ``/proc/self/cgroup`` contains relative ``..`` segments,
     SGLang's ``_cgroup_headroom`` raises ``ValueError`` which ``available_host_memory_bytes`` does not catch; catch
     ``ValueError`` and return ``None`` so host memory sizing falls back cleanly to ``psutil.virtual_memory().available``.
  6. Mixed device/host HiCache prefix promise fix (``FullComponent.finalize_match_result_in_tree_core``):
     Under ``write_through``, a node still resident on GPU (``cd.value is not None``) can also carry ``cd.host_value``.
     Counting such nodes in ``kv_host_hit`` over-promises H2D load-back tokens (``LOAD_BACK`` stops at the first
     device-resident node) and aborts the request / crashes the scheduler (Lsglang #40). Count only evicted nodes
     (``cd.value is None and cd.host_value is not None``).
  7. Python ``TreeCore`` pin while unpatched Rust ``full.rs`` would bypass hook #6
     (``tree_core_registry.resolve_tree_core_backend``):
     When ``XTIER_KV_OFFLOAD=1`` and ``FullComponent`` still lacks the ``value is None`` guard, route ``rust`` to
     ``python`` so environments with ``cargo``/prebuilt ``sglang_radix_tree`` cannot bypass hook #6 via ``full.rs``.
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any

import torch

from xtier import kvstore as kv_cfg

logger = logging.getLogger(__name__)

_ASSEMBLER = 'sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler'
_SPEC_BASE = 'sglang.srt.speculative.base_spec_worker'
_KV_BUILDER = 'sglang.srt.mem_cache.kv_cache_builder'
_HOST_MEM = 'sglang.srt.mem_cache.host_memory'
_FULL_COMPONENT = 'sglang.srt.mem_cache.unified_cache.components.full'
_TREE_CORE_REGISTRY = 'sglang.srt.mem_cache.unified_cache.tree_core_registry'


def _source_needs_full_host_hit_fix(fn: Any) -> bool:
    """Return True when FullComponent.finalize_match_result_in_tree_core omits the ``value is None`` guard."""
    target = fn
    seen: set[int] = set()
    while hasattr(target, '__wrapped__') and id(target) not in seen:
        seen.add(id(target))
        target = target.__wrapped__
    try:
        fn_src = inspect.getsource(target)
    except (OSError, TypeError):
        return True
    return re.search(r'\.value\s+is\s+None\b', fn_src) is None


def _full_component_needs_host_hit_fix() -> bool:
    try:
        from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
    except Exception:
        return True
    return _source_needs_full_host_hit_fix(FullComponent.finalize_match_result_in_tree_core)


# ----- Generic geometry & coverage inspection helpers -----------------------


def _primary_kv_info(pool: Any):
    if pool is None or not hasattr(pool, 'get_device_pool_infos'):
        return None
    try:
        infos = pool.get_device_pool_infos()
    except Exception:
        return None
    if not infos:
        return None
    for info in infos:
        if str(info.pool_name) == 'kv':
            return info
    return infos[0]


def _pool_info_names(pool: Any) -> set[str]:
    if pool is None or not hasattr(pool, 'get_device_pool_infos'):
        return set()
    try:
        return {str(info.pool_name) for info in pool.get_device_pool_infos()}
    except Exception:
        return set()


def _resolve_full_view(pool: Any):
    router = getattr(pool, 'layer_router', None)
    if router is None:
        return pool
    try:
        wrapped = router.wrapped_view()
        if wrapped is not None:
            return wrapped
    except Exception:
        pass
    try:
        from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
        if router.sub_pool(IdSpaceKind.SLIDING_WINDOW) is not None and router.routes_to(IdSpaceKind.FULL):
            return router.view(IdSpaceKind.FULL)
    except Exception:
        pass
    return pool


def _normalize_mha_packed_dims(bi: Any) -> tuple[Any, bool]:
    """Align MHABufferInfo head_dim/v_head_dim to physical storage width for packed sub-byte KV (e.g. NVFP4)."""
    import msgspec
    from sglang.srt.mem_cache.device_pool_info import KVBufferLayout, MHABufferInfo

    if not isinstance(bi, MHABufferInfo):
        return bi, False
    k_bufs = getattr(bi.buffers, 'k', None) or ()
    v_bufs = getattr(bi.buffers, 'v', None) or ()
    if not k_bufs or not v_bufs:
        return bi, False
    k0, v0 = k_bufs[0], v_bufs[0]
    if k0.ndim != 3 or v0.ndim != 3 or k0.dtype != bi.dtype or v0.dtype != bi.dtype:
        return bi, False
    layout = getattr(bi.buffers, 'layout', None)
    if layout == KVBufferLayout.NHD:
        head_axis, dim_axis = 1, 2
    elif layout == KVBufferLayout.HND:
        head_axis, dim_axis = 0, 2
    else:
        return bi, False
    if k0.shape[head_axis] != bi.head_num or v0.shape[head_axis] != bi.head_num:
        return bi, False
    actual_k_dim = int(k0.shape[dim_axis])
    actual_v_dim = int(v0.shape[dim_axis])
    if actual_k_dim == bi.head_dim and actual_v_dim == bi.v_head_dim:
        return bi, False
    return msgspec.structs.replace(bi, head_dim=actual_k_dim, v_head_dim=actual_v_dim), True


def _draft_pack_compatible(target_pool: Any, draft_pool: Any) -> tuple[bool, str]:
    """Return (compatible, reason) checking generic KV buffer geometry/dtype/scales between target and draft."""
    from sglang.srt.mem_cache.device_pool_info import MHABufferInfo, MLABufferInfo

    t_view = _resolve_full_view(target_pool)
    d_view = _resolve_full_view(draft_pool)

    t_names = _pool_info_names(t_view)
    d_names = _pool_info_names(d_view)
    if t_names and d_names and not d_names.issubset(t_names):
        return False, f'pool sets incompatible: draft={sorted(d_names)} not subset of target={sorted(t_names)}'

    t_info = _primary_kv_info(t_view)
    d_info = _primary_kv_info(d_view)
    if t_info is None or d_info is None:
        return True, ''
    # If target and draft primary info represent different sub-pool kinds (e.g. compressed KV vs SWA),
    # compare matching pool names instead of mismatched first entries.
    if str(t_info.pool_name) != str(d_info.pool_name):
        t_by_name = {str(i.pool_name): i for i in t_view.get_device_pool_infos()}
        if str(d_info.pool_name) in t_by_name:
            t_info = t_by_name[str(d_info.pool_name)]
        else:
            return False, f'draft pool {d_info.pool_name} absent from target {sorted(t_by_name)}'

    tb, db = t_info.buffer_info, d_info.buffer_info
    if type(tb) is not type(db):
        return False, f'buffer_info type mismatch: draft={type(db).__name__} vs target={type(tb).__name__}'

    if isinstance(tb, MHABufferInfo):
        tb_norm, _ = _normalize_mha_packed_dims(tb)
        db_norm, _ = _normalize_mha_packed_dims(db)
        t_geom = (tb_norm.head_num, tb_norm.head_dim, tb_norm.v_head_dim, tb_norm.dtype, tb_norm.device)
        d_geom = (db_norm.head_num, db_norm.head_dim, db_norm.v_head_dim, db_norm.dtype, db_norm.device)
        if t_geom != d_geom:
            return False, f'MHA geometry mismatch: draft={d_geom} vs target={t_geom}'
        t_sc = tb.scales.layout if tb.scales is not None else None
        d_sc = db.scales.layout if db.scales is not None else None
        if t_sc != d_sc:
            return False, f'MHA scale layout mismatch: draft={d_sc} vs target={t_sc}'
    elif isinstance(tb, MLABufferInfo):
        if getattr(tb, 'dtype', None) != getattr(db, 'dtype', None):
            return False, f'MLA dtype mismatch: draft={getattr(db, "dtype", None)} vs target={getattr(tb, "dtype", None)}'
        t_dim = getattr(t_view, 'kv_cache_dim', None)
        d_dim = getattr(d_view, 'kv_cache_dim', None)
        if t_dim is not None and d_dim is not None and t_dim != d_dim:
            return False, f'MLA kv_cache_dim mismatch: draft={d_dim} vs target={t_dim}'

    return True, ''


def _collect_cuda_tensors(obj: Any, prefix: str, out: dict[int, tuple[str, tuple[int, ...], str]], seen_objs: set[int]) -> None:
    """Recursively collect persistent CUDA tensors from buffer_info / tensor containers."""
    if obj is None or id(obj) in seen_objs:
        return
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda and obj.nbytes > 0:
            out.setdefault(obj.data_ptr(), (prefix, tuple(obj.shape), str(obj.dtype)))
        return
    if isinstance(obj, (list, tuple)):
        seen_objs.add(id(obj))
        for i, item in enumerate(obj):
            _collect_cuda_tensors(item, f'{prefix}[{i}]', out, seen_objs)
        return
    if isinstance(obj, dict):
        seen_objs.add(id(obj))
        for k, v in obj.items():
            _collect_cuda_tensors(v, f'{prefix}.{k}', out, seen_objs)


def _extract_buffer_info_tensors(buffer_info: Any, prefix: str, out: dict[int, tuple[str, tuple[int, ...], str]]) -> None:
    seen: set[int] = set()
    if buffer_info is None:
        return
    # MHABufferInfo: buffers.k, buffers.v, scales.k, scales.v
    bufs = getattr(buffer_info, 'buffers', None)
    if bufs is not None:
        if isinstance(bufs, (list, tuple, torch.Tensor)):
            _collect_cuda_tensors(bufs, f'{prefix}.buffers', out, seen)
        else:
            _collect_cuda_tensors(getattr(bufs, 'k', None), f'{prefix}.k', out, seen)
            _collect_cuda_tensors(getattr(bufs, 'v', None), f'{prefix}.v', out, seen)
    scales = getattr(buffer_info, 'scales', None)
    if scales is not None:
        _collect_cuda_tensors(getattr(scales, 'k', None), f'{prefix}.scale_k', out, seen)
        _collect_cuda_tensors(getattr(scales, 'v', None), f'{prefix}.scale_v', out, seen)
    # MLABufferInfo / IndexKeyBufferInfo / DSV4PagedBufferInfo / MambaStateBufferInfo
    for attr in (
        'kv_buffer', 'kv_buffers', 'k_buffer', 'v_buffer',
        'index_k_with_scale_buffer', 'pages',
        'temporal', 'conv', 'extra_checkpoint_buffers',
    ):
        val = getattr(buffer_info, attr, None)
        if val is not None:
            _collect_cuda_tensors(val, f'{prefix}.{attr}', out, seen)


def collect_device_pool_ptrs(device_pool: Any, label: str = 'kvcache') -> dict[int, tuple[str, tuple[int, ...], str]]:
    """Collect persistent CUDA tensor data_ptr()s owned by a device KV/state pool (model-agnostic)."""
    out: dict[int, tuple[str, tuple[int, ...], str]] = {}
    visited: set[int] = set()

    def walk_pool(pool: Any, pfx: str) -> None:
        if pool is None or id(pool) in visited:
            return
        visited.add(id(pool))

        # 1. Unified / routed sub-pools
        router = getattr(pool, 'layer_router', None)
        if router is not None:
            try:
                wrapped = router.wrapped_view()
                if wrapped is not None and wrapped is not pool:
                    walk_pool(wrapped, f'{pfx}.wrapped')
            except Exception:
                pass
            try:
                subs = router.sub_pools()
                if len(subs) > 1:
                    for s in subs:
                        sub_name = getattr(s, 'name', s)
                        try:
                            v = router.view(sub_name)
                            if v is not None and v is not pool:
                                walk_pool(v, f'{pfx}.{sub_name}')
                        except Exception:
                            pass
            except Exception:
                pass

        for sub_attr in ('full_kv_pool', 'swa_kv_pool', 'index_k_pool', 'mamba_pool'):
            sub = getattr(pool, sub_attr, None)
            if sub is not None and sub is not pool:
                walk_pool(sub, f'{pfx}.{sub_attr}')

        # 2. Structured DevicePoolInfo declarations (preferred authoritative source)
        if hasattr(pool, 'get_device_pool_infos'):
            try:
                for info in pool.get_device_pool_infos():
                    _extract_buffer_info_tensors(info.buffer_info, f'{pfx}.{info.pool_name}', out)
            except Exception:
                pass

        # 3. DeepSeek-V4 extra_host_regions (if present)
        if hasattr(pool, 'extra_host_regions'):
            try:
                seen: set[int] = set()
                for reg in pool.extra_host_regions():
                    _collect_cuda_tensors(reg.device_buffers, f'{pfx}.{reg.pool_name}', out, seen)
            except Exception:
                pass

        # 4. Raw persistent attributes (catches pools without full DevicePoolInfo, e.g. Mamba/SSM, K/V scales, indexers)
        seen_raw: set[int] = set()
        for attr in (
            'k_buffer', 'v_buffer', 'k_scale_buffer', 'v_scale_buffer',
            'native_k_scale_buffer', 'native_v_scale_buffer',
            'kv_buffer', 'index_k_with_scale_buffer', 'index_k_buffer',
            'conv_state', 'temporal_state', 'ssm_state', 'conv_buffer', 'temporal_buffer',
        ):
            val = getattr(pool, attr, None)
            if val is not None:
                _collect_cuda_tensors(val, f'{pfx}.{attr}', out, seen_raw)

    walk_pool(device_pool, label)
    return out


def collect_host_bound_ptrs(host_pool_group: Any) -> set[int]:
    """Collect CUDA device data_ptr()s actually bound for H2D/D2H transfer by registered host pools."""
    ptrs: set[int] = set()
    if host_pool_group is None:
        return ptrs

    def add_ptrs_from_obj(obj: Any, seen: set[int]) -> None:
        if obj is None or id(obj) in seen:
            return
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and obj.nbytes > 0:
                ptrs.add(obj.data_ptr())
            return
        if isinstance(obj, (list, tuple)):
            seen.add(id(obj))
            for item in obj:
                add_ptrs_from_obj(item, seen)
            return
        if isinstance(obj, dict):
            seen.add(id(obj))
            for v in obj.values():
                add_ptrs_from_obj(v, seen)

    def add_ptr_table(tbl: Any) -> None:
        if isinstance(tbl, torch.Tensor) and tbl.dtype in (torch.int64, torch.uint64):
            try:
                for p in tbl.detach().cpu().tolist():
                    if isinstance(p, int) and p > 0:
                        ptrs.add(p)
            except Exception:
                pass

    for entry in host_pool_group.entry_map.values():
        hp = entry.host_pool
        # Unwrap PageInterleaveHostPool or wrapper host pools
        candidates = [hp]
        for wrap_attr in ('inner', '_inner', 'host_pool', 'base_pool'):
            inner = getattr(hp, wrap_attr, None)
            if inner is not None:
                candidates.append(inner)

        seen: set[int] = set()
        for h in candidates:
            # Explicit device tensor lists bound by host pools (DeepSeekV4PagedHostPool, Indexer, Mamba, PageEnvelope)
            for attr in (
                'device_buffers', 'device_k_buffers', 'device_v_buffers',
                'device_kv_buffers', 'device_index_buffers',
                'temporal_device_layers', 'conv_device_layers', 'slot_state_device_tensors',
                '_device_buffers', '_k_buffers', '_v_buffers',
            ):
                add_ptrs_from_obj(getattr(h, attr, None), seen)

            # Pointer tables stored as int64 tensors on host/device (MHATokenToKVPoolHost, MLATokenToKVPoolHost, etc.)
            for ptr_attr in (
                'device_ptrs', 'device_k_ptrs', 'device_v_ptrs',
                'k_ptrs_device', 'v_ptrs_device', 'kv_ptrs_device',
                'temporal_device_ptrs', 'conv_device_ptrs',
                'src_ptrs', 'dst_ptrs',
            ):
                add_ptr_table(getattr(h, ptr_attr, None))

            # Structured buffer_info bound on host pool (note: MHA host ignores TOKEN scales unless handled!)
            for bi_attr in ('buffer_info', '_buffer_info', 'device_buffer_info'):
                bi = getattr(h, bi_attr, None)
                if bi is not None:
                    bufs = getattr(bi, 'buffers', None)
                    if bufs is not None:
                        add_ptrs_from_obj(bufs if isinstance(bufs, (list, tuple, torch.Tensor)) else (getattr(bufs, 'k', None), getattr(bufs, 'v', None)), seen)
                    for b_attr in ('kv_buffer', 'kv_buffers', 'pages', 'temporal', 'conv', 'extra_checkpoint_buffers'):
                        add_ptrs_from_obj(getattr(bi, b_attr, None), seen)
                    # Check MXFP8 host pools that natively transfer scales
                    if 'MXFP8' in type(h).__name__ or 'ModelSlim' in type(h).__name__:
                        sc = getattr(bi, 'scales', None)
                        if sc is not None:
                            add_ptrs_from_obj((getattr(sc, 'k', None), getattr(sc, 'v', None)), seen)

            # Mamba host pools hold direct references to mamba_pool / conv_state / temporal_state
            if 'mamba' in str(entry.name).lower() or 'Mamba' in type(h).__name__:
                dp = entry.device_pool
                if dp is not None:
                    for m_attr in ('conv_state', 'temporal_state', 'ssm_state', 'conv_buffer', 'temporal_buffer'):
                        add_ptrs_from_obj(getattr(dp, m_attr, None), seen)

            # UnifiedPageEnvelopeHostPool backs its entire arena slice (all sub-tensors inside [base, base+bytes))
            if 'Envelope' in type(h).__name__:
                dp = entry.device_pool
                if dp is not None:
                    dp_map = collect_device_pool_ptrs(dp, str(entry.name))
                    ptrs.update(dp_map.keys())

            # Generic fallback for MHA/MLA host pools: they store k_buffer/v_buffer/kv_buffer refs from device_pool
            # (NOTE: deliberately excludes k_scale_buffer/v_scale_buffer so missing scale sidecars are caught!)
            cls_name = type(h).__name__
            if ('MHATokenToKVPool' in cls_name or 'MLATokenToKVPool' in cls_name) and entry.device_pool is not None:
                pools_to_bind = (entry.device_pool, *(entry.packed_draft_device_pools or ()))
                for dp in pools_to_bind:
                    for b_attr in ('k_buffer', 'v_buffer', 'kv_buffer'):
                        add_ptrs_from_obj(getattr(dp, b_attr, None), seen)
                    if 'MXFP8' in cls_name:
                        for s_attr in ('k_scale_buffer', 'v_scale_buffer'):
                            add_ptrs_from_obj(getattr(dp, s_attr, None), seen)
            if 'IndexerPoolHost' in cls_name and entry.device_pool is not None:
                pools_to_bind = (entry.device_pool, *(entry.packed_draft_device_pools or ()))
                for dp in pools_to_bind:
                    for i_attr in ('index_k_with_scale_buffer', 'index_k_buffer'):
                        add_ptrs_from_obj(getattr(dp, i_attr, None), seen)
                    if hasattr(dp, 'get_device_pool_infos'):
                        try:
                            for info in dp.get_device_pool_infos():
                                if str(info.pool_name) == str(entry.name):
                                    tmp: dict[int, tuple[str, tuple[int, ...], str]] = {}
                                    _extract_buffer_info_tensors(info.buffer_info, str(entry.name), tmp)
                                    ptrs.update(tmp.keys())
                        except Exception:
                            pass

    return ptrs


def _verify_coverage(cache: Any, device_pools: list[tuple[str, Any]], *, context: str) -> bool:
    """Verify every persistent CUDA buffer in device_pools is bound by cache.host_pool_group."""
    hpg = getattr(cache, 'host_pool_group', None)
    cc = getattr(cache, 'cache_controller', None)
    storage = getattr(cc, 'storage_backend', None) if cc is not None else None
    if hpg is None or storage is None:
        return True

    expected: dict[int, tuple[str, tuple[int, ...], str]] = {}
    for label, dp in device_pools:
        if dp is not None:
            expected.update(collect_device_pool_ptrs(dp, label))

    bound = collect_host_bound_ptrs(hpg)
    uncovered = [(ptr, meta) for ptr, meta in expected.items() if ptr not in bound]
    if not uncovered:
        logger.info(
            'xtier kvstore: coverage check OK (%s): %d/%d device KV tensors bound across %d host pools %s',
            context, len(expected), len(expected), len(hpg.entry_map), sorted(str(k) for k in hpg.entry_map),
        )
        return True

    sample = ', '.join(f'{m[0]}{m[1]}:{m[2]}@{hex(p)}' for p, m in uncovered[:6])
    more = f' (+{len(uncovered) - 6} more)' if len(uncovered) > 6 else ''
    msg = (
        f'coverage check ({context}): {len(uncovered)}/{len(expected)} device KV tensors uncovered '
        f'(host pools={sorted(str(k) for k in hpg.entry_map)}): {sample}{more}'
    )
    if kv_cfg.strict_coverage() and hasattr(storage, 'disable'):
        storage.disable(msg)
    else:
        logger.warning('xtier kvstore: %s (strict_coverage=0, continuing)', msg)
    return False


# ----- Gap bridges ----------------------------------------------------------


def _attach_missing_declared_sidecars(cache: Any) -> int:
    """Gap 1: attach any IndexKeyBufferInfo sidecar declared by anchor.device_pool missing from entry_map."""
    from sglang.srt.mem_cache.device_pool_info import IndexKeyBufferInfo
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.hybrid_cache.host_pool_config import prepare_host_pool_config
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _build_sidecar_host_pool,
        build_pool_entry,
    )

    hpg = getattr(cache, 'host_pool_group', None)
    cc = getattr(cache, 'cache_controller', None)
    if hpg is None or cc is None:
        return 0
    anchor = getattr(hpg, 'anchor_entry', None)
    if anchor is None or anchor.device_pool is None or not hasattr(anchor.device_pool, 'get_device_pool_infos'):
        return 0

    try:
        infos = anchor.device_pool.get_device_pool_infos()
    except Exception:
        return 0
    missing_names = [
        info.pool_name for info in infos
        if info.pool_name != PoolName.KV
        and isinstance(info.buffer_info, IndexKeyBufferInfo)
        and info.pool_name not in hpg.entry_map
    ]
    if not missing_names:
        return 0

    page_size = getattr(anchor.host_pool, 'page_size', None) or getattr(cache, 'page_size', 64)
    target_layers = getattr(anchor.device_pool, 'layer_num', cc.transfer_layer_id_max)
    full_layer_mapping = {
        lid: mapped
        for lid in range(cc.transfer_layer_id_max)
        if (mapped := anchor.layer_mapper(lid)) is not None and mapped < target_layers
    }
    config = prepare_host_pool_config(
        root=anchor.device_pool,
        full_layer_mapping=full_layer_mapping,
        transfer_layer_id_max=cc.transfer_layer_id_max,
        transfer_page_size=page_size,
        packed_draft_device_pools=tuple(anchor.packed_draft_device_pools or ()),
    )
    added = 0
    for sc in config.sidecars:
        if sc.pool_name in hpg.entry_map:
            continue
        host_pool = _build_sidecar_host_pool(
            config=sc, root=config.root, root_host_pool=anchor.host_pool,
        )
        entry = build_pool_entry(
            name=sc.pool_name,
            host_pool=host_pool,
            device_pool=config.root,
            layer_mapping=sc.layer_mapping,
            transfer_layer_id_max=config.transfer_layer_id_max,
            packed_draft_device_pools=config.packed_draft_device_pools,
            device_translate=anchor.device_translate,
        )
        cache.register_sidecar_pool(sc.sidecar_spec(), entry)
        added += 1
        logger.info('xtier kvstore bridge: attached missing declared sidecar pool=%s', sc.pool_name)
    return added


def _as_paged_uint8_rows(tensors: tuple[torch.Tensor, ...], page_size: int) -> list[torch.Tensor]:
    rows = []
    for t in tensors:
        if t.ndim < 1:
            raise ValueError(f'scale tensor must be at least 1D, got shape {tuple(t.shape)}')
        num_pages = t.shape[0] // page_size
        if num_pages <= 0:
            raise ValueError(f'scale tensor slots {t.shape[0]} < page_size {page_size}')
        sliced = t[: num_pages * page_size]
        if not sliced.is_contiguous():
            raise ValueError(f'scale tensor slice is not contiguous: shape={tuple(t.shape)}, stride={t.stride()}')
        rows.append(sliced.view(torch.uint8).reshape(num_pages, -1))
    return rows


def _attach_token_scale_sidecars(cache: Any) -> int:
    """Gap 3: mirror stripped KVScaleLayout.TOKEN K/V scale tensors as page-aligned sidecar host pools."""
    from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, SidecarPoolSpec
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _get_allocator_type,
        build_pool_entry,
    )
    from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool

    hpg = getattr(cache, 'host_pool_group', None)
    cc = getattr(cache, 'cache_controller', None)
    if hpg is None or cc is None:
        return 0

    added = 0
    for entry in list(hpg.entry_map.values()):
        scales = getattr(entry.host_pool, '_xtier_token_scales', None)
        if scales is None:
            continue
        page_size = int(getattr(entry.host_pool, 'page_size', 64))
        num_host_pages = int(getattr(entry.host_pool, 'page_num', 0) or (entry.host_pool.size // page_size))
        layout = getattr(entry.host_pool, 'layout', 'page_first')
        alloc_type = _get_allocator_type()

        for suffix, bufs in (('k_scale', scales.k), ('v_scale', scales.v)):
            sname = f'{entry.name}_{suffix}'
            if sname in hpg.entry_map:
                continue
            buf_tuple = (bufs,) if isinstance(bufs, torch.Tensor) else tuple(bufs)
            if not buf_tuple:
                continue
            paged_rows = _as_paged_uint8_rows(buf_tuple, page_size)
            item_bytes = int(paged_rows[0].shape[1])
            scale_host = DeepSeekV4PagedHostPool(
                pool_name=sname,
                device_buffers=paged_rows,
                item_bytes=item_bytes,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout=layout,
                allocator_type=alloc_type,
                page_aligned_only=True,
            )
            target_layers = len(paged_rows)
            scale_layer_mapping = {
                lid: mapped
                for lid in range(cc.transfer_layer_id_max)
                if (mapped := entry.layer_mapper(lid)) is not None and mapped < target_layers
            }
            scale_entry = build_pool_entry(
                name=sname,
                host_pool=scale_host,
                device_pool=entry.device_pool,
                layer_mapping=scale_layer_mapping,
                transfer_layer_id_max=cc.transfer_layer_id_max,
                device_translate=entry.device_translate,
            )
            spec = SidecarPoolSpec(
                pool_name=sname,
                indices_from_pool=entry.name,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            )
            cache.register_sidecar_pool(spec, scale_entry)
            added += 1
            logger.info(
                'xtier kvstore bridge: attached KVScaleLayout.TOKEN sidecar pool=%s (layers=%d, item_bytes=%d, host_pages=%d)',
                sname, len(paged_rows), item_bytes, num_host_pages,
            )
    return added


# ----- Hook installation ----------------------------------------------------


def install(registry, hook_type) -> list[str]:
    """Register generic KV-offload bridge hooks ONLY when XTIER_KV_OFFLOAD=1."""
    if not kv_cfg.enabled():
        return []

    def around_draft_plan(original_fn, self, *args, **kwargs):
        from sglang.srt.speculative.base_spec_worker import HiCacheDraftMode, HiCacheDraftPlan

        plan = original_fn(self, *args, **kwargs)
        if plan.mode != HiCacheDraftMode.PACKED or not plan.device_pools:
            return plan
        target_runner = self.target_worker.model_runner
        target_pool = getattr(target_runner, 'token_to_kv_pool', None)
        if target_pool is None:
            return plan
        for depth, dpool in enumerate(plan.device_pools):
            ok, reason = _draft_pack_compatible(target_pool, dpool)
            if not ok:
                logger.info(
                    'xtier kvstore bridge: packed draft[%d] incompatible with target (%s); falling back to SIDECAR',
                    depth, reason,
                )
                target_runner.mtp_draft_device_pools = ()
                target_runner.mtp_draft_mamba_pools = ()
                return HiCacheDraftPlan(mode=HiCacheDraftMode.SIDECAR, device_pools=plan.device_pools[:1])
        return plan

    def around_build_kv_host_pool(original_fn, *args, **kwargs):
        import msgspec
        from sglang.srt.mem_cache.device_pool_info import KVScaleLayout, MHABufferInfo

        kv_pool = kwargs.get('kv_pool')
        orig_infos_fn = getattr(kv_pool, 'get_device_pool_infos', None) if kv_pool is not None else None
        orig_infos = orig_infos_fn() if callable(orig_infos_fn) else None
        first_bi = orig_infos[0].buffer_info if orig_infos else None
        norm_bi, dim_fixed = _normalize_mha_packed_dims(first_bi)

        if dim_fixed:
            # Drop oversized pre-allocated staging buffers built against unpacked head_dim.
            staging = getattr(kv_pool, 'hicache_write_back_staging', None)
            if isinstance(staging, tuple) and len(staging) == 2:
                if staging[0].shape[-1] != norm_bi.head_dim or staging[1].shape[-1] != norm_bi.v_head_dim:
                    kv_pool.hicache_write_back_staging = None
            logger.info(
                'xtier kvstore bridge: normalized packed MHA buffer_info dims (%d/%d -> %d/%d) to match physical storage',
                first_bi.head_dim, first_bi.v_head_dim, norm_bi.head_dim, norm_bi.v_head_dim,
            )

        def _call_with_bi(bi):
            if orig_infos is None or bi is first_bi:
                return original_fn(*args, **kwargs)
            patched_infos = (msgspec.structs.replace(orig_infos[0], buffer_info=bi), *orig_infos[1:])
            kv_pool.get_device_pool_infos = lambda: patched_infos
            try:
                return original_fn(*args, **kwargs)
            finally:
                kv_pool.get_device_pool_infos = orig_infos_fn

        try:
            return _call_with_bi(norm_bi)
        except ValueError as err:
            if "KV scale layout 'token'" not in str(err):
                raise
            if not isinstance(norm_bi, MHABufferInfo) or norm_bi.scales is None or norm_bi.scales.layout != KVScaleLayout.TOKEN:
                raise
            stripped_bi = msgspec.structs.replace(norm_bi, scales=None)
            host_pool = _call_with_bi(stripped_bi)
            host_pool._xtier_token_scales = norm_bi.scales
            logger.info(
                'xtier kvstore bridge: deferred KVScaleLayout.TOKEN (%d layers) to paged scale sidecars',
                len(norm_bi.scales.k),
            )
            return host_pool

    def after_attach_hybrid_pool(result, cache, params, *args, **kwargs):
        _attach_missing_declared_sidecars(cache)
        _attach_token_scale_sidecars(cache)
        alloc = getattr(params, 'token_to_kv_pool_allocator', None)
        kvcache = alloc.get_kvcache() if alloc is not None else None
        pools: list[tuple[str, Any]] = [('kvcache', kvcache)]
        state = getattr(alloc, 'state', None)
        if state is not None and getattr(state, 'pool', None) is not None:
            pools.append(('mamba_state', state.pool))
        for i, dp in enumerate(getattr(params, 'mtp_draft_device_pools', ()) or ()):
            pools.append((f'packed_draft[{i}]', dp))
        for i, mp in enumerate(getattr(params, 'mtp_draft_mamba_pools', ()) or ()):
            pools.append((f'packed_draft_mamba[{i}]', mp))
        _verify_coverage(cache, pools, context='attach')
        return result

    def after_register_draft_pools(result, *args, **kwargs):
        draft_plan = kwargs.get('draft_plan')
        tree_cache = kwargs.get('tree_cache')
        if draft_plan is not None and tree_cache is not None and getattr(draft_plan, 'device_pools', None):
            pools = [(f'sidecar_draft[{i}]', dp) for i, dp in enumerate(draft_plan.device_pools)]
            _verify_coverage(tree_cache, pools, context='draft_sidecar')
        return result

    def around_cgroup_memory_headroom(original_fn, *args, **kwargs):
        try:
            return original_fn(*args, **kwargs)
        except ValueError as err:
            logger.warning(
                'xtier kvstore bridge: unresolvable container cgroup memory path (%s); '
                'falling back to host available memory',
                err,
            )
            return None

    def around_full_finalize_match(original_fn, self, result, params, value_chunks, best_value_len):
        if not _source_needs_full_host_hit_fix(original_fn):
            return original_fn(self, result, params, value_chunks, best_value_len)
        ct = self.component_type
        kv_host_hit = 0
        node = result.best_match_node
        root_node = self.tree_core.root_node
        while node is not result.last_device_node and node is not root_node:
            cd = node.component_data[ct]
            if cd.value is None and cd.host_value is not None:
                kv_host_hit += len(cd.host_value)
            node = node.parent
        if kv_host_hit > 0:
            return result._replace(
                host_hit_length=max(result.host_hit_length, kv_host_hit)
            )
        return result

    def around_resolve_tree_core_backend(original_fn, name, params):
        backend = original_fn(name, params)
        if backend == 'rust' and _full_component_needs_host_hit_fix():
            logger.info(
                'xtier kvstore bridge: pinning unified radix TreeCore to python so mixed '
                'device/host HiCache prefix fix applies (bypassing unpatched Rust full.rs)'
            )
            return 'python'
        return backend

    registry.register(f'{_SPEC_BASE}.BaseSpecWorker._build_hicache_draft_plan', around_draft_plan, hook_type.AROUND)
    registry.register(f'{_ASSEMBLER}.build_kv_host_pool', around_build_kv_host_pool, hook_type.AROUND)
    registry.register(f'{_ASSEMBLER}.attach_hybrid_pool_to_unified_cache', after_attach_hybrid_pool, hook_type.AFTER)
    registry.register(f'{_KV_BUILDER}.maybe_register_hicache_draft', after_register_draft_pools, hook_type.AFTER)
    registry.register(f'{_HOST_MEM}._cgroup_memory_headroom', around_cgroup_memory_headroom, hook_type.AROUND)
    registry.register(
        f'{_FULL_COMPONENT}.FullComponent.finalize_match_result_in_tree_core',
        around_full_finalize_match,
        hook_type.AROUND,
    )
    registry.register(
        f'{_TREE_CORE_REGISTRY}.resolve_tree_core_backend',
        around_resolve_tree_core_backend,
        hook_type.AROUND,
    )
    return ['kvstore.bridge']
