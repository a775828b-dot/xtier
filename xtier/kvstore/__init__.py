"""Generic multi-drive NVMe KV cache offloading for XTier (SGLang HiCache buffer_only + libxtier_kvstore.so).

Model-agnostic by construction: contains zero model-name branches. Controlled via ``XTIER_KV_*`` environment variables:
  XTIER_KV_OFFLOAD=1          Enable NVMe KV offloading (default 0; when 0, zero SGLang bridge hooks are registered)
  XTIER_KV_PATHS=...          Colon-separated pool directories (default /data/fast/cold{1..4}/xtier-kv)
  XTIER_KV_SIZE_GIB=200       Total NVMe pool capacity in GiB, split equally across drives
  XTIER_KV_STAGING_RATIO=1.25 Host pinned staging ratio vs GPU KV pool (--hicache-ratio, universal across all models)
  XTIER_KV_STAGING_GIB=0      Optional explicit host staging GiB (--hicache-size, overrides ratio when >0)
  XTIER_KV_WRITE_QD=4         Max concurrent background writes per drive
  XTIER_KV_READ_QD=64         Max concurrent reads per drive
  XTIER_KV_REGION_MIB=256     Ring recycle region size in MiB
  XTIER_KV_RELOCATE_PCT=50    Max percentage of a region relocated for CLOCK hot entries on recycle
  XTIER_KV_STRICT_COVERAGE=1  Require exact device->host data_ptr() coverage check at attach (default 1)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_PATHS = (
    '/data/fast/cold1/xtier-kv:'
    '/data/fast/cold2/xtier-kv:'
    '/data/fast/cold3/xtier-kv:'
    '/data/fast/cold4/xtier-kv'
)


def enabled() -> bool:
    return os.environ.get('XTIER_KV_OFFLOAD', '0') == '1'


def paths() -> list[str]:
    raw = os.environ.get('XTIER_KV_PATHS', DEFAULT_PATHS)
    return [p for p in raw.split(':') if p]


def size_gib() -> int:
    return max(1, int(os.environ.get('XTIER_KV_SIZE_GIB', '200')))


def staging_ratio(default: float = 1.25) -> float:
    raw = os.environ.get('XTIER_KV_STAGING_RATIO', '')
    return max(0.25, float(raw)) if raw else default


def staging_gib(default: int = 0) -> int:
    raw = os.environ.get('XTIER_KV_STAGING_GIB', '')
    return max(0, int(raw)) if raw else default


def write_qd() -> int:
    return max(1, int(os.environ.get('XTIER_KV_WRITE_QD', '4')))


def read_qd() -> int:
    return max(1, int(os.environ.get('XTIER_KV_READ_QD', '64')))


def region_mib() -> int:
    return max(1, int(os.environ.get('XTIER_KV_REGION_MIB', '256')))


def relocate_pct() -> int:
    return max(0, min(90, int(os.environ.get('XTIER_KV_RELOCATE_PCT', '50'))))


def strict_coverage() -> bool:
    return os.environ.get('XTIER_KV_STRICT_COVERAGE', '1') != '0'


def available(check_paths: list[str] | None = None) -> tuple[bool, str]:
    """Return (ok, reason) checking native library and writable drive directories."""
    from xtier.engine import lib_dir

    so = lib_dir() / 'libxtier_kvstore.so'
    if not so.is_file():
        return False, f'missing native library {so}'
    dirs = check_paths if check_paths is not None else paths()
    if not dirs:
        return False, 'XTIER_KV_PATHS is empty'
    missing = [d for d in dirs if not Path(d).is_dir() or not os.access(d, os.W_OK)]
    if missing:
        return False, f'NVMe KV directories missing or not writable: {missing[0]}'
    return True, ''


def sglang_args(staging: int | None = None, ratio: float | None = None) -> list[str]:
    """Build standard SGLang CLI flags for buffer_only + dynamic XTierNVMeStorage."""
    extra = json.dumps(
        {
            'backend_name': 'xtier_nvme',
            'module_path': 'xtier.kvstore.backend',
            'class_name': 'XTierNVMeStorage',
            'interface_v1': 1,
        },
        separators=(',', ':'),
    )
    gib = staging if staging is not None else staging_gib()
    size_flags = (
        ['--hicache-size', str(gib)]
        if gib > 0
        else ['--hicache-ratio', str(ratio if ratio is not None else staging_ratio())]
    )
    return [
        '--enable-hierarchical-cache',
        '--hicache-host-memory-mode', 'buffer_only',
        *size_flags,
        '--hicache-write-policy', 'write_through',
        '--hicache-mem-layout', 'page_first',
        '--hicache-io-backend', 'kernel',
        '--hicache-storage-prefetch-policy', 'timeout',
        '--hicache-storage-backend', 'dynamic',
        '--hicache-storage-backend-extra-config', extra,
    ]
