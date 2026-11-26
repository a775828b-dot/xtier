"""ctypes binding for libxtier_kvstore.so (multi-drive O_DIRECT + io_uring KV cache tier)."""

from __future__ import annotations

import ctypes as C
import hashlib
import os
import struct
from pathlib import Path
from typing import Sequence

from xtier.engine import lib_dir


class XkvKey(C.Structure):
    _fields_ = [('lo', C.c_uint64), ('hi', C.c_uint64)]


class XkvIov(C.Structure):
    _fields_ = [('base', C.c_void_p), ('len', C.c_size_t)]


class XkvItem(C.Structure):
    _fields_ = [
        ('pool_id', C.c_uint16),
        ('reserved', C.c_uint16),
        ('iov_count', C.c_uint32),
        ('key', XkvKey),
        ('iovs', C.POINTER(XkvIov)),
    ]


class XkvConfig(C.Structure):
    _fields_ = [
        ('bytes_per_drive', C.c_uint64),
        ('region_bytes', C.c_uint64),
        ('write_qd', C.c_uint32),
        ('read_qd', C.c_uint32),
        ('bounce_bytes', C.c_uint32),
        ('relocate_pct', C.c_uint32),
        ('ring_depth', C.c_uint32),
        ('flags', C.c_uint32),
    ]


class XkvStats(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        'live_entries', 'live_bytes', 'capacity_bytes',
        'puts_ok', 'puts_skipped_dup', 'puts_failed',
        'gets_hit', 'gets_miss', 'gets_failed',
        'bytes_written', 'bytes_read', 'regions_recycled',
        'hot_relocated', 'hot_dropped_cap', 'direct_ios', 'bounce_ios', 'evict_wait_count',
    )]


_lib = None


def load_library() -> C.CDLL:
    global _lib
    if _lib is None:
        so = lib_dir() / 'libxtier_kvstore.so'
        lib = C.CDLL(str(so))
        lib.xtier_kv_open.argtypes = [C.POINTER(C.c_char_p), C.c_int, C.POINTER(XkvConfig)]
        lib.xtier_kv_open.restype = C.c_void_p
        lib.xtier_kv_close.argtypes = [C.c_void_p]
        lib.xtier_kv_clear.argtypes = [C.c_void_p]
        lib.xtier_kv_stats_get.argtypes = [C.c_void_p, C.POINTER(XkvStats)]
        lib.xtier_kv_last_error.restype = C.c_char_p
        lib.xtier_kv_exists.argtypes = [
            C.c_void_p, C.c_uint16, C.POINTER(XkvKey), C.c_int, C.POINTER(C.c_uint8),
        ]
        lib.xtier_kv_put.argtypes = [C.c_void_p, C.POINTER(XkvItem), C.c_int, C.POINTER(C.c_uint8)]
        lib.xtier_kv_put.restype = C.c_int
        lib.xtier_kv_get.argtypes = [C.c_void_p, C.POINTER(XkvItem), C.c_int, C.POINTER(C.c_uint8)]
        lib.xtier_kv_get.restype = C.c_int
        lib.xtier_kv_set_test_read_delay_us.argtypes = [C.c_void_p, C.c_uint32]
        _lib = lib
    return _lib


def key_to_128(key: str) -> tuple[int, int]:
    """Parse an SGLang storage key into (lo, hi) 64-bit unsigned ints without collisions."""
    if len(key) >= 32:
        try:
            raw = bytes.fromhex(key[-32:])
            lo, hi = struct.unpack('<QQ', raw)
            return lo, hi
        except ValueError:
            pass
    digest = hashlib.blake2b(key.encode('utf-8'), digest_size=16).digest()
    return struct.unpack('<QQ', digest)


class NativeKVStore:
    """Thin resource-owning wrapper around xtier_kv_store_t."""

    def __init__(
        self,
        dirs: Sequence[str],
        bytes_per_drive: int,
        *,
        region_bytes: int = 256 * 1024 * 1024,
        write_qd: int = 4,
        read_qd: int = 64,
        bounce_bytes: int = 32 * 1024 * 1024,
        relocate_pct: int = 50,
        ring_depth: int = 128,
        pin_numa: bool = True,
    ):
        self._lib = load_library()
        self._handle = None
        file_paths = []
        for d in dirs:
            p = Path(d)
            if p.suffix == '.dat':
                p.parent.mkdir(parents=True, exist_ok=True)
                file_paths.append(str(p).encode('utf-8'))
            else:
                p.mkdir(parents=True, exist_ok=True)
                file_paths.append(str(p / 'pool.dat').encode('utf-8'))
        c_paths = (C.c_char_p * len(file_paths))(*file_paths)
        cfg = XkvConfig(
            bytes_per_drive=bytes_per_drive,
            region_bytes=region_bytes,
            write_qd=write_qd,
            read_qd=read_qd,
            bounce_bytes=bounce_bytes,
            relocate_pct=relocate_pct,
            ring_depth=ring_depth,
            flags=1 if pin_numa else 0,
        )
        handle = self._lib.xtier_kv_open(c_paths, len(file_paths), C.byref(cfg))
        if not handle:
            err = self._lib.xtier_kv_last_error()
            msg = err.decode('utf-8', 'replace') if err else 'unknown error'
            raise RuntimeError(f'xtier_kv_open failed: {msg}')
        self._handle = handle
        self.file_paths = [p.decode('utf-8') for p in file_paths]

    def close(self) -> None:
        h = self._handle
        self._handle = None
        if h:
            self._lib.xtier_kv_close(h)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def clear(self) -> None:
        if self._handle:
            self._lib.xtier_kv_clear(self._handle)

    def stats(self) -> dict[str, int]:
        st = XkvStats()
        if self._handle:
            self._lib.xtier_kv_stats_get(self._handle, C.byref(st))
        return {name: int(getattr(st, name)) for name, _ in XkvStats._fields_}

    def exists(self, pool_id: int, keys: Sequence[str]) -> list[bool]:
        n = len(keys)
        if not self._handle or n == 0:
            return [False] * n
        c_keys = (XkvKey * n)()
        for i, k in enumerate(keys):
            lo, hi = key_to_128(k)
            c_keys[i].lo = lo
            c_keys[i].hi = hi
        bm = (C.c_uint8 * n)()
        self._lib.xtier_kv_exists(self._handle, pool_id, c_keys, n, bm)
        return [bool(bm[i]) for i in range(n)]

    def _batch_io(self, entries: Sequence[tuple[int, str, Sequence[tuple[int, int]]]], *, write: bool) -> list[bool]:
        """Execute a batch of (pool_id, key_str, [(ptr, len), ...]) in a single C++ call."""
        n = len(entries)
        if not self._handle or n == 0:
            return [False] * n
        total_iovs = sum(len(iovs) for _, _, iovs in entries)
        c_iovs = (XkvIov * max(1, total_iovs))()
        c_items = (XkvItem * n)()
        iov_base = C.cast(c_iovs, C.POINTER(XkvIov))
        pos = 0
        for i, (pool_id, key_str, iovs) in enumerate(entries):
            lo, hi = key_to_128(key_str)
            cnt = len(iovs)
            for ptr, length in iovs:
                c_iovs[pos].base = ptr
                c_iovs[pos].len = length
                pos += 1
            c_items[i].pool_id = pool_id
            c_items[i].reserved = 0
            c_items[i].iov_count = cnt
            c_items[i].key.lo = lo
            c_items[i].key.hi = hi
            c_items[i].iovs = C.cast(
                C.addressof(c_iovs) + (pos - cnt) * C.sizeof(XkvIov),
                C.POINTER(XkvIov),
            ) if cnt else iov_base
        ok = (C.c_uint8 * n)()
        fn = self._lib.xtier_kv_put if write else self._lib.xtier_kv_get
        fn(self._handle, c_items, n, ok)
        return [bool(ok[i]) for i in range(n)]

    def put_batch(self, entries: Sequence[tuple[int, str, Sequence[tuple[int, int]]]]) -> list[bool]:
        return self._batch_io(entries, write=True)

    def get_batch(self, entries: Sequence[tuple[int, str, Sequence[tuple[int, int]]]]) -> list[bool]:
        return self._batch_io(entries, write=False)

    def set_test_read_delay_us(self, delay_us: int) -> None:
        if self._handle:
            self._lib.xtier_kv_set_test_read_delay_us(self._handle, int(delay_us))
