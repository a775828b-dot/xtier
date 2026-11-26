"""Phase 0 benchmark for libxtier_kvstore.so: 4x 9100 PRO O_DIRECT+io_uring -> Pinned -> GPU end-to-end throughput.

Measures:
1. 4-drive parallel synchronous write (write_qd=4, O_DIRECT + io_uring)
2. 4-drive parallel read into pinned host staging (O_DIRECT + io_uring)
3. End-to-end NVMe -> Pinned Host -> GPU H2D (RTX PRO 5000) pipelined & sequential
4. Linux /proc/meminfo Page Cache (Cached + Dirty) delta to verify 0 OS page cache growth

Usage:
  /data/fast/workspaces/xtier-dev/env/bin/python tools/kvstore_bench.py [--gib 4] [--page-kb 512]
"""

import argparse
import ctypes as C
import os
import sys
import time
from pathlib import Path

import torch


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


def meminfo_kb():
    out = {}
    with open('/proc/meminfo') as f:
        for line in f:
            k, _, rest = line.partition(':')
            if k in ('Cached', 'Dirty', 'MemAvailable'):
                out[k] = int(rest.strip().split()[0])
    return out


def load_lib():
    root = Path(__file__).resolve().parents[1]
    lib = C.CDLL(str(root / 'xtier' / 'engine' / 'lib' / 'libxtier_kvstore.so'))
    lib.xtier_kv_open.argtypes = [C.POINTER(C.c_char_p), C.c_int, C.POINTER(XkvConfig)]
    lib.xtier_kv_open.restype = C.c_void_p
    lib.xtier_kv_close.argtypes = [C.c_void_p]
    lib.xtier_kv_stats_get.argtypes = [C.c_void_p, C.POINTER(XkvStats)]
    lib.xtier_kv_last_error.restype = C.c_char_p
    lib.xtier_kv_put.argtypes = [C.c_void_p, C.POINTER(XkvItem), C.c_int, C.POINTER(C.c_uint8)]
    lib.xtier_kv_put.restype = C.c_int
    lib.xtier_kv_get.argtypes = [C.c_void_p, C.POINTER(XkvItem), C.c_int, C.POINTER(C.c_uint8)]
    lib.xtier_kv_get.restype = C.c_int
    return lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paths', default='/data/fast/cold1/xtier-kv:/data/fast/cold2/xtier-kv:/data/fast/cold3/xtier-kv:/data/fast/cold4/xtier-kv')
    ap.add_argument('--drive-gib', type=int, default=50)
    ap.add_argument('--gib', type=float, default=4.0, help='Total data to write & read back (GiB)')
    ap.add_argument('--page-kb', type=int, default=512, help='Page size in KiB')
    ap.add_argument('--batch-pages', type=int, default=128)
    args = ap.parse_args()

    paths = [os.path.join(p, 'pool.dat').encode() for p in args.paths.split(':') if p]
    c_paths = (C.c_char_p * len(paths))(*paths)
    cfg = XkvConfig(
        bytes_per_drive=args.drive_gib * (1024 ** 3),
        region_bytes=256 * 1024 * 1024,
        write_qd=4,
        read_qd=64,
        bounce_bytes=32 * 1024 * 1024,
        relocate_pct=50,
        ring_depth=128,
        flags=1,
    )

    lib = load_lib()
    store = lib.xtier_kv_open(c_paths, len(paths), C.byref(cfg))
    if not store:
        print(f'FAIL: xtier_kv_open: {lib.xtier_kv_last_error().decode()}')
        sys.exit(1)

    try:
        page_bytes = args.page_kb * 1024
        total_pages = int((args.gib * (1024 ** 3)) // page_bytes)
        batch_pages = min(args.batch_pages, total_pages)
        # Allocate 4 KiB-aligned pinned host tensor (cudaHostAlloc via pin_memory; PyTorch pinned allocs are page-aligned)
        host_w = torch.empty((batch_pages, page_bytes), dtype=torch.uint8, pin_memory=True)
        host_r = torch.empty((batch_pages, page_bytes), dtype=torch.uint8, pin_memory=True)
        # Fill deterministic check bytes per batch slot
        host_w[:, :8] = torch.arange(batch_pages, dtype=torch.uint8).unsqueeze(1)
        host_w[:, -8:] = 0x5A

        assert host_w.data_ptr() % 4096 == 0 and host_r.data_ptr() % 4096 == 0, 'pinned tensor not 4K aligned!'
        gpu_buf = torch.empty((batch_pages, page_bytes), dtype=torch.uint8, device='cuda')
        torch.cuda.synchronize()

        def build_items(tensor, key_start, n):
            iovs = (XkvIov * n)()
            items = (XkvItem * n)()
            base = tensor.data_ptr()
            for i in range(n):
                iovs[i].base = base + i * page_bytes
                iovs[i].len = page_bytes
                items[i].pool_id = 0
                items[i].iov_count = 1
                items[i].key.lo = key_start + i
                items[i].key.hi = 0xCAFE0000 + key_start + i
                items[i].iovs = C.pointer(iovs[i])
            return iovs, items

        m0 = meminfo_kb()

        # 1. Write benchmark (4 drives, O_DIRECT + io_uring, write_qd=4)
        ok_arr = (C.c_uint8 * batch_pages)()
        t0 = time.perf_counter()
        written = 0
        for start in range(0, total_pages, batch_pages):
            n = min(batch_pages, total_pages - start)
            _iovs, items = build_items(host_w, start, n)
            got = lib.xtier_kv_put(store, items, n, ok_arr)
            if got != n:
                print(f'FAIL: put {got}/{n}')
                sys.exit(1)
            written += n
        t_write = time.perf_counter() - t0
        write_gbs = (written * page_bytes) / 1e9 / t_write

        # 2. Pure NVMe -> Pinned Host read benchmark
        t0 = time.perf_counter()
        read_n = 0
        for start in range(0, total_pages, batch_pages):
            n = min(batch_pages, total_pages - start)
            _iovs, items = build_items(host_r, start, n)
            got = lib.xtier_kv_get(store, items, n, ok_arr)
            if got != n:
                print(f'FAIL: get {got}/{n}')
                sys.exit(1)
            read_n += n
        t_read = time.perf_counter() - t0
        read_gbs = (read_n * page_bytes) / 1e9 / t_read

        # Verify byte integrity on last batch
        if not torch.equal(host_r[:n, :8], host_w[:n, :8]) or not torch.equal(host_r[:n, -8:], host_w[:n, -8:]):
            print('FAIL: readback data mismatch')
            sys.exit(1)

        # 3. Pure Pinned Host -> GPU H2D benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for start in range(0, total_pages, batch_pages):
            n = min(batch_pages, total_pages - start)
            gpu_buf[:n].copy_(host_r[:n], non_blocking=True)
        torch.cuda.synchronize()
        t_h2d = time.perf_counter() - t0
        h2d_gbs = (total_pages * page_bytes) / 1e9 / t_h2d

        # 4. End-to-end NVMe -> Pinned -> GPU (sequential batch & double-buffered overlapped)
        # 4a: Sequential per batch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for start in range(0, total_pages, batch_pages):
            n = min(batch_pages, total_pages - start)
            _iovs, items = build_items(host_r, start, n)
            got = lib.xtier_kv_get(store, items, n, ok_arr)
            if got != n:
                print(f'FAIL: e2e seq get {got}/{n}')
                sys.exit(1)
            gpu_buf[:n].copy_(host_r[:n], non_blocking=True)
        torch.cuda.synchronize()
        t_e2e_seq = time.perf_counter() - t0
        e2e_seq_gbs = (total_pages * page_bytes) / 1e9 / t_e2e_seq

        # 4b: Double-buffered overlapped NVMe read + H2D copy
        host_r2 = torch.empty((batch_pages, page_bytes), dtype=torch.uint8, pin_memory=True)
        bufs = [host_r, host_r2]
        ev = [torch.cuda.Event(), torch.cuda.Event()]
        copy_stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx = 0
        for start in range(0, total_pages, batch_pages):
            n = min(batch_pages, total_pages - start)
            slot = idx & 1
            if idx >= 2:
                ev[slot].synchronize()
            _iovs, items = build_items(bufs[slot], start, n)
            got = lib.xtier_kv_get(store, items, n, ok_arr)
            if got != n:
                print(f'FAIL: e2e pipe get {got}/{n}')
                sys.exit(1)
            with torch.cuda.stream(copy_stream):
                gpu_buf[:n].copy_(bufs[slot][:n], non_blocking=True)
                ev[slot].record(copy_stream)
            idx += 1
        copy_stream.synchronize()
        t_e2e_pipe = time.perf_counter() - t0
        e2e_pipe_gbs = (total_pages * page_bytes) / 1e9 / t_e2e_pipe

        m1 = meminfo_kb()
        st = XkvStats()
        lib.xtier_kv_stats_get(store, C.byref(st))

        cached_delta_mb = (m1['Cached'] - m0['Cached']) / 1024.0
        dirty_delta_mb = (m1['Dirty'] - m0['Dirty']) / 1024.0
        print(f'=== Phase 0 Benchmark ({args.gib:.1f} GiB, page={args.page_kb} KiB, {len(paths)} drives) ===')
        print(f'  Write (O_DIRECT, QD=4)        : {write_gbs:6.2f} GB/s ({t_write:.3f} s)')
        print(f'  Read  (NVMe -> Pinned)        : {read_gbs:6.2f} GB/s ({t_read:.3f} s)')
        print(f'  H2D   (Pinned -> RTX PRO 5000): {h2d_gbs:6.2f} GB/s ({t_h2d:.3f} s)')
        print(f'  E2E   (NVMe->Pin->GPU seq)    : {e2e_seq_gbs:6.2f} GB/s ({t_e2e_seq:.3f} s)')
        print(f'  E2E   (NVMe->Pin->GPU pipe)   : {e2e_pipe_gbs:6.2f} GB/s ({t_e2e_pipe:.3f} s)')
        print(f'  IOs: direct={st.direct_ios}, bounce={st.bounce_ios}')
        print(f'  Page Cache delta: Cached={cached_delta_mb:+.1f} MiB, Dirty={dirty_delta_mb:+.1f} MiB')
        if e2e_pipe_gbs < 20.0:
            print(f'WARN: E2E pipe {e2e_pipe_gbs:.2f} GB/s < 20 GB/s target')
    finally:
        lib.xtier_kv_close(store)


if __name__ == '__main__':
    main()
