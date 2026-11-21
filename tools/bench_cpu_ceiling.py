#!/usr/bin/env python3
"""AMX 决策的 Phase 0：CPU 专家计算是「带宽受限」还是「算力受限」？

用真实的 TP=2 rank 形状（E=384, H=5120, I=1152, MXFP4）建一个 CPU engine plan，对同一组路由比较：

  t_read  = xtier_read_probe：只把被路由到的专家权重和 scale 流过一遍，不做任何算术
            （引擎自带的探针，csrc/engine/engine.cpp:1057）
  t_fwd   = xtier_forward：真实的 Gate/Up + Down 计算

  t_fwd ≈ t_read  → 内存带宽受限，AMX（算力更强）帮不上忙
  t_fwd ≫ t_read  → 算力受限，AMX-BF16 有空间

用法：python tools/bench_cpu_ceiling.py [--m 1,2,4,8,16] [--threads 32]
"""
import argparse
import ctypes
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, '/opt/Xtier')
from xtier.engine import CpuPlan, engine  # noqa: E402

E, H, I, K = 384, 5120, 1152, 6          # TP=2 的 rank 本地形状
SCALE_GROUP = 32                         # MXFP4


def build_plan(threads):
    lib = engine()
    # 权重缓冲（MXFP4：E2M1 nibble + UE8M0 每 32 个一个 scale）
    w13 = np.zeros((E, 2 * I, H // 2), dtype=np.uint8)
    w2 = np.zeros((E, H, I // 2), dtype=np.uint8)
    s13 = np.zeros((E, 2 * I, H // SCALE_GROUP), dtype=np.uint8)
    s2 = np.zeros((E, H, I // SCALE_GROUP), dtype=np.uint8)
    w13[:] = np.random.randint(0, 255, w13.shape, dtype=np.uint8)
    w2[:] = np.random.randint(0, 255, w2.shape, dtype=np.uint8)
    s13[:] = 127
    s2[:] = 127
    ptrs = (ctypes.c_void_p(w13.ctypes.data), ctypes.c_void_p(w2.ctypes.data),
            ctypes.c_void_p(s13.ctypes.data), ctypes.c_void_p(s2.ctypes.data))
    plan = CpuPlan(E, threads, ptrs, H, I, 0.0, 'mxfp4')
    print(f'plan built: E={E} H={H} I={I} threads={threads} '
          f'weights={w13.nbytes + w2.nbytes + s13.nbytes + s2.nbytes} B', flush=True)
    return plan, lib


def bench(fn, args, repeat=7):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        rc = fn(*args)
        dt = time.perf_counter() - t0
        if rc < 0:
            raise RuntimeError('engine error')
        ts.append(dt)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', default='1,2,4,8,16')
    ap.add_argument('--threads', type=int, default=32)
    args = ap.parse_args()

    plan, lib = build_plan(args.threads)
    # read_probe 的原型：int xtier_read_probe(void*, int m, int k, const uint16_t*, const int32_t*, const float*, float*)
    lib.xtier_read_probe.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.xtier_read_probe.restype = ctypes.c_int

    print(f'{"m":>4} {"t_read_ms":>11} {"t_fwd_ms":>10} {"fwd/read":>9} {"GB/s":>8} {"算术占比":>9}')
    for m in [int(v) for v in args.m.split(',')]:
        x = (np.random.randn(m, H) * 0.1).astype(np.float16)
        ids = np.random.randint(0, E, (m, K)).astype(np.int32)
        rw = np.random.rand(m, K).astype(np.float32)
        out = np.zeros((m, H), dtype=np.float32)
        # 每次 forward 读到的专家权重字节
        uniq = len(set(ids.reshape(-1).tolist()))
        bytes_read = uniq * (2 * I * H // 2 + H * I // 2
                              + 2 * I * H // SCALE_GROUP + H * I // SCALE_GROUP)
        t_read = bench(lib.xtier_read_probe,
                       (plan.plan, m, K, x.ctypes.data, ids.ctypes.data, rw.ctypes.data, out.ctypes.data))
        t_fwd = bench(plan._forward_once,
                      (m, K, x.ctypes.data, ids.ctypes.data, rw.ctypes.data, out.ctypes.data))
        gbs = bytes_read / t_read / 1e9
        # 算术占比 ≈ (t_fwd - t_read) / t_fwd：超过它才是算力受限
        print(f'{m:>4} {t_read*1e3:>11.2f} {t_fwd*1e3:>10.2f} {t_fwd/t_read:>9.2f} '
              f'{gbs:>8.1f} {max(0.0, 1 - t_read/t_fwd)*100:>8.0f}%')


if __name__ == '__main__':
    main()
