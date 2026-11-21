#!/usr/bin/env python3
"""为 L20 (SM89) 生成 SGLang W8A8 block-FP8 Triton GEMM 的 tuned configs（逐 M 调优版）。

与 tools/tune_l20_fp8.py 的区别：
  1. **逐 M 调优**：原版只在 M=8 上测速，却把同一份 tile 写进全部 8 个 M bucket。
     decode 时 M = batch × topk，从 1 到 48 都有，小 M 与大 M 的最优 tile 完全不同。
  2. **输出到 XTier 加速包**：除 SGLang 的 configs/ 外，同时写入
     xtier/packs/deepseek_v4/fp8_configs/（device_name=NVIDIA_L20），让加速包
     按硬件查得到配置，而不是回落到默认 tile。
  3. 报告每个 (shape, M) 相对 SGLang 默认配置的加速比。

用法（用空闲 GPU，不要占用服务卡）：
  CUDA_VISIBLE_DEVICES=1 python tools/tune_l20_fp8_perm.py
  CUDA_VISIBLE_DEVICES=1 python tools/tune_l20_fp8_perm.py --out pack   # 只写加速包
"""
import argparse
import itertools
import json
import os
import statistics

import torch
from sglang.kernels.ops.gemm import fp8_kernel as fk

SGLANG_CFG_DIR = "/opt/sglang/python/sglang/kernels/ops/gemm/configs"
PACK_CFG_DIR = "/opt/Xtier/xtier/packs/deepseek_v4/fp8_configs"
BLOCK = [32, 32]

# decode/verify 真正用到的形状（TP=2，来自服务日志的 "Config file not found at N=...,K=..."）
SHAPES = [
    (1792, 5120), (1280, 5120), (1536, 5120), (2304, 5120),
    (512, 5120), (5120, 1152), (16384, 1280), (4096, 1280),
    (5120, 4096), (5120, 15360), (25600, 6144), (1280, 4096),
    (1536, 7168), (2304, 7168), (7168, 1024), (2048, 512),
]
SMALL_M = [1, 2, 4, 8, 16, 32, 64, 128]

DEFAULT = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
               GROUP_SIZE_M=32, num_warps=4, num_stages=3)

# SM89 的共享内存只有 99 KiB，BLOCK_SIZE_N=128 时 num_stages 必须小，
# 否则 Triton 编译期就 out of resource。网格里保留 stages=2 配大 BN。
#
# SPLIT_K / SWAP_AB：SGLang 原本只给 SM90 用（hopper_tuned），但那个 kernel 是
# 通用 Triton（2D grid + 归约 / 操作数转置），SM89 的 FP8 张量核同样支持。
# 小 M（decode）时 K 维很长，split-K 把 K 切给多个 program，SWAP_AB 让权重 tile
# 占 M 维 —— 两者都只在小 M 下有意义，所以只对 M <= 32 搜索。
GRID = list(itertools.product(
    (16, 32, 64),        # BLOCK_SIZE_M
    (32, 64, 128),       # BLOCK_SIZE_N
    (32,),               # BLOCK_SIZE_K 必须 = 32（scale 粒度）
    (8, 32),             # GROUP_SIZE_M
    (2, 4, 8),           # num_warps
    (2, 3, 4),           # num_stages
))

SPLIT_GRID = [
    (False, 1), (False, 2), (False, 4), (False, 8),
    (True, 1), (True, 2), (True, 4),
]
SPLIT_K_MAX_M = 32       # M 超过它就不搜 split-K（收益消失，还多一次归约）


def make_inputs(M, N, K):
    A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
    As = torch.rand(M, K // 32, device="cuda")
    Bs = torch.rand(N // 32, K // 32, device="cuda")
    return A, B, As, Bs


def bench(cfg, data, repeat=15):
    M = data[0].shape[0]
    fk.get_w8a8_block_fp8_configs = lambda *a, **k: {M: cfg}
    try:
        for _ in range(3):
            fk.w8a8_block_fp8_matmul_triton(*data, BLOCK, torch.bfloat16)
        torch.cuda.synchronize()
        ts = []
        for _ in range(repeat):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fk.w8a8_block_fp8_matmul_triton(*data, BLOCK, torch.bfloat16)
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts)
    except Exception:
        return float("inf")


def cfg_of(t):
    bm, bn, bk, gm, nw, ns = t
    return dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                GROUP_SIZE_M=gm, num_warps=nw, num_stages=ns)


def cfg_of_split(t, swap, split_k):
    cfg = cfg_of(t)
    if swap:
        cfg["SWAP_AB"] = True
    if split_k > 1:
        cfg["SPLIT_K"] = split_k
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", choices=["both", "sglang", "pack"], default="both")
    ap.add_argument("--top", type=int, default=8, help="每个 M 保留的候选数")
    args = ap.parse_args()

    device = torch.cuda.get_device_name(0).replace(" ", "_")
    print(f"device={device}", flush=True)
    results = {}          # (N,K) -> {M: cfg}
    gains = []            # (speedup, N, K, M)

    # 阶段 1：每个 M 在代表形状上全网格，选出该 M 的候选
    # 小 M 额外搜索 SPLIT_K / SWAP_AB
    N0, K0 = SHAPES[0]
    shortlists = {}
    for M in SMALL_M:
        data = make_inputs(M, N0, K0)
        scored = []
        for grid in GRID:
            for swap, split_k in (SPLIT_GRID if M <= SPLIT_K_MAX_M else [(False, 1)]):
                cfg = cfg_of_split(grid, swap, split_k)
                ms = bench(cfg, data, repeat=5)
                scored.append((ms, grid, swap, split_k))
        scored.sort(key=lambda x: x[0])
        shortlists[M] = [(g, sw, sk) for _, g, sw, sk in scored[:args.top]]
        best = scored[0]
        print(f"M={M:>4}: best={best[0]:.4f} ms {best[1]} swap={best[2]} split_k={best[3]}",
              flush=True)
        del data

    # 阶段 2：每个形状 × 每个 M 用该 M 的候选复测
    for N, K in SHAPES:
        per_m = {}
        for M in SMALL_M:
            data = make_inputs(M, N, K)
            cands = list(shortlists[M]) + [(tuple(DEFAULT[k] for k in (
                "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                "GROUP_SIZE_M", "num_warps", "num_stages")), False, 1)]
            best, best_ms = None, float("inf")
            for g, sw, sk in cands:
                cfg = cfg_of_split(g, sw, sk)
                ms = bench(cfg, data)
                if ms < best_ms:
                    best, best_ms = cfg, ms
            base_ms = bench(DEFAULT, data)
            per_m[M] = best
            gains.append((base_ms / best_ms if best_ms > 0 else 1.0, N, K, M))
            del data
        results[(N, K)] = per_m
        line = " ".join(f"M{M}:{gains[-len(SMALL_M) + i][0]:.2f}x"
                        for i, M in enumerate(SMALL_M))
        print(f"N={N:>5} K={K:>5}: {line}", flush=True)

    # 写文件
    targets = []
    if args.out in ("both", "sglang"):
        targets.append(SGLANG_CFG_DIR)
    if args.out in ("both", "pack"):
        targets.append(PACK_CFG_DIR)
    for out_dir in targets:
        os.makedirs(out_dir, exist_ok=True)
        for (N, K), per_m in results.items():
            name = (f"N={N},K={K},device_name={device},"
                    f"dtype=fp8_w8a8,block_shape=[32, 32].json")
            with open(os.path.join(out_dir, name), "w") as f:
                json.dump({str(m): cfg for m, cfg in per_m.items()}, f, indent=2)
        print(f"wrote {len(results)} files to {out_dir}", flush=True)

    gains.sort(reverse=True)
    print("\n相对 SGLang 默认 tile 的加速比（top 12）:")
    for sp, N, K, M in gains[:12]:
        print(f"  {sp:5.2f}x  N={N:>5} K={K:>5} M={M}")
    print(f"中位数 {statistics.median(g for g, *_ in gains):.2f}x, "
          f"最差 {min(g for g, *_ in gains):.2f}x")


if __name__ == "__main__":
    main()
