#!/usr/bin/env python3
"""为 L20 (SM89) 生成 SGLang W8A8 block-FP8 Triton GEMM 的 tuned configs。

SGLang 自带配置全是 H20/H200/B200/PRO5000 的，L20 一个都没有，运行时会打
"Using default W8A8 Block FP8 kernel config" 并落到 BLOCK_SIZE_M=64 的保守参数，
而 decode 的 M 只有 1..48，浪费极大。

用法（在空闲 GPU 上跑，不要占用服务卡）：
  CUDA_VISIBLE_DEVICES=1 python tools/tune_l20_fp8.py
输出写到 sglang 的 configs/ 目录，文件名带 device_name=NVIDIA_L20。
"""
import itertools
import json
import os
import statistics

import torch
from sglang.kernels.ops.gemm import fp8_kernel as fk

CFG_DIR = "/opt/sglang/python/sglang/kernels/ops/gemm/configs"
BLOCK = [32, 32]
# decode 真正用到的形状（来自服务日志里 "Config file not found at N=...,K=..."）
SHAPES = [
    (1792, 5120), (1280, 5120), (1536, 5120), (2304, 5120),
    (512, 5120), (5120, 1152), (16384, 1280), (4096, 1280),
    (5120, 4096), (5120, 15360), (25600, 6144), (1280, 4096),
    (1536, 7168), (2304, 7168), (7168, 1024), (2048, 512),
]
M_PROBE = 8
SMALL_M = [1, 2, 4, 8, 16, 32, 64, 128]

DEFAULT = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
               GROUP_SIZE_M=32, num_warps=4, num_stages=3)

GRID = list(itertools.product(
    (16, 32, 64),        # BLOCK_SIZE_M
    (32, 64, 128),       # BLOCK_SIZE_N
    (32,),               # BLOCK_SIZE_K 必须 = 32：scale 的粒度，改大读数错位会算错
    (8, 32),             # GROUP_SIZE_M
    (2, 4, 8),           # num_warps
    (2, 3),              # num_stages
))


def make_inputs(M, N, K):
    A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
    As = torch.rand(M, K // 32, device="cuda")
    Bs = torch.rand(N // 32, K // 32, device="cuda")
    return A, B, As, Bs


def bench(cfg, data, repeat=12):
    """返回毫秒。monkeypatch 配置查找，让被测 config 真正生效。"""
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


def main():
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    results = {}

    # 阶段 1：代表形状全网格，取最好的若干结构
    N, K = SHAPES[0]
    data = make_inputs(M_PROBE, N, K)
    scored = []
    for i, grid in enumerate(GRID):
        ms = bench(cfg_of(grid), data, repeat=6)
        scored.append((ms, grid))
    scored.sort()
    print(f"grid done: {len(scored)} candidates, best={scored[0][0]:.4f} ms", flush=True)
    top = [g for _, g in scored[:8]]
    del data

    # 阶段 2：每个形状用 top 配置 + default 复测
    for N, K in SHAPES:
        data = make_inputs(M_PROBE, N, K)
        cands = [tuple(DEFAULT[k] for k in
                       ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                        "GROUP_SIZE_M", "num_warps", "num_stages"))] + top
        best, best_ms = DEFAULT, float("inf")
        for g in cands:
            cfg = cfg_of(g)
            ms = bench(cfg, data)
            if ms < best_ms:
                best, best_ms = cfg, ms
        base_ms = bench(DEFAULT, data)
        results[(N, K)] = best
        print(f"N={N:>5} K={K:>5}: {base_ms:.3f} -> {best_ms:.3f} ms "
              f"({base_ms / best_ms:.2f}x)  {best}", flush=True)
        del data

    # 写文件：文件名格式由 fp8_kernel.get_w8a8_block_fp8_configs 决定
    device = torch.cuda.get_device_name(0).replace(" ", "_")
    os.makedirs(CFG_DIR, exist_ok=True)
    for (N, K), cfg in results.items():
        name = (f"N={N},K={K},device_name={device},"
                f"dtype=fp8_w8a8,block_shape=[32, 32].json")
        with open(os.path.join(CFG_DIR, name), "w") as f:
            json.dump({str(m): cfg for m in SMALL_M}, f, indent=2)
        print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
