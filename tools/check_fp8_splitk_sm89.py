"""校验 SM89 上的 SPLIT_K / SWAP_AB 路径：与参考实现的数值误差 + 速度。

参考实现用 fp32 反量化手算（block-wise FP8：每 32 个 K 元素一个 scale）。
"""
import json
import os
import statistics
import sys

import torch
from sglang.kernels.ops.gemm import fp8_kernel as fk

BLOCK = [32, 32]
CFG_DIR = "/opt/sglang/python/sglang/kernels/ops/gemm/configs"


def make_inputs(M, N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.randn(M, K, device="cuda", generator=g).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda", generator=g).to(torch.float8_e4m3fn)
    As = torch.rand(M, K // 32, device="cuda", generator=g) + 0.5
    Bs = torch.rand(N // 32, K // 32, device="cuda", generator=g) + 0.5
    return A, B, As, Bs


def reference(A, B, As, Bs):
    """fp32 参考：逐 block 反量化后 matmul。
    As: (M, K//32) -> (M, K)；Bs: (N//32, K//32) -> (N, K)。"""
    a = A.to(torch.float32) * As.repeat_interleave(32, dim=1)
    b = B.to(torch.float32) * Bs.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    return a @ b.T


def run(cfg, data):
    M = data[0].shape[0]
    fk.get_w8a8_block_fp8_configs = lambda *a, **k: {M: cfg}
    return fk.w8a8_block_fp8_matmul_triton(*data, BLOCK, torch.bfloat16)


def bench(cfg, data, repeat=20):
    M = data[0].shape[0]
    fk.get_w8a8_block_fp8_configs = lambda *a, **k: {M: cfg}
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


def rel_err(out, ref):
    out = out.to(torch.float32)
    ref = ref.to(torch.float32)
    return ((out - ref).abs().max() / ref.abs().max()).item()


def main():
    N, K = 16384, 1280
    print(f"device={torch.cuda.get_device_name(0)}")
    cases = []
    for M in (1, 2, 4, 8, 16, 32):
        for swap, split_k in ((False, 1), (False, 2), (False, 4), (True, 1),
                              (True, 2), (True, 4)):
            cases.append((M, swap, split_k))

    print(f"{'M':>4} {'swap':>5} {'split_k':>8} {'rel_err':>10} {'ms':>9}")
    for M, swap, split_k in cases:
        data = make_inputs(M, N, K)
        cfg = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
                   GROUP_SIZE_M=8, num_warps=4, num_stages=3)
        if swap:
            cfg["SWAP_AB"] = True
        if split_k > 1:
            cfg["SPLIT_K"] = split_k
        try:
            out = run(cfg, data)
            ref = reference(*data)
            err = rel_err(out, ref)
            ms = bench(cfg, data)
            print(f"{M:>4} {str(swap):>5} {split_k:>8} {err:>10.5f} {ms:>9.4f}")
        except Exception as e:
            print(f"{M:>4} {str(swap):>5} {split_k:>8}  FAILED: {str(e)[:60]}")
        del data


if __name__ == "__main__":
    main()
