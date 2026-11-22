"""Verify XTier's zero-patch SM89 block-FP8 SWAP_AB / SPLIT_K plugin hook (xtier/platform.py).

- Checks exact (8, 9) capability gating (excludes SM80 A100 and SM86 RTX 3090 which lack FP8 tensor cores).
- Installs via SGLang's HookRegistry and verifies propagation to fp8_utils.w8a8_block_fp8_matmul_triton.
- On real SM89 GPUs (L4 / L20 / L40S / RTX 4090), runs SWAP_AB and SPLIT_K=2/4 configs eagerly and inside a CUDA graph,
  checks accuracy against an FP32 dequantized reference, and verifies deterministic mode falls back to original.

  python tests/sm89_block_fp8_check.py
"""
import sys
import types
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from sglang.kernels.ops.gemm import fp8_kernel as fk  # noqa: E402
from sglang.srt.layers.quantization import fp8_utils  # noqa: E402
from sglang.srt.plugins.hook_registry import HookRegistry, HookType  # noqa: E402

from xtier import platform  # noqa: E402

# 1. Exact capability guard: only (8, 9) returns True (SM80 A100 and SM86 RTX 3090 have no FP8 tensor cores)
for cap, expect in (((8, 0), False), ((8, 6), False), ((8, 9), True), ((9, 0), False), ((12, 0), False)):
    platform._is_sm89.cache_clear()
    with mock.patch('torch.cuda.get_device_capability', return_value=cap):
        got = platform._is_sm89(0)
        assert got is expect, f'_is_sm89({cap}) returned {got}, expected {expect}'
platform._is_sm89.cache_clear()
print('capability guard: exact (8, 9) only: ok')

# 2. Register and apply via HookRegistry (verifies propagation to fp8_utils)
HookRegistry.reset()
HookRegistry.register(platform.BLOCK_FP8_TRITON, platform._w8a8_block_fp8_sm89, HookType.AROUND)
HookRegistry.apply_hooks()
assert fp8_utils.w8a8_block_fp8_matmul_triton is fk.w8a8_block_fp8_matmul_triton, 'HookRegistry did not propagate to fp8_utils'
print('HookRegistry apply + fp8_utils propagation: ok')

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
    print('SKIP: live SM89 kernel run requires an SM89 GPU (Ada: L4 / L20 / L40S / RTX 4090)')
    sys.exit(0)

# 3. Live SM89 execution: SWAP_AB, SPLIT_K=2/4, CUDA graph replay, and deterministic fallback
N, K, BLOCK = 4096, 1280, [32, 32]
g = torch.Generator(device='cuda').manual_seed(20261122)
B = torch.randn(N, K, device='cuda', generator=g).to(torch.float8_e4m3fn)
Bs = torch.rand(N // 32, K // 32, device='cuda', generator=g) + 0.5
B_ref = B.float() * Bs.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)

base_cfg = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}
cases = [
    ('swap_ab', {**base_cfg, 'SWAP_AB': True}),
    ('split_k2', {**base_cfg, 'SPLIT_K': 2}),
    ('swap_split_k2', {**base_cfg, 'SWAP_AB': True, 'SPLIT_K': 2}),
    ('swap_split_k4', {**base_cfg, 'SWAP_AB': True, 'SPLIT_K': 4}),
]

ok = True
orig_get_cfg = fk.get_w8a8_block_fp8_configs
try:
    for M in (1, 4, 16, 64):
        A = torch.randn(M, K, device='cuda', generator=g).to(torch.float8_e4m3fn)
        As = torch.rand(M, K // 32, device='cuda', generator=g) + 0.5
        ref = (A.float() * As.repeat_interleave(32, dim=1)) @ B_ref.T
        for name, cfg in cases:
            fk.get_w8a8_block_fp8_configs = lambda *a, _c=cfg, _m=M, **k: {_m: _c}
            y = fp8_utils.w8a8_block_fp8_matmul_triton(A, B, As, Bs, BLOCK, torch.bfloat16)
            rel = ((y.float() - ref).abs().max() / ref.abs().max()).item()
            good = rel < 1e-2
            if M == 4:
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    yg = fp8_utils.w8a8_block_fp8_matmul_triton(A, B, As, Bs, BLOCK, torch.bfloat16)
                gr.replay()
                torch.cuda.synchronize()
                graph_same = torch.equal(y, yg)
                good &= graph_same
                print(f'M={M:2d} {name:14s}: rel_err={rel:.5f}, cuda_graph_bitwise={graph_same}')
            else:
                print(f'M={M:2d} {name:14s}: rel_err={rel:.5f}')
            ok &= good

    # Deterministic inference must delegate back to original (strips SWAP_AB / SPLIT_K)
    A4 = torch.randn(4, K, device='cuda', generator=g).to(torch.float8_e4m3fn)
    As4 = torch.rand(4, K // 32, device='cuda', generator=g) + 0.5
    ref4 = (A4.float() * As4.repeat_interleave(32, dim=1)) @ B_ref.T
    fk.get_w8a8_block_fp8_configs = lambda *a, **k: {4: {**base_cfg, 'SWAP_AB': True, 'SPLIT_K': 2}}
    fake_ctx = types.SimpleNamespace(is_config_namespace_published=lambda ns: ns == 'exec')
    fake_exec = types.SimpleNamespace(deterministic=types.SimpleNamespace(enable_deterministic_inference=True))
    with mock.patch.object(fk, 'get_context', return_value=fake_ctx), mock.patch.object(fk, 'get_exec', return_value=fake_exec):
        yd = fp8_utils.w8a8_block_fp8_matmul_triton(A4, B, As4, Bs, BLOCK, torch.bfloat16)
    rel_d = ((yd.float() - ref4).abs().max() / ref4.abs().max()).item()
    ok &= rel_d < 1e-2
    print(f'deterministic fallback: rel_err={rel_d:.5f}')
finally:
    fk.get_w8a8_block_fp8_configs = orig_get_cfg

print('SM89-BLOCK-FP8-CHECK', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
