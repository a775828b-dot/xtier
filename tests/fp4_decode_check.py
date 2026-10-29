"""FP4 decode of the expert cache kernels: the software path (RTX 30 / 40 series, no cvt.rn.f16x2.e2m1x2) against the
E2M1 table and, on GPUs that have the instruction (SM100 and newer), against the hardware path bit for bit: all 256
bytes, then the pool's gate/up and down kernels on random NVFP4 and MXFP4 slots (decode shapes of Qwen3.8, GLM-5.3 and
DeepSeek V4.1), run once with XTIER_FP4_DECODE=hw and once with =soft in child processes.

  python tests/fp4_decode_check.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
SHAPES = {'qwen38': (2560, 640, False), 'glm53': (4096, 2048, False), 'ds41': (5120, 2304, True)}


def worker(out):
    import torch
    import triton
    import triton.language as tl

    from xtier import arch
    from xtier.cache import pool

    @triton.jit
    def decode_bytes(SRC, LO, HI):
        i = tl.arange(0, 256)
        b = tl.load(SRC + i).to(tl.uint32)
        tl.store(LO + i, arch.e2m1_f16(b & 15).to(tl.uint16, bitcast=True))
        tl.store(HI + i, arch.e2m1_f16(b >> 4).to(tl.uint16, bitcast=True))

    dev = torch.device('cuda')
    src = torch.arange(256, dtype=torch.uint8, device=dev)
    lo = torch.empty(256, dtype=torch.uint16, device=dev)
    hi = torch.empty(256, dtype=torch.uint16, device=dev)
    decode_bytes[(1,)](src, lo, hi)
    result = {'lo': lo.cpu(), 'hi': hi.cpu(), 'hw': bool(arch.HW_FP4)}
    for name, (H, I, MX) in SHAPES.items():
        g = torch.Generator(device='cpu').manual_seed(20261029)
        S, E, m, k = 4, 8, 4, 2
        G = 32 if MX else 16
        q13 = torch.randint(0, 256, (S, 2 * I, H // 2), dtype=torch.uint8, generator=g).to(dev)
        q2 = torch.randint(0, 256, (S, H, I // 2), dtype=torch.uint8, generator=g).to(dev)
        lo_s, hi_s = (118, 128) if MX else (0x28, 0x48)   # UE8M0 2^-9..2^0 / E4M3 about 0.03..0.5
        s13 = torch.randint(lo_s, hi_s, (S, 2 * I, H // G), dtype=torch.uint8, generator=g).to(dev)
        s2 = torch.randint(lo_s, hi_s, (S, H, I // G), dtype=torch.uint8, generator=g).to(dev)
        g13 = (torch.rand((E, 2), generator=g) * 0.02 + 0.01).to(dev)
        g2 = (torch.rand((E,), generator=g) * 0.02 + 0.01).to(dev)
        x = torch.randn((m, H), generator=g).to(torch.bfloat16).to(dev)
        slot_list = torch.tensor([0, 1, 2, 3, 1, 2, 0, 3], dtype=torch.int32, device=dev)[: m * k]
        slot_exp = torch.tensor([5, 1, 7, 2, 1, 7, 5, 2], dtype=torch.int32, device=dev)[: m * k]
        mid = torch.zeros((pool.MAX_ROUTES, pool.MAX_M, I), dtype=torch.bfloat16, device=dev)
        part = torch.zeros((pool.MAX_ROUTES, pool.MAX_M, H), dtype=torch.float32, device=dev)
        cfg, nk = pool.CONFIG, m * k
        pool._gate_up[(nk, I // cfg['gbn'])](x, slot_list, slot_exp, q13, s13, g13, mid, m, H, I, 10.0 if MX else 0.0,
                                             pool.MAX_M, cfg['gbn'], pool._tile_k(H), MX=MX,
                                             num_warps=cfg['gw'], num_stages=cfg['gs'])
        pool._down[(nk, H // cfg['dbn'])](mid, slot_list, slot_exp, q2, s2, g2, part, m, H, I, pool.MAX_M, cfg['dbn'],
                                          pool._tile_k(I), MX=MX, num_warps=cfg['dw'], num_stages=cfg['ds'])
        torch.cuda.synchronize()
        result[name] = (mid[:nk, :m].cpu(), part[:nk, :m].cpu())
    torch.save(result, out)


def main():
    import torch
    with tempfile.TemporaryDirectory() as d:
        runs = {}
        for mode in ('soft', 'hw'):
            path = os.path.join(d, mode + '.pt')
            env = dict(os.environ, XTIER_FP4_DECODE=mode)
            r = subprocess.run([sys.executable, __file__, '--worker', path], env=env, capture_output=True, text=True)
            if r.returncode != 0:
                if mode == 'hw':
                    print('hardware path not available on this GPU:', (r.stderr.strip().splitlines() or ['?'])[-1][:200])
                    break
                print(r.stdout, r.stderr)
                raise SystemExit('FP4-DECODE-CHECK FAIL: software path did not run')
            runs[mode] = torch.load(path)
    soft = runs['soft']
    # 1. the software decode against the E2M1 table (sign in bit 3, -0.0 for code 8)
    want = []
    for b in range(256):
        for nib in (b & 15, b >> 4):
            v = E2M1[nib & 7] * (-1.0 if nib & 8 else 1.0)
            want.append(torch.tensor(v, dtype=torch.float16).view(torch.uint16).item())
    got = torch.stack([soft['lo'], soft['hi']], 1).reshape(-1).tolist()
    bad = sum(int(a != b) for a, b in zip(got, want))
    print(f'software E2M1 -> FP16 over 256 bytes: {bad} differences from the table')
    ok = bad == 0
    # 2. hardware against software, bit for bit
    if 'hw' in runs:
        hw = runs['hw']
        same = torch.equal(hw['lo'], soft['lo']) and torch.equal(hw['hi'], soft['hi'])
        print(f'hardware cvt.rn.f16x2.e2m1x2 vs software over 256 bytes: {"identical" if same else "DIFFERENT"}')
        ok &= same
        for name in SHAPES:
            (m_hw, p_hw), (m_sw, p_sw) = hw[name], soft[name]
            same = (torch.equal(m_hw.view(torch.int16), m_sw.view(torch.int16))
                    and torch.equal(p_hw.view(torch.int32), p_sw.view(torch.int32)))
            print(f'{name}: gate/up and down outputs, hardware vs software decode: '
                  f'{"bitwise identical" if same else "DIFFERENT"} (|part| max {p_sw.abs().max().item():.3g})')
            ok &= same
    print('FP4-DECODE-CHECK', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        main()
